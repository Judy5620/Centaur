#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a base model (optionally with a QLoRA adapter) on
marcelbinz/Psych-101-test using HF Transformers + PEFT + TRL.

- If adapter_path is given: load base model in 4bit + QLoRA adapter (PEFT)
- If adapter_path is None/empty: load base model in bf16/fp16 (no quantization)
- Uses SFTTrainer + DataCollatorForCompletionOnlyLM
- Computes eval_loss per experiment (task) and writes a CSV.
"""

import os
import argparse
from typing import List, Optional

import torch
import pandas as pd
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_model(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    max_seq_length: int = 32768,
):
    """
    Load model for evaluation.

    - If adapter_path is provided: 4bit quantized base + QLoRA adapter
    - If adapter_path is None/empty: non-quantized base model (bf16/fp16)
      → 이렇게 해야 Trainer가 "양자화 + no adapter"라고 오해하지 않음.
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1) QLoRA 경로가 있는 경우: 4bit + PEFT
    if adapter_path is not None and adapter_path.strip() != "":
        print(f"[INFO] Loading base model in 4bit + PEFT adapter from: {adapter_path}", flush=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # A100이면 bf16
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    # 2) 어댑터가 없는 경우: non-quantized base model (bf16/fp16)
    else:
        print("[INFO] No adapter_path given → evaluating pure base model only (non-quantized).", flush=True)

        # bf16 지원 여부 체크
        dtype = torch.float16
        if torch.cuda.is_available():
            major_cc, _ = torch.cuda.get_device_capability()
            if major_cc >= 8:  # A100 등
                dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    # 캐시 비활성화 (gradient checkpointing 대비 & 메모리 감소)
    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()
    return model, tokenizer


def build_task_list(
    dataset,
    task_list_path: Optional[str] = None,
) -> List[str]:
    """
    Decide which tasks (experiments) to evaluate.

    - If task_list_path is given: load one task name per line.
    - Else: use all unique values of dataset['test']['experiment'].
    """
    if task_list_path is not None:
        with open(task_list_path, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip()]
        return tasks

    experiments = dataset["test"]["experiment"]
    unique_experiments = sorted(set(experiments))
    return unique_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base model (and optional QLoRA adapter) on marcelbinz/Psych-101-test."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base causal LM (HF hub id or local path), e.g. meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help=(
            "Optional: Path or HF id of QLoRA adapter (PEFT), "
            "e.g. /path/to/minitaur_adapter or marcelbinz/Llama-3.1-Centaur-8B-adapter. "
            "If omitted or empty, evaluate base model only."
        ),
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default=None,
        help=(
            "Optional: path to a text file with one task (experiment) name per line. "
            "If not provided, all unique dataset['test']['experiment'] values are used."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where the CSV result file will be written (default: results)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Name for this evaluation run (used as CSV column name and filename). "
            "Default: basename of adapter_path if given, otherwise derived from base_model."
        ),
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32768,
        help="Maximum sequence length for SFTTrainer (default: 32768; 실사용 시 2048~4096 추천)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="per_device_eval_batch_size (default: 1)",
    )
    args = parser.parse_args()

    # run_name 결정
    run_name = args.run_name
    if run_name is None:
        if args.adapter_path is not None and args.adapter_path.strip() != "":
            run_name = os.path.basename(os.path.normpath(args.adapter_path)) or "qlora_run"
        else:
            safe_base = args.base_model.replace("/", "_")
            run_name = f"{safe_base}_base"

    os.makedirs(args.results_dir, exist_ok=True)

    print("=== Loading dataset: marcelbinz/Psych-101-test ===", flush=True)
    dataset = load_dataset("marcelbinz/Psych-101-test")

    print("=== Building task list ===", flush=True)
    task_names = build_task_list(dataset, args.task_list)
    print(f"Number of tasks: {len(task_names)}", flush=True)
    print("Tasks:", task_names, flush=True)

    print("=== Loading model (base + optional QLoRA adapter) ===", flush=True)
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    # Completion-only collator (" <<" ~ ">>" 사이만 loss)
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )

    results_data = []

    for task_name in task_names:
        print(f"\n=== Evaluating task: {task_name} ===", flush=True)

        eval_dataset = dataset["test"].filter(
            lambda example: example["experiment"].startswith(task_name)
        )

        num_examples = len(eval_dataset)
        if num_examples == 0:
            print(f"[WARN] No examples found for task '{task_name}', skipping.", flush=True)
            continue

        print(f"Number of examples: {num_examples}", flush=True)

        task_output_dir = os.path.join(args.results_dir, f"tmp_{task_name}")
        os.makedirs(task_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=task_output_dir,
            per_device_eval_batch_size=args.batch_size,
            do_train=False,
            do_eval=True,
            logging_steps=50,
            bf16=True,
            fp16=False,
            eval_accumulation_steps=1,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=eval_dataset,   # SFTTrainer 시그니처 맞추기용 (실제로는 train 안 함)
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            data_collator=collator,
        )

        result = trainer.evaluate()
        eval_loss = result.get("eval_loss", float("nan"))

        print(f"[RESULT] task={task_name}, eval_loss={eval_loss}", flush=True)
        results_data.append(
            {
                "task": task_name,
                "num_examples": num_examples,
                run_name: eval_loss,
            }
        )

    if not results_data:
        print("No results to save (no tasks evaluated).", flush=True)
        return

    df = pd.DataFrame(results_data).set_index("task")
    print("\n=== Final results ===", flush=True)
    print(df, flush=True)

    csv_path = os.path.join(args.results_dir, f"psych101_eval_{run_name}.csv")
    df.to_csv(csv_path)
    print(f"\nSaved results to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
