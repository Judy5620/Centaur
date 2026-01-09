#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a QLoRA adapter on the marcelbinz/Psych-101-test dataset
using pure Hugging Face Transformers + PEFT (no Unsloth).

- Loads a base causal LM in 4-bit (bitsandbytes)
- Loads a PEFT QLoRA adapter on top
- Uses TRL's SFTTrainer + DataCollatorForCompletionOnlyLM
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


def load_qlora_model(
    base_model_name: str,
    adapter_path: str,
    max_seq_length: int = 32768,
):
    """
    Load a base causal LM in 4-bit and attach a QLoRA adapter (PEFT).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100이면 bfloat16 추천
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # base model in 4-bit, placed automatically across GPUs
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # attach QLoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
    )

    # 캐시 비활성화해서 메모리 사용량 조금 줄이기
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
    - Else: use all unique values of dataset["test"]["experiment"].
    """
    if task_list_path is not None:
        with open(task_list_path, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip()]
        return tasks

    # No external task list: derive from dataset
    experiments = dataset["test"]["experiment"]
    unique_experiments = sorted(set(experiments))
    return unique_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QLoRA adapter on marcelbinz/Psych-101-test (HF + PEFT)."
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
        required=True,
        help="Path or HF id of QLoRA adapter (PEFT), e.g. marcelbinz/Llama-3.1-Centaur-8B-adapter",
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default=None,
        help="Optional: path to a text file with one task (experiment) name per line. "
             "If not provided, all unique dataset['test']['experiment'] values are used.",
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
        help="Name for this evaluation run (used as CSV column name and filename). "
             "Default: basename of adapter_path.",
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

    # Decide run_name
    run_name = args.run_name
    if run_name is None:
        run_name = os.path.basename(os.path.normpath(args.adapter_path))
        if not run_name:
            run_name = "qlora_run"

    # Make results dir
    os.makedirs(args.results_dir, exist_ok=True)

    print("=== Loading dataset: marcelbinz/Psych-101-test ===", flush=True)
    dataset = load_dataset("marcelbinz/Psych-101-test")

    print("=== Building task list ===", flush=True)
    task_names = build_task_list(dataset, args.task_list)
    print(f"Number of tasks: {len(task_names)}", flush=True)
    print("Tasks:", task_names, flush=True)

    print("=== Loading model + QLoRA adapter ===", flush=True)
    model, tokenizer = load_qlora_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    # Build completion-only collator
    # 템플릿은 원래 Unsloth 코드와 동일하게 " <<", ">>" 토큰 기준
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )

    results_data = []

    # 메인 루프: 각 task(=experiment 이름)에 대해 eval_loss 계산
    for task_name in task_names:
        print(f"\n=== Evaluating task: {task_name} ===", flush=True)

        # experiment 필드가 task_name으로 시작하는 예제만 사용
        # (원래 Binz 코드와 같은 startswith 로직)
        eval_dataset = dataset["test"].filter(
            lambda example: example["experiment"].startswith(task_name)
        )

        num_examples = len(eval_dataset)
        if num_examples == 0:
            print(f"[WARN] No examples found for task '{task_name}', skipping.", flush=True)
            continue

        print(f"Number of examples: {num_examples}", flush=True)

        # Temporary output dir per task (logs etc.)
        task_output_dir = os.path.join(args.results_dir, f"tmp_{task_name}")
        os.makedirs(task_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=task_output_dir,
            per_device_eval_batch_size=args.batch_size,
            do_train=False,
            do_eval=True,
            logging_steps=50,
            bf16=True,                 # A100 → bf16
            fp16=False,                # 명시적으로 fp16 끄기
            eval_accumulation_steps=1, # 메모리 폭주 방지용
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=eval_dataset,   # 실제 train 안 하지만 SFTTrainer 시그니처 맞추기용
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

    # Save as CSV
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
