#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a QLoRA adapter on selected Psych-101 tasks with custom metrics
using pure Hugging Face Transformers + PEFT (no Unsloth).

- Loads a base causal LM in 4-bit (bitsandbytes)
- Loads a PEFT QLoRA adapter on top
- Uses TRL's SFTTrainer + DataCollatorForCompletionOnlyLM
- Uses custom preprocess_logits_for_metrics + compute_metrics
- Writes a CSV with custom_loss per task.
"""

import os
import argparse
from typing import List

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


# ====== 1. Custom metrics (원래 코드 그대로) ======

def preprocess_logits_for_metrics(logits, labels):
    """
    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)

    배치크기 1을 가정하고, -100 토큰을 기준으로 각 item별 loss를 쪼갬.
    """
    with torch.no_grad():
        logits = logits.cpu()
        labels = labels.cpu()

        # batch dim 제거 (batch=1 가정)
        # labels: (1, T) -> (T-1,) ; 마지막 one-step shift
        labels = torch.cat(
            (labels[0, 1:], -100 * torch.ones(1).long()),
            dim=0,
        )
        logits = logits[0]  # (T, vocab_size)

        ce = torch.nn.functional.cross_entropy(
            logits, labels, reduction="none"
        )  # (T,)

        total_loss = []
        item_loss = 0.0
        item_counter = 0
        for i in range(ce.shape[0]):
            if labels[i] != -100:
                item_loss += ce[i]
                item_counter += 1
            else:
                if item_counter != 0:
                    total_loss.append(item_loss)
                    item_loss = 0.0
                    item_counter = 0

        if len(total_loss) == 0:
            return torch.tensor([])

        return torch.tensor(total_loss)


def compute_metrics(pred):
    """
    pred.predictions : preprocess_logits_for_metrics가 반환한 텐서들을
    하나로 모은 것 → 여기서 평균 custom loss 계산.
    """
    print("pred.predictions shape:", pred.predictions.shape, flush=True)
    return {"custom_loss": pred.predictions.mean().item()}


# ====== 2. QLoRA 모델 로딩 (HF + PEFT) ======

def load_qlora_model(
    base_model_name: str,
    adapter_path: str,
    max_seq_length: int = 4096,
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
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # ★ SDPA 대신 eager attention 사용 (device-side assert 완화)
    )

    # attach QLoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
    )

    # 캐시 비활성화해서 메모리 사용량 조금 줄이기
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        # config에도 eager 명시 (모델 내부에서 참조할 수 있음)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
        elif hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    model.eval()
    return model, tokenizer


# ====== 3. 메인 ======

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QLoRA adapter with custom metrics on Psych-101 tasks (HF + PEFT)."
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
        "--results_dir",
        type=str,
        default="results",
        help="Directory where the CSV result file will be written (default: results)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this evaluation run (used in CSV filename and column). "
             "Default: basename of adapter_path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length for SFTTrainer (default: 4096; 2048~4096 추천)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="per_device_eval_batch_size (default: 1)",
    )
    args = parser.parse_args()

    # task 목록 (원래 코드 그대로)
    task_names: List[str] = [
        "collsiöö2023MCPL",
        "cox2017information",
        "garcia2023experiential",
        "jansen2021dunningkruger",
        "krueger2022identifying",
        "kumar2023disentangling",
        "popov2023intent",
        "wise2019acomputational",
        "wu2018generalisation",
        "zhu2020bayesian",
    ]

    # run_name 설정
    run_name = args.run_name
    if run_name is None:
        run_name = os.path.basename(os.path.normpath(args.adapter_path))
        if not run_name:
            run_name = "qlora_custom_metrics"

    # results 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)

    print("=== Loading dataset: marcelbinz/Psych-101-test ===", flush=True)
    dataset = load_dataset("marcelbinz/Psych-101-test")

    print("=== Loading model + QLoRA adapter ===", flush=True)
    model, tokenizer = load_qlora_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    # Completion-only collator (" <<", ">>" 템플릿 사용)
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )

    results_rows = []

    # ★ evaluate 내부에서 이미 no_grad를 쓰므로, 여기서는 감싸지 않음
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

        # task별 임시 출력 디렉토리
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
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        result = trainer.evaluate()
        print(task_name, flush=True)
        print(result, flush=True)

        # Trainer가 metric key에 "eval_" prefix를 붙임
        custom_loss = result.get("eval_custom_loss", float("nan"))
        results_rows.append([task_name, num_examples, custom_loss])

    if not results_rows:
        print("No results to save (no tasks evaluated).", flush=True)
        return

    df = pd.DataFrame(results_rows, columns=["task", "num_examples", run_name])
    print("\n=== Final results ===", flush=True)
    print(df, flush=True)

    csv_path = os.path.join(
        args.results_dir,
        f"custom_metrics_{run_name}.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
