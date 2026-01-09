#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a QLoRA adapter on selected Psych-101 tasks with custom metrics
using pure Hugging Face Transformers + PEFT (no Unsloth).

- Loads a base causal LM in 4-bit (bitsandbytes)
- Loads a PEFT QLoRA adapter on top
- Uses DataCollatorForCompletionOnlyLM + manual evaluation loop
- Uses custom preprocess_logits_for_metrics
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
)
from peft import PeftModel
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader


# ====== 1. Custom metrics ======

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


# ====== 2. QLoRA 모델 로딩 (HF + PEFT) ======

def load_qlora_model(
    base_model_name: str,
    adapter_path: str,
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # SDPA 이슈 피하기용
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
        elif hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    model.eval()
    return model, tokenizer


# ====== 3. 메인 ======

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QLoRA adapter with custom metrics on Psych-101 tasks (HF + PEFT, manual loop)."
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
        default=2048,
        help="Maximum sequence length for tokenization (default: 2048)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Eval batch size (default: 1) — custom loss 구현이 batch=1 가정.",
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

    run_name = args.run_name
    if run_name is None:
        run_name = os.path.basename(os.path.normpath(args.adapter_path))
        if not run_name:
            run_name = "qlora_custom_metrics"

    os.makedirs(args.results_dir, exist_ok=True)

    print("=== Loading dataset: marcelbinz/Psych-101-test ===", flush=True)
    dataset = load_dataset("marcelbinz/Psych-101-test")

    print("=== Loading model + QLoRA adapter ===", flush=True)
    model, tokenizer = load_qlora_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
    )

    # Completion-only collator (" <<", ">>" 템플릿 사용)
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_rows = []

    for task_name in task_names:
        print(f"\n=== Evaluating task: {task_name} ===", flush=True)

        # 1) 해당 task만 필터
        raw_eval_dataset = dataset["test"].filter(
            lambda example: example["experiment"].startswith(task_name)
        )

        num_examples = len(raw_eval_dataset)
        if num_examples == 0:
            print(f"[WARN] No examples found for task '{task_name}', skipping.", flush=True)
            continue

        print(f"Number of examples: {num_examples}", flush=True)

        # 2) 토크나이즈 (★ 여기 없어서 방금 에러가 난 것)
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )

        tokenized_eval = raw_eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_eval_dataset.column_names,  # text/experiment/participant 제거
        )
        tokenized_eval.set_format(type="torch")

        # 3) DataLoader + collator
        eval_loader = DataLoader(
            tokenized_eval,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        all_item_losses = []

        # 4) 수동 루프 + custom loss 계산
        for batch in eval_loader:
            # collator가 이미 tensor로 만들어줌
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                labels = batch["labels"]

                item_losses = preprocess_logits_for_metrics(logits, labels)
                if item_losses.numel() > 0:
                    all_item_losses.append(item_losses)

        if len(all_item_losses) == 0:
            custom_loss = float("nan")
        else:
            all_item_losses = torch.cat(all_item_losses, dim=0)
            custom_loss = all_item_losses.mean().item()

        print(f"[RESULT] task={task_name}, custom_loss={custom_loss}", flush=True)
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
