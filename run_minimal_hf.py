#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HF Transformers + PEFT version of run_minimal.py
(Original Unsloth example rewritten for our own QLoRA adapter)

This script:
1. Loads base model (Llama 3.1 8B Instruct)
2. Loads our QLoRA adapter (optional)
3. Runs inference on triplet input
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_model(base_model, adapter_path=None):
    """Load base model in 16-bit + apply our PEFT QLoRA adapter if provided."""
    print("\n=== Loading base model ===")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path is not None:
        print(f"=== Loading adapter from {adapter_path} ===")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("=== No adapter provided → using pure base model ===")

    model.eval()
    return model


def load_tokenizer(base_model, adapter_path=None):
    """Tokenizer is loaded from adapter if exists (it contains added special tokens)."""
    if adapter_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_inference(model, tokenizer, prompt, max_new_tokens=1):
    """Generate model output."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False
    ).to(model.device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    print("\n=== Model output ===")
    print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.base_model, args.adapter_path)
    model = load_model(args.base_model, args.adapter_path)

    run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
