#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune Llama 3.1 8B (or other HF causal LMs) on Psych-101 using
HF Transformers + PEFT (QLoRA) + TRL SFTTrainer.

- 4-bit quantization with bitsandbytes
- QLoRA (rank r=8, applied to attention + FFN linear layers)
- Completion-only training with loss masked to human responses
  via DataCollatorForCompletionOnlyLM and "<<", ">>" tags.

Example usage:

python finetune_psych101_hf.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./outputs/psych101_llama3_8b_qlora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --optim adamw_bnb_8bit \
  --warmup_steps 100 \
  --lr_scheduler_type linear \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --bf16 True
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)


# ========================
# 1. Argument dataclasses
# ========================

@dataclass
class ModelArguments:
    """Arguments for model and LoRA configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "HF model name or local path to base causal LM."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank r (paper: r=8)."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (scaling). You can adjust; paper only specifies r=8."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout. Paper uses no dropout."},
    )
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Linear layers in attention + FFN to apply LoRA to."},
    )


@dataclass
class DataArguments:
    """Arguments for data loading and text processing."""
    train_dataset_name: str = field(
        default="marcelbinz/Psych-101",
        metadata={"help": "HF dataset for training."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Split name for training."},
    )
    eval_dataset_name: str = field(
        default="marcelbinz/Psych-101-test",
        metadata={"help": "HF dataset for evaluation."},
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Split name for evaluation."},
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the text column in the dataset."},
    )
    max_seq_length: int = field(
        default=32768,
        metadata={"help": "Maximum sequence length (paper: 32k context)."},
    )


# ========================
# 2. Model preparation
# ========================

def get_bnb_config():
    """Create a 4-bit BitsAndBytesConfig suitable for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # change to torch.bfloat16 if desired
        bnb_4bit_quant_type="nf4",
    )


def prepare_tokenizer(model_name_or_path: str):
    """Load tokenizer and set padding configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Use EOS as PAD if PAD is not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_model(model_args: ModelArguments, training_args: TrainingArguments):
    """Load 4-bit base model and attach LoRA adapters."""
    bnb_config = get_bnb_config()

    # Load base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",  # can be changed to "cuda" for single-GPU
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config (QLoRA)
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Attach LoRA
    model = get_peft_model(model, lora_config)

    # Optional: gradient checkpointing (helps memory)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Print trainable params (for sanity check)
    model.print_trainable_parameters()

    return model


# ========================
# 3. Main training logic
# ========================

def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):

    # Set seed
    set_seed(training_args.seed)

    # Load datasets
    train_dataset = load_dataset(
        data_args.train_dataset_name,
        split=data_args.train_split,
    )
    eval_dataset = load_dataset(
        data_args.eval_dataset_name,
        split=data_args.eval_split,
    )

    # Load tokenizer & model
    tokenizer = prepare_tokenizer(model_args.model_name_or_path)
    model = prepare_model(model_args, training_args)

    # Completion-only collator:
    # Assume the dataset text uses " <<" before human response
    # and ">>" after, following the original Unsloth script.
    response_template = tokenizer(" <<").input_ids[1:]  # drop BOS
    instruction_template = tokenizer(">>").input_ids[1:]  # drop BOS

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
    )

    # ===== Hyperparameter defaults adjusted to paper-style recipe =====
    # If user didn't override them on CLI / config, adjust a few defaults here.
    # (Transformers TrainingArguments already defaults learning_rate=5e-5.)

    if training_args.num_train_epochs is None or training_args.num_train_epochs == 3.0:
        # Transformers default is 3.0; paper uses 1 epoch.
        training_args.num_train_epochs = 1.0

    if training_args.per_device_train_batch_size == 8:
        # Default is 8; reduce to 4 so that with grad_acc=8 -> effective batch=32.
        training_args.per_device_train_batch_size = 4

    if training_args.gradient_accumulation_steps == 1:
        training_args.gradient_accumulation_steps = 8  # 4 * 8 = 32 effective batch

    if training_args.weight_decay == 0.0:
        training_args.weight_decay = 0.01

    if training_args.warmup_steps == 0:
        training_args.warmup_steps = 100

    if training_args.optim == "adamw_torch":
        training_args.optim = "adamw_bnb_8bit"

    if training_args.lr_scheduler_type == "linear":
        # already linear; just keep. If it's not set, user can override via CLI.
        pass

    # Use bf16 or fp16 according to hardware (if not explicitly set)
    if not training_args.fp16 and not training_args.bf16:
        # Heuristic: if GPU capability >= 8.0, bf16 is usually supported.
        if torch.cuda.is_available():
            major_cc, _ = torch.cuda.get_device_capability()
            if major_cc >= 8:
                training_args.bf16 = True
            else:
                training_args.fp16 = True

    # Create SFTTrainer (TRL)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        data_collator=collator,
        args=training_args,
    )

    # (SFTTrainer에는 print_trainable_parameters 메서드가 없으니 호출하지 않음)

    # Train
    trainer.train()

    # Save final adapter (and tokenizer)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load arguments from a JSON config file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=sys.argv[1]
        )
    else:
        # Parse arguments from CLI
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)
