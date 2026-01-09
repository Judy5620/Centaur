import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, set_seed
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth import FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)

@dataclass
class DataTrainingArguments:
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=32768)

def main(model_args, data_args, training_args):
    # set seed for reproducibility
    set_seed(training_args.seed)

    # datasets
    train_dataset = load_dataset("marcelbinz/Psych-101")['train'].shuffle()
    eval_datasets = load_dataset("marcelbinz/Psych-101-test")['test']

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = data_args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj",
            "up_proj", "down_proj",
        ],
        lora_alpha = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = training_args.seed,
        use_rslora = True,
        loftq_config = None,
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)

    # trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=eval_datasets,
        dataset_text_field = data_args.dataset_text_field,
        max_seq_length = data_args.max_seq_length,
        dataset_num_proc = 8,
        data_collator=collator,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            per_device_eval_batch_size =  training_args.per_device_eval_batch_size,
            gradient_accumulation_steps = training_args.gradient_accumulation_steps,
            warmup_steps = training_args.warmup_steps,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            embedding_learning_rate = training_args.learning_rate / 10,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            log_level = training_args.log_level,
            logging_strategy = training_args.logging_strategy,
            logging_steps = training_args.logging_steps,
            evaluation_strategy = training_args.evaluation_strategy,
            eval_steps = training_args.eval_steps,
            save_strategy = training_args.save_strategy,
            save_steps = training_args.save_steps,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            seed = training_args.seed,
            output_dir = training_args.output_dir,
        ),
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # train
    trainer.train(resume_from_checkpoint=None)

    # saving final model
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
