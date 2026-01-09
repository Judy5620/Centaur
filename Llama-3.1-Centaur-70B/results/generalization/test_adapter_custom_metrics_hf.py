# /home/work/user12/projects/Llama-3.1-Centaur-70B/generalization/test_adapter_custom_metrics_hf.py
# - HF + PEFT only (no unsloth/4bit/xformers)
# - per-task evaluate using the SAME metric logic you attached
# - OOM-safety: batch=1, SDPA, use_cache=False, CPU aggregation

import os, math, argparse
import torch, pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# === EXACTLY the task list from your attached test_adapter_custom_metrics.py ===
TASK_NAMES = [
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

# === same metric logic you attached, adapted to HF ===
def preprocess_logits_for_metrics(logits, labels):
    # logits: (batch=1, seq, vocab) or tuple(...)
    with torch.no_grad():
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # move to CPU early (to avoid GPU OOM when aggregating)
        logits = logits.detach().float().cpu()
        labels = labels.detach().cpu()

        # align labels to logits like your original code
        # labels shape: (1, seq) -> shift left by 1 & pad -100 at end
        labels = labels[0]
        labels = torch.cat((labels[1:], -100 * torch.ones(1, dtype=torch.long)), dim=0)

        # ce per token
        ce = torch.nn.functional.cross_entropy(logits[0], labels, reduction="none")  # (seq,)

        # accumulate until we see -100 (your "item boundary")
        total_loss = []
        item_loss = 0.0
        item_counter = 0
        for i in range(ce.shape[0]):
            if labels[i] != -100:
                item_loss += ce[i].item()
                item_counter += 1
            else:
                if item_counter != 0:
                    total_loss.append(item_loss / item_counter)  # normalize per item (more stable)
                    item_loss = 0.0
                    item_counter = 0
        if item_counter != 0:  # tail (robustness)
            total_loss.append(item_loss / max(item_counter, 1))

        # return a 1D tensor (trainer will concatenate on CPU)
        return torch.tensor(total_loss, dtype=torch.float32)

def compute_metrics(pred):
    # pred.predictions is the concatenated CPU array of per-item losses
    import numpy as np
    arr = pred.predictions
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    arr = np.array(arr, dtype=float).reshape(-1)
    return {"custom_loss": float(arr.mean()) if arr.size > 0 else float("nan")}

def load_model_and_tok(adapter_id: str, base_id: str, dtype: str = "bfloat16"):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=HF_HOME,
    )
    model = PeftModel.from_pretrained(base, adapter_id, cache_dir=HF_HOME)

    # OOM-safety / stability
    model.config.use_cache = False
    try:
        model.config.attn_implementation = "sdpa"  # good kernel without xformers
    except Exception:
        setattr(model.config, "_attn_implementation", "sdpa")

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, cache_dir=HF_HOME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.config.pad_token_id = tok.eos_token_id

    # same response/instruction templates
    l_id = tok(" <<").input_ids[1:]
    r_id = tok(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tok
    )
    return model, tok, collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter repo id, e.g. marcelbinz/Llama-3.1-Centaur-8B-adapter")
    ap.add_argument("--base",  required=True, help="base repo id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bfloat16","float16"], default="bfloat16")
    args = ap.parse_args()

    model, tokenizer, collator = load_model_and_tok(args.model, args.base, args.dtype)
    dataset = load_dataset("marcelbinz/Psych-101-test")
    model_dash = args.model.replace("/", "-")

    rows = []
    with torch.no_grad():
        for task_name in TASK_NAMES:
            eval_dataset = dataset["test"].filter(lambda ex: str(ex.get("experiment","")).startswith(task_name))
            if len(eval_dataset) == 0:
                print(f"[WARN] No samples for task: {task_name}")
                continue

            training_args = TrainingArguments(
                output_dir=f"eval_custom_{task_name}",
                per_device_eval_batch_size=1,
                eval_accumulation_steps=1,
                report_to="none",
                # prediction_loss_only=False  # we need predictions for custom metric
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=eval_dataset,   # required by SFTTrainer; no training occurs
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=args.max_len,
                data_collator=collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )

            result = trainer.evaluate()
            print(task_name, result, flush=True)
            rows.append([task_name, float(result["eval_custom_loss"])])

    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame(rows, columns=["task", model_dash])
    out_path = f"results/custom_metrics_task_{model_dash}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
