# /home/work/user12/projects/Llama-3.1-Centaur-70B/generalization/generalization_custom_metrics_hf.py
# HF + PEFT only. Memory-safe generalization evaluate (file-per-split).

import os, json, math, argparse
import torch, pandas as pd
from datasets import Dataset, Features, Value
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def robust_jsonl_to_dataset(fp: str) -> Dataset:
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    # LFS 포인터 감지
    try:
        if os.path.getsize(fp) < 64:
            head = open(fp, "r", encoding="utf-8", errors="ignore").read(200)
            if head.startswith("version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{fp} looks like a Git LFS pointer. Run `git lfs pull` first.")
    except Exception:
        pass

    rows = []
    with open(fp, "r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                text = None
                if isinstance(obj, dict):
                    text = obj.get("text") or obj.get("prompt")
                    if not text and isinstance(obj.get("messages"), list):
                        parts = [str(m.get("content","")) for m in obj["messages"]]
                        text = "\n".join(parts).strip()
                    if not text:
                        text = json.dumps(obj, ensure_ascii=False)
                elif isinstance(obj, str):
                    text = obj
                else:
                    text = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                text = line
            if text:
                rows.append({"text": text})
    if not rows:
        raise ValueError(f"No usable lines in {fp}")
    return Dataset.from_list(rows, features=Features({"text": Value("string")}))

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

    model.config.use_cache = False
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        setattr(model.config, "_attn_implementation", "sdpa")

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, cache_dir=HF_HOME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.config.pad_token_id = tok.eos_token_id

    l_id = tok(" <<").input_ids[1:]
    r_id = tok(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tok
    )
    return model, tok, collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base",  required=True)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bfloat16","float16"], default="bfloat16")
    ap.add_argument("--files", nargs="*", default=[
        "jansen2021logic/prompts.jsonl"   # → gen0
    ])
    args = ap.parse_args()

    model, tok, collator = load_model_and_tok(args.model, args.base, args.dtype)
    model_dash = args.model.replace("/", "-")
    os.makedirs("results", exist_ok=True)

    for i, fp in enumerate(args.files):
        tag = f"gen{i}"
        test = robust_jsonl_to_dataset(fp)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tok,
            args=TrainingArguments(
                output_dir=f"eval_{tag}_custom",
                per_device_eval_batch_size=1,
                report_to="none",
                prediction_loss_only=True,  # ✅ logits 저장 안 함
            ),
            train_dataset=test,
            eval_dataset=test,
            dataset_text_field="text",
            max_seq_length=args.max_len,
            data_collator=collator,
        )

        out = trainer.evaluate()
        eval_loss = float(out.get("eval_loss"))
        eval_ppl  = math.exp(eval_loss)

        df = pd.DataFrame(
            [[tag, eval_loss, eval_ppl]],
            columns=["task", f"{model_dash}_loss", f"{model_dash}_ppl"]
        )
        out_path = f"results/revised_custom_metrics_{tag}_{model_dash}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{tag}] loss={eval_loss:.6f}, ppl={eval_ppl:.3f}")
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
