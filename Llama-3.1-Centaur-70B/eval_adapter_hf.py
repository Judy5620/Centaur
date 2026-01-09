# eval_adapter_hf.py
# ---------------------------------------------------------
# 목적:
# - 4bit / Unsloth / xformers 없이, 순정 HF + PEFT로 어댑터 평가
# - Psych-101(참가자 일반화)와 일반화(JSONL 프롬프트) 모두 지원
# - 결과는 results/ 아래 CSV로 저장
#
# 사용 예:
#   export HF_HOME=~/.cache/huggingface
#   CUDA_VISIBLE_DEVICES=0 python eval_adapter_hf.py \
#     --model marcelbinz/Llama-3.1-Centaur-8B-adapter \
#     --base meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --mode psych101
#
#   CUDA_VISIBLE_DEVICES=0 python eval_adapter_hf.py \
#     --model marcelbinz/Llama-3.1-Centaur-8B-adapter \
#     --base meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --mode generalization
# ---------------------------------------------------------

import argparse
import os
import sys
import json
import torch
import pandas as pd

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from datasets import load_dataset, Dataset, Features, Value
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def _jsonl_to_text_dataset(fp: str) -> Dataset:
    """prompts.jsonl을 유연하게 읽어서 'text' 컬럼 하나짜리 Dataset으로 반환
    - UTF-8 BOM 처리
    - 빈 줄/주석(#, //) 무시
    - 각 줄 JSON 파싱 실패 시 그 줄을 그대로 text로 사용(폴백)
    - 파일이 비어 있으면 명확한 에러
    """
    import json
    from datasets import Dataset, Features, Value

    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")
    if os.path.getsize(fp) == 0:
        raise ValueError(f"Input file is empty: {fp}")

    texts = []
    with open(fp, "r", encoding="utf-8-sig") as f:  # utf-8 BOM 안전
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 주석 라인 스킵
            if line.startswith("#") or line.startswith("//"):
                continue

            item_text = None
            try:
                obj = json.loads(line)
                # 우선순위: text -> prompt -> instruction(+input) -> messages -> 임의의 문자열 값
                if isinstance(obj, dict):
                    if "text" in obj and isinstance(obj["text"], str):
                        item_text = obj["text"]
                    elif "prompt" in obj and isinstance(obj["prompt"], str):
                        item_text = obj["prompt"]
                    elif "instruction" in obj:
                        inst = obj.get("instruction") or ""
                        inp = obj.get("input") or ""
                        item_text = (inst + ("\n" + inp if inp else "")).strip()
                    elif "messages" in obj and isinstance(obj["messages"], list):
                        parts = []
                        for m in obj["messages"]:
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            parts.append(f"{role}: {content}")
                        item_text = "\n".join(parts)
                    else:
                        # dict이긴 한데 문자열 값이 없다면, dict 자체를 문자열화
                        for k, v in obj.items():
                            if isinstance(v, str):
                                item_text = v
                                break
                        if item_text is None:
                            item_text = json.dumps(obj, ensure_ascii=False)
                elif isinstance(obj, str):
                    item_text = obj
                else:
                    # 리스트/숫자 등은 문자열화
                    item_text = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                # JSON이 아니면 라인 전체를 텍스트로 사용
                item_text = line

            if item_text:
                texts.append({"text": item_text})

    if not texts:
        raise ValueError(f"No usable lines found in: {fp}")

    feats = Features({"text": Value("string")})
    return Dataset.from_list(texts, features=feats)


def load_adapter_or_base(model_name: str, base: str | None, dtype: str = "bfloat16"):
    """
    - base를 순정 HF 체크포인트로 먼저 로드
    - 그 위에 어댑터(LoRA)를 PeftModel로 주입
    - 어댑터 내부의 4bit/unsloth/bnb 설정을 전혀 건드리지 않음(근본 원인 우회)
    """
    torch_dtype = torch.bfloat16 if dtype.lower() == "bfloat16" else torch.float16

    if base is None:
        # 반드시 베이스를 명시해서 Unsloth/bnb-4bit 경로를 방지
        base = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # 1) 순정 베이스 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=HF_HOME,
    )

    # 2) LoRA 어댑터 주입
    model = PeftModel.from_pretrained(
        base_model,
        model_name,               # ex) "marcelbinz/Llama-3.1-Centaur-8B-adapter"
        cache_dir=HF_HOME,
    )

    tok = AutoTokenizer.from_pretrained(base, use_fast=True, cache_dir=HF_HOME)

    # ✅ Llama 계열은 pad 토큰이 없음 → EOS를 pad로 사용
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # 평가/학습 모두 안전
    model.config.pad_token_id = tok.eos_token_id

    # 응답 템플릿 토큰(<<" <<, >>)
    l_id = tok(" <<").input_ids[1:]
    r_id = tok(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tok
    )
    return model, tok, collator


def run_eval_split(args, dataset, tag: str, tasks: list[str] | None):
    """
    - tasks가 주어지면 experiment prefix별로 필터링하여 각각 평가
    - 아니면 dataset['test'] 전체를 평가
    - 결과를 results/{tag}_{model-dashed}.csv 로 저장
    """
    data = []
    model_dash = args.model.replace("/", "-")

    for name in (tasks or [None]):
        if name is None:
            eval_dataset = dataset["test"]
            task_label = tag
        else:
            # experiment 필드 prefix로 필터 (없으면 빈 문자열)
            eval_dataset = dataset["test"].filter(lambda ex: ex.get("experiment", "").startswith(name))
            task_label = name

        trainer = SFTTrainer(
            model=args.model_ref,
            tokenizer=args.tok,
            args=TrainingArguments(
                output_dir=f"eval_{tag}",
                per_device_eval_batch_size=1,
                report_to="none",
            ),
            train_dataset=eval_dataset,  # SFTTrainer 요구사항상 넣지만 학습은 안 함
            eval_dataset=eval_dataset,
            dataset_text_field="text",   # TRL 0.9.x 경고는 무시 가능
            max_seq_length=args.max_len,
            data_collator=args.collator,
        )
        out = trainer.evaluate()
        data.append([task_label, out["eval_loss"]])
        print(f"[{tag}] {task_label}: {out}", flush=True)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(data, columns=["task", model_dash])
    out_path = f"results/{tag}_{model_dash}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Adapter repo or full model id")
    parser.add_argument("--base", default=None, help="Base model repo id (gated model은 로그인 필요)")
    parser.add_argument("--mode", choices=["psych101", "generalization"], default="psych101")
    parser.add_argument("--max_len", type=int, default=16384)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    # 모델/토크나이저/콜레이터 로드
    print(f"Loading base={args.base} + adapter={args.model} (dtype={args.dtype}) ...")
    args.model_ref, args.tok, args.collator = load_adapter_or_base(
        args.model, args.base, dtype=args.dtype
    )

    if args.mode == "psych101":
        # Psych-101 참가자 일반화: 주요 실험 prefix
        task_names = [
            "badham2017deficits","bahrami2020four","enkavi2019adaptivenback",
            "enkavi2019digitspan","enkavi2019gonogo","enkavi2019recentprobes",
            "feng2021dynamics","flesch2018comparing","frey2017cct","frey2017risk",
            "gershman2018deconstructing","gershman2020reward","hebart2023things",
            "hilbig2014generalized","kool2016when","kool2017cost","lefebvre2017behavioural",
            "levering2020revisiting","ludwig2023human","peterson2021using","plonsky2018when",
            "ruggeri2022globalizability","sadeghiyeh2020temporal","schulz2020finding",
            "somerville2017charting","speekenbrink2008learning","steingroever2015data",
            "tomov2020discovery","tomov2021multitask","waltz2020differential","wilson2014humans",
            "wu2023chunking","wulff2018description","wulff2018sampling","xiong2023neural","zorowitz2023data",
        ]
        ds = load_dataset("marcelbinz/Psych-101-test")
        run_eval_split(args, ds, tag="psych101", tasks=task_names)

    else:  # generalization
        # 스크립트 파일 위치 기준 절대경로 생성 (어디서 실행해도 OK)
        here = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(here, "generalization")
        files = [
            os.path.join(base_dir, "feher2020humans", "prompts.jsonl"),
            os.path.join(base_dir, "dubois2022value", "prompts.jsonl"),
        ]
        for i, fp in enumerate(files):
            try:
                # 1차: datasets의 json 로더
                ds = load_dataset("json", data_files={"test": [fp]})
                test = ds["test"]
                if "text" not in test.column_names:
                    for cand in ["prompt", "instruction"]:
                        if cand in test.column_names:
                            test = test.rename_column(cand, "text")
                            break
                    else:
                        # 적절한 컬럼이 없으면 폴백
                        raise KeyError("No 'text'/'prompt'/'instruction' column")
                ds = {"test": test}
            except Exception:
                # 2차: 수동 파서(형식이 달라도 흡수)
                test = _jsonl_to_text_dataset(fp)
                ds = {"test": test}

            run_eval_split(args, ds, tag=f"gen{i}", tasks=None)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL:", repr(e), file=sys.stderr)
        raise
