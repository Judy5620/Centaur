# test_adapter_hf.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"   # 4bit 베이스 (A100 40GB OK)
ADAPTER = "marcelbinz/Llama-3.1-Centaur-8B-adapter"

# 토크나이저
tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)

# 4bit 베이스 로드 (bitsandbytes)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",            # 단일 GPU(0)
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

# LoRA 어댑터 주입
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

# Centaur 권장 프롬프트 포맷 (<< >>)
prompt = (
    "You will see options. Choose like a human.\n"
    "Question: Which shape is more typical?\n"
    "<<Option A: circle>>\n"
    "<<Option B: hexagon>>\n"
    "Answer:"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))

