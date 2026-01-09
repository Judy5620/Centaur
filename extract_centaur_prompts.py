import json
import re
from pathlib import Path

NB_PATH = "Centaur_8b_Test.ipynb"
OUT_PATH = "centaur_examples.json"

# prompt를 잡기 위한 휴리스틱 패턴들
PROMPT_PATTERNS = [
    r'prompt\s*=\s*"""(.*?)"""',
    r'prompt\s*=\s*"(.*?)"',
    r'You are taking part in a reasoning experiment\.(.*?)You press <<',
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

examples = []
counter = 1

for cell in nb["cells"]:
    if "source" not in cell:
        continue

    cell_text = "".join(cell["source"])

    for pat in PROMPT_PATTERNS:
        matches = re.findall(pat, cell_text, flags=re.DOTALL)
        for m in matches:
            prompt = m.strip()

            # 마지막에 항상 "You press <<"를 붙여 Centaur 형식 유지
            if not prompt.endswith("You press <<"):
                prompt = prompt.strip() + "\nYou press <<"

            examples.append({
                "id": f"example_{counter:03d}",
                "prompt": prompt
            })
            counter += 1

# 중복 제거 (prompt 기준)
unique = []
seen = set()
for ex in examples:
    if ex["prompt"] not in seen:
        seen.add(ex["prompt"])
        unique.append(ex)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(unique)} prompts → {OUT_PATH}")
