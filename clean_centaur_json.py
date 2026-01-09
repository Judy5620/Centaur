import json
import re
import argparse
from collections import OrderedDict

# -------------------------
# Helper functions
# -------------------------

def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection"""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_broken_prompt(prompt: str) -> bool:
    """Detect obviously broken or incomplete prompts"""
    if prompt is None:
        return True
    if len(prompt.strip()) < 50:
        return True
    if prompt.strip().endswith("<<"):
        return True
    if prompt.strip() in ["you press <<", "you press <<1>>"]:
        return True
    return False

def has_multiple_questions(prompt: str) -> bool:
    """Detect multiple task mixing"""
    return prompt.lower().count("which of the following") > 1

def has_valid_output_format(prompt: str) -> bool:
    """Check whether output format is explicitly constrained"""
    patterns = [
        r"<<\s*1\s*>>",
        r"<<\s*2\s*>>",
        r"<<\s*3\s*>>",
        r"<<\s*score\s*>>",
        r"<<\s*\d+\s*>>",
    ]
    return any(re.search(p, prompt.lower()) for p in patterns)

# -------------------------
# Main cleaning logic
# -------------------------

def clean_examples(examples):
    seen_prompts = set()
    cleaned = []
    removed = []

    for ex in examples:
        ex_id = ex.get("id", "UNKNOWN_ID")
        prompt = ex.get("prompt", "")

        reason = None

        if is_broken_prompt(prompt):
            reason = "broken_or_too_short_prompt"
        elif has_multiple_questions(prompt):
            reason = "multiple_tasks_in_single_prompt"
        elif not has_valid_output_format(prompt):
            reason = "missing_or_unclear_output_format"
        else:
            norm = normalize_text(prompt)
            if norm in seen_prompts:
                reason = "duplicate_prompt"
            else:
                seen_prompts.add(norm)

        if reason:
            removed.append({
                "id": ex_id,
                "reason": reason,
                "prompt_preview": prompt[:200]
            })
        else:
            cleaned.append(ex)

    return cleaned, removed

# -------------------------
# Entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples")

    cleaned, removed = clean_examples(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(removed, f, indent=2, ensure_ascii=False)

    print("=== Cleaning Summary ===")
    print(f"Original examples : {len(data)}")
    print(f"Kept examples     : {len(cleaned)}")
    print(f"Removed examples  : {len(removed)}")
    print(f"Saved cleaned JSON to: {args.output}")
    print(f"Saved removal log to : {args.log}")

if __name__ == "__main__":
    main()
