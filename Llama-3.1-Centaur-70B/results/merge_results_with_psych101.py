# results/merge_results_with_psych101.py
# - custom_metrics_*_{MODEL_DASH}.csv (gen0, gen1, psych101) 병합
# - + results/psych101_{MODEL_DASH}.csv 읽어서 per-task 평균을 1행으로 추가
# - 출력: combined_custom_metrics_with_psych101.csv, summary_custom_metrics_with_psych101.csv

import os, glob, argparse, math
import pandas as pd

def load_custom_metrics(results_dir: str, model_dash: str) -> pd.DataFrame:
    pat = os.path.join(results_dir, f"custom_metrics_*_{model_dash}.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No files matched: {pat}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        # 기대 컬럼: ["task", f"{model_dash}_loss", f"{model_dash}_ppl"]
        keep = ["task"] + [c for c in df.columns if c.endswith("_loss") or c.endswith("_ppl")]
        dfs.append(df[keep])
    return pd.concat(dfs, ignore_index=True)

def load_psych101_mean(results_dir: str, model_dash: str) -> pd.DataFrame | None:
    # 기대 파일: results/psych101_{MODEL_DASH}.csv
    fp = os.path.join(results_dir, f"psych101_{model_dash}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    # 기대 컬럼: ["task", f"{model_dash}"] (per-task eval_loss)
    value_col = model_dash
    if value_col not in df.columns:
        # 혹시 다른 이름으로 저장됐을 가능성 방어 (대소문자/공백)
        # 숫자 1개짜리 열을 찾되 task는 제외
        cand = [c for c in df.columns if c.lower() != "task"]
        if not cand:
            return None
        value_col = cand[0]

    # 숫자화 & NaN 제거
    vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(vals) == 0:
        return None

    mean_loss = float(vals.mean())
    mean_ppl  = math.exp(mean_loss)

    # custom_metrics 스키마에 맞춰 컬럼명 통일
    row = pd.DataFrame(
        [["psych101_per_task_mean", mean_loss, mean_ppl]],
        columns=["task", f"{model_dash}_loss", f"{model_dash}_ppl"]
    )
    return row

def make_summary(combined: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in combined.columns if c != "task"]
    overall = combined[num_cols].mean(numeric_only=True).to_frame().T
    overall.insert(0, "task", "OVERALL_MEAN")
    by_task = combined.groupby("task", as_index=False)[num_cols].mean(numeric_only=True)
    return pd.concat([overall, by_task], ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dash", required=True,
                    help="예: marcelbinz-Llama-3.1-Centaur-8B-adapter (슬래시 대신 대시)")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1) custom_metrics 병합
    combined = load_custom_metrics(args.results_dir, args.model_dash)

    # 2) psych101 per-task 평균 1행 추가(있을 때만)
    psych_mean = load_psych101_mean(args.results_dir, args.model_dash)
    if psych_mean is not None:
        combined = pd.concat([combined, psych_mean], ignore_index=True)

    # 3) 요약 생성(전체 평균 + task별 평균)
    summary = make_summary(combined)

    # 4) 저장 (기존 파일을 덮어쓰지 않도록 with_psych101 접미사 사용)
    combined_out = os.path.join(args.results_dir, "combined_custom_metrics_with_psych101.csv")
    summary_out  = os.path.join(args.results_dir, "summary_custom_metrics_with_psych101.csv")
    combined.to_csv(combined_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("Combined:", combined_out)
    print("Summary :", summary_out)

if __name__ == "__main__":
    main()
