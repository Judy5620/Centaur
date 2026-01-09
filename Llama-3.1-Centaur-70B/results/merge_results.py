# merge_results.py
import os, glob, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="CSV들이 있는 폴더")
    ap.add_argument("--model", required=True,
                    help="모델 ID 또는 대시형식(ex: marcelbinz/Llama-3.1-Centaur-8B-adapter 또는 marcelbinz-Llama-3.1-Centaur-8B-adapter)")
    ap.add_argument("--out_prefix", default="all_data", help="출력 파일 접두어")
    ap.add_argument("--strict", action="store_true",
                    help="비숫자(결측) 값이 있으면 실패(기본은 무시하고 드롭)")
    args = ap.parse_args()

    model_dash = args.model.replace("/", "-")
    pattern = os.path.join(args.results_dir, f"*_{model_dash}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No CSV matches: {pattern}")

    rows = []
    dropped_info = []

    for f in files:
        df = pd.read_csv(f)
        tag = os.path.basename(f).split("_")[0]   # psych101, gen0, gen1 ...

        # 컬럼 정규화: (task, <metric>)
        non_meta_cols = [c for c in df.columns if c not in ("task", "source")]
        if len(non_meta_cols) != 1:
            raise ValueError(f"Unexpected columns in {f}: {df.columns.tolist()}")
        metric_col = non_meta_cols[0]
        if metric_col != model_dash:
            df = df.rename(columns={metric_col: model_dash})

        # 숫자 변환(중간에 섞인 문자열/LFS 포인터 제거)
        before = len(df)
        df[model_dash] = pd.to_numeric(df[model_dash], errors="coerce")
        bad = df[model_dash].isna()
        n_bad = int(bad.sum())
        if n_bad > 0:
            sample_vals = df.loc[bad, model_dash].head(3).tolist()
            dropped_info.append(f"{os.path.basename(f)}: dropped {n_bad}/{before} non-numeric rows")
            # 메트릭이 NaN인 행 제거 (task만 있고 값이 없는 행)
            df = df.loc[~bad].copy()

        df["source"] = tag
        rows.append(df[["task", "source", model_dash]])

    out = pd.concat(rows, ignore_index=True)
    if args.strict and out[model_dash].isna().any():
        raise SystemExit("Strict mode: found NaNs after merge.")

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.out_prefix}_{model_dash}.csv")
    out.to_csv(out_path, index=False)

    # 소스별 평균
    summary = (
        out.groupby("source", as_index=False)[model_dash]
          .mean()
          .rename(columns={model_dash: "mean_eval_loss"})
          .sort_values("source")
    )
    sum_path = os.path.join(args.results_dir, f"{args.out_prefix}_{model_dash}_summary.csv")
    summary.to_csv(sum_path, index=False)

    print(f"Saved merged:  {out_path}")
    print(f"Saved summary: {sum_path}")
    if dropped_info:
        print("Note: non-numeric rows were dropped:")
        for line in dropped_info:
            print(" -", line)

if __name__ == "__main__":
    main()
