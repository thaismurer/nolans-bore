#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def is_numeric_colname(name: str) -> bool:
    try:
        float(str(name))
        return True
    except Exception:
        return False

def main():
    p = argparse.ArgumentParser(description="Generate deterministic GroupKFold indices JSON for spectral data.")
    p.add_argument("--csv", required=True, help="Path to the input CSV.")
    p.add_argument("--out", default="folds_groupkfold_v1.json", help="Output JSON filename.")
    p.add_argument("--group-col", default="interval", help="Name of the group/interval column.")
    p.add_argument("--depth-col", default="depth", help="Optional depth column used for stable sorting if present.")
    p.add_argument("--target-col", default="totalREE", help="Optional target column name (not required for fold split).")
    p.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # Stable ordering: by group then depth (if available)
    sort_cols = [c for c in [args.group_col, args.depth_col] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        # fallback: only by group if depth missing
        if args.group_col in df.columns:
            df = df.sort_values([args.group_col]).reset_index(drop=True)

    if args.group_col not in df.columns:
        raise ValueError(f"Group column '{args.group_col}' not found in CSV. Available: {list(df.columns)}")

    # Identify spectral columns (numeric names like '400','401.5', etc.).
    # We don't actually need X to produce the fold indices, but keeping this
    # selection here helps validate the dataset and future-proof the script.
    non_spectral = {args.group_col, args.depth_col, args.target_col}
    spectral_cols = [c for c in df.columns if c not in non_spectral and is_numeric_colname(c)]
    if not spectral_cols:
        print("Warning: No numeric-named spectral columns detected. Proceeding anyway (only groups are needed).")

    X = df[spectral_cols].values if spectral_cols else df.drop(columns=list(non_spectral & set(df.columns)), errors='ignore').values
    # y is not required; using zeros placeholder to satisfy the API signature
    y = np.zeros(len(df), dtype=float)
    groups = df[args.group_col].values

    gkf = GroupKFold(n_splits=args.n_splits)

    folds = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        folds.append({
            "train": np.asarray(train_idx, dtype=int).tolist(),
            "test":  np.asarray(test_idx,  dtype=int).tolist(),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False)

    # Optional: also write a CSV with a fold id for quick auditing (one-vs-holdout scheme)
    fold_id = np.empty(len(df), dtype=int); fold_id.fill(-1)
    for i, fdict in enumerate(folds):
        test_idx = np.array(fdict["test"], dtype=int)
        fold_id[test_idx] = i
    df_out = df.copy()
    df_out["fold"] = fold_id
    from pathlib import Path as _P
    aux_csv = _P(args.out).with_suffix(".aux_folds.csv")
    df_out.to_csv(aux_csv, index=False)
    print(f"[OK] Wrote JSON folds to: {args.out}")
    print(f"[OK] Wrote auxiliary CSV (per-row fold id) to: {aux_csv}")

if __name__ == "__main__":
    main()
