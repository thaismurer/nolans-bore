from __future__ import annotations
from typing import List, Tuple, Optional
import json
import sys
import argparse
import numpy as np
import pandas as pd

# Default intervals copied from the notebook. Each tuple is (start, end).
INTERVALS: List[Tuple[float, float]] = [
    (6, 7), (7, 8.006), (8.006, 9), (9, 10), (10, 11), (11, 12.007), (12.007, 13.5),
    (13.5, 15), (15, 17.6), (17.6, 19), (19, 21), (21, 23), (23, 25.007), (25.007, 27),
    (27, 29), (29, 29.6), (29.6, 31.008), (31.008, 32.007), (32.007, 33.007), (33.007, 34),
    (34, 34.6), (34.6, 35.9), (35.9, 37.008), (37.008, 38), (38, 39.008), (39.008, 40),
    (40, 41), (41, 42.007), (42.007, 43.007), (43.007, 44), (44, 45), (45, 46), (46, 47.008),
    (47.008, 48.007), (48.007, 49.007), (49.007, 50.01), (50.01, 51.01), (51.01, 51.710),
    (51.710, 53.3), (53.3, 55.008), (55.008, 56.007), (56.007, 57.006), (57.006, 57.7),
    (57.7, 59), (59, 61.6), (61.6, 63.007), (63.007, 64.704), (64.704, 66.018), (66.018, 67),
    (67, 68), (68, 69.5), (69.5, 71.008), (71.008, 73.007), (73.007, 73.7), (73.7, 75),
    (75, 77), (77, 79), (79, 81), (81, 83), (83, 84.1), (84.1, 85.006), (85.006, 86.007),
    (86.007, 87), (87, 88), (88, 89), (89, 90.6), (90.6, 92.8), (92.8, 94.008), (94.008, 95)
]


def depth_to_interval_label(depth: Optional[float], intervals: List[Tuple[float, float]] = INTERVALS) -> Optional[str]:
    """Return the interval label 'start-end' that contains the given depth.

    Rule: start <= depth < end for all intervals except the last where end is inclusive.
    Returns None if depth is missing or does not fall into any interval.
    """
    if depth is None:
        return None
    try:
        d = float(depth)
    except Exception:
        return None

    for i, (start, end) in enumerate(intervals):
        if i < len(intervals) - 1:
            if start <= d < end:
                return f"{start}-{end}"
        else:
            # last interval: include end
            if start <= d <= end:
                return f"{start}-{end}"
    return None


def add_interval_column(df: pd.DataFrame, depth_col: str = "Depth", intervals: List[Tuple[float, float]] = INTERVALS) -> pd.DataFrame:
    """Add/replace the `interval` column in `df` using `depth_col`.

    The function will try to find the depth column case-insensitively if the
    provided name is not present. Returned DataFrame is a copy of the input with
    a new `interval` column containing labels like '6-7'.
    """
    df_out = df.copy()

    # Find the depth column if needed (case-insensitive)
    if depth_col not in df_out.columns:
        lc_map = {c.lower(): c for c in df_out.columns}
        if depth_col.lower() in lc_map:
            depth_col = lc_map[depth_col.lower()]
        else:
            raise KeyError(f"Depth column '{depth_col}' not found in DataFrame columns: {list(df_out.columns)}")

    # Apply mapping
    df_out["interval"] = df_out[depth_col].apply(lambda v: depth_to_interval_label(v, intervals))
    return df_out


def load_intervals_from_json(path: str) -> List[Tuple[float, float]]:
    """Load intervals from a JSON file. The file should contain a list of [start, end] pairs.
    Example: [[6,7], [7,8.006], ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intervals: List[Tuple[float, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            intervals.append((float(item[0]), float(item[1])))
        else:
            raise ValueError("Invalid interval format in JSON. Expected list of [start, end] pairs.")
    return intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add an 'interval' column to a CSV based on depth intervals.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file. If omitted, adds '_interval' before extension")
    parser.add_argument("--depth-col", default="Depth", help="Name of the depth column (default: Depth)")
    parser.add_argument("--intervals-json", help="Optional JSON file with intervals (list of [start, end]) to override defaults")

    args = parser.parse_args(argv)

    if args.intervals_json:
        intervals = load_intervals_from_json(args.intervals_json)
    else:
        intervals = INTERVALS

    df = pd.read_csv(args.input_csv)
    df_new = add_interval_column(df, depth_col=args.depth_col, intervals=intervals)

    # Output path
    out_path = args.output
    if not out_path:
        if args.input_csv.lower().endswith('.csv'):
            out_path = args.input_csv[:-4] + '_interval.csv'
        else:
            out_path = args.input_csv + '_interval.csv'

    df_new.to_csv(out_path, index=False)
    print(f"Saved output with 'interval' column to: {out_path}")

    # Print a short summary
    counts = df_new['interval'].value_counts(dropna=False).sort_index()
    print('\nInterval counts (first 20 shown):')
    print(counts.head(20).to_string())


if __name__ == '__main__':
    main()