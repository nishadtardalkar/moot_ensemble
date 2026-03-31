import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from get_metrics import calculate_pareto_metrics


def parse_objective_columns(columns):
    obj_cols = [c for c in columns if c.endswith(("+", "-"))]
    maximize_mask = [c.endswith("+") for c in obj_cols]
    return obj_cols, maximize_mask


def to_numeric_objectives(df, obj_cols):
    out = df[obj_cols].copy()
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=obj_cols)
    return out.to_numpy(dtype=float)


def to_minimization(values, maximize_mask):
    arr = values.copy()
    mask = np.asarray(maximize_mask, dtype=bool)
    arr[:, mask] = -arr[:, mask]
    return arr


def non_dominated_front(values_min):
    idx = NonDominatedSorting().do(values_min, only_non_dominated_front=True)
    return values_min[idx], idx


def find_source_csv(dataset_filename, search_roots):
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        direct = root_path / dataset_filename
        if direct.exists():
            return direct
        matches = list(root_path.rglob(dataset_filename))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute HV/Spread/IGD/MDH for all algorithm outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing per-algorithm frontier CSVs.",
    )
    parser.add_argument(
        "--data-roots",
        nargs="*",
        default=["moot/behavior_data", "moot/optimize/behavior_data", "."],
        help="Directories to search for source dataset CSV files.",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/metrics_summary.csv",
        help="Where to write aggregated metrics CSV.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    rows = []
    files = sorted(outputs_dir.glob("*/*.csv"))

    for path in files:
        algorithm = path.parent.name
        dataset_name = path.stem

        frontier_df = pd.read_csv(path)
        obj_cols, maximize_mask = parse_objective_columns(frontier_df.columns)
        if not obj_cols:
            print(f"Skipping {path}: no objective columns ending in +/-")
            continue

        frontier_vals = to_numeric_objectives(frontier_df, obj_cols)
        if len(frontier_vals) == 0:
            print(f"Skipping {path}: no valid numeric frontier rows")
            continue

        source_path = find_source_csv(f"{dataset_name}.csv", args.data_roots)
        ref_front = None
        if source_path is not None:
            source_df = pd.read_csv(source_path)
            if all(c in source_df.columns for c in obj_cols):
                source_vals = to_numeric_objectives(source_df, obj_cols)
                if len(source_vals) > 0:
                    source_min = to_minimization(source_vals, maximize_mask)
                    _, nd_idx = non_dominated_front(source_min)
                    ref_front = source_vals[nd_idx]

        metrics = calculate_pareto_metrics(
            frontier=frontier_vals,
            reference_front=ref_front,
            maximize_mask=maximize_mask,
            normalize=True,
        )

        rows.append(
            {
                "algorithm": algorithm,
                "dataset": f"{dataset_name}.csv",
                "reference_rows": int(len(ref_front)) if ref_front is not None else 0,
                "hv": metrics["hv"],
                "spread": metrics["spread"],
                "igd": metrics["igd"],
                "mdh": metrics["mdh"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    if summary_df.empty:
        print("No metrics were computed. Check outputs folder/content.")
        return

    print(f"Wrote metrics summary: {summary_path}")
    print(summary_df.sort_values(["dataset", "algorithm"]).to_string(index=False))


if __name__ == "__main__":
    main()

