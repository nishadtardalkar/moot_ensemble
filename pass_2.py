import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from get_metrics import calculate_pareto_metrics


def parse_objective_columns(columns):
    obj_cols = [c for c in columns if c.endswith(("+", "-"))]
    maximize_mask = np.array([c.endswith("+") for c in obj_cols], dtype=bool)
    return obj_cols, maximize_mask


def to_minimization_matrix(df, obj_cols, maximize_mask):
    obj = df[obj_cols].copy()
    for c in obj_cols:
        obj[c] = pd.to_numeric(obj[c], errors="coerce")
    valid = obj.notna().all(axis=1).to_numpy()
    obj_np = obj.to_numpy(dtype=float)
    obj_np[:, maximize_mask] = -obj_np[:, maximize_mask]
    return obj_np, valid


def non_dominated_mask(F):
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        mask[dominates_i] = False
        mask[i] = True
    return mask


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


def to_numeric_objectives(df, obj_cols):
    out = df[obj_cols].copy()
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=obj_cols)
    return out.to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Create merged Pareto frontiers per dataset from outputs/*/*.csv."
    )
    parser.add_argument("--outputs-dir", default="outputs", help="Root outputs directory.")
    parser.add_argument(
        "--out-dir",
        default="outputs/combined_frontiers",
        help="Directory to write merged Pareto frontier files.",
    )
    parser.add_argument(
        "--data-roots",
        nargs="*",
        default=["moot/behavior_data", "moot/optimize/behavior_data", "."],
        help="Directories to search for source dataset CSV files (for IGD reference).",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(outputs_dir.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No frontier CSV files found under: {outputs_dir}")

    groups = {}
    for f in files:
        groups.setdefault(f.name, []).append(f)

    summary_rows = []

    for dataset_file, paths in groups.items():
        all_rows = []
        objective_cols = None
        maximize_mask = None

        for p in paths:
            df = pd.read_csv(p)
            obj_cols, max_mask = parse_objective_columns(df.columns)
            if not obj_cols:
                continue

            if objective_cols is None:
                objective_cols = obj_cols
                maximize_mask = max_mask
            elif obj_cols != objective_cols:
                # Skip mismatched schema for same dataset filename.
                continue

            df = df.copy()
            df["_source_algorithm"] = p.parent.name
            df["_source_file"] = str(p)
            all_rows.append(df)

        if not all_rows or objective_cols is None:
            continue

        merged = pd.concat(all_rows, ignore_index=True)
        merged = merged.drop_duplicates()

        F, valid_mask = to_minimization_matrix(merged, objective_cols, maximize_mask)
        merged_valid = merged.loc[valid_mask].reset_index(drop=True)
        F_valid = F[valid_mask]

        if len(merged_valid) == 0:
            continue

        nd_mask = non_dominated_mask(F_valid)
        pareto = merged_valid.loc[nd_mask].copy().reset_index(drop=True)

        out_path = out_dir / dataset_file
        pareto.to_csv(out_path, index=False)

        frontier_values = to_numeric_objectives(pareto, objective_cols)
        reference_values = None
        source_path = find_source_csv(dataset_file, args.data_roots)
        if source_path is not None:
            source_df = pd.read_csv(source_path)
            if all(c in source_df.columns for c in objective_cols):
                source_values = to_numeric_objectives(source_df, objective_cols)
                if len(source_values) > 0:
                    source_min = source_values.copy()
                    source_min[:, maximize_mask] = -source_min[:, maximize_mask]
                    source_nd = non_dominated_mask(source_min)
                    reference_values = source_values[source_nd]

        metrics = calculate_pareto_metrics(
            frontier=frontier_values,
            reference_front=reference_values,
            maximize_mask=maximize_mask,
            normalize=True,
        )

        summary_rows.append(
            {
                "dataset": dataset_file,
                "input_files": len(paths),
                "merged_rows": int(len(merged_valid)),
                "pareto_rows": int(len(pareto)),
                "reference_rows": int(len(reference_values)) if reference_values is not None else 0,
                "hv": metrics["hv"],
                "spread": metrics["spread"],
                "igd": metrics["igd"],
                "mdh": metrics["mdh"],
                "output_file": str(out_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote combined frontiers to: {out_dir}")
    print(f"Wrote summary: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

