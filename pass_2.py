import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from get_metrics import calculate_pareto_metrics

# Subdirs under outputs/ that are not pass_1 algorithm runs (skip when scanning).
_SKIP_OUTPUT_SUBDIRS = frozenset({"combined_frontiers", "pass3"})


def discover_pass1_frontier_csvs(outputs_dir: Path):
    """
    CSVs written by pass_1: pass_1_outputs/<DatasetName>/<AlgorithmName>.csv
    Excludes merged outputs and per-run summary.csv files.
    """
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    paths = []
    for sub in sorted(outputs_dir.iterdir()):
        if not sub.is_dir() or sub.name in _SKIP_OUTPUT_SUBDIRS:
            continue
        for p in sorted(sub.glob("*.csv")):
            if p.name.lower() == "summary.csv":
                continue
            paths.append(p)
    return sorted(paths)


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


def normalize_dataset_name(dataset_value):
    return Path(str(dataset_value)).stem


def compare_pass2_vs_pass1_totals(pass2_summary: pd.DataFrame, pass1_summary: pd.DataFrame):
    """
    Aggregate comparison counts across all datasets:
    - hv/spread: higher is better
    - igd/mdh: lower is better
    """
    if pass2_summary.empty or pass1_summary.empty:
        return pd.DataFrame()

    p2 = pass2_summary.copy()
    p1 = pass1_summary.copy()

    required_cols = {"dataset", "hv", "spread", "igd", "mdh"}
    if not required_cols.issubset(set(p2.columns)):
        raise ValueError(
            "pass_2 summary must contain 'dataset', 'hv', 'spread', 'igd', and 'mdh' columns."
        )
    if not required_cols.issubset(set(p1.columns)):
        raise ValueError(
            "pass_1 summary must contain 'dataset', 'hv', 'spread', 'igd', and 'mdh' columns."
        )

    p2["dataset_key"] = p2["dataset"].map(normalize_dataset_name)
    p1["dataset_key"] = p1["dataset"].map(normalize_dataset_name)
    p2["hv"] = pd.to_numeric(p2["hv"], errors="coerce")
    p1["hv"] = pd.to_numeric(p1["hv"], errors="coerce")
    p2["spread"] = pd.to_numeric(p2["spread"], errors="coerce")
    p1["spread"] = pd.to_numeric(p1["spread"], errors="coerce")
    p2["igd"] = pd.to_numeric(p2["igd"], errors="coerce")
    p1["igd"] = pd.to_numeric(p1["igd"], errors="coerce")
    p2["mdh"] = pd.to_numeric(p2["mdh"], errors="coerce")
    p1["mdh"] = pd.to_numeric(p1["mdh"], errors="coerce")

    hv_better_total = 0
    spread_better_total = 0
    igd_better_total = 0
    mdh_better_total = 0
    hv_cmp_total = 0
    spread_cmp_total = 0
    igd_cmp_total = 0
    mdh_cmp_total = 0

    for _, p2_row in p2.iterrows():
        dataset_key = p2_row["dataset_key"]
        pass2_hv = p2_row["hv"]
        pass2_spread = p2_row["spread"]
        pass2_igd = p2_row["igd"]
        pass2_mdh = p2_row["mdh"]
        if pd.isna(pass2_hv) or pd.isna(pass2_spread) or pd.isna(pass2_igd) or pd.isna(pass2_mdh):
            continue

        p1_dataset = p1[p1["dataset_key"] == dataset_key]
        p1_hv = p1_dataset["hv"].dropna()
        p1_spread = p1_dataset["spread"].dropna()
        p1_igd = p1_dataset["igd"].dropna()
        p1_mdh = p1_dataset["mdh"].dropna()
        if p1_hv.empty or p1_spread.empty or p1_igd.empty or p1_mdh.empty:
            continue

        hv_cmp_total += int(len(p1_hv))
        spread_cmp_total += int(len(p1_spread))
        igd_cmp_total += int(len(p1_igd))
        mdh_cmp_total += int(len(p1_mdh))
        hv_better_total += int((pass2_hv > p1_hv).sum())
        # higher spread is better (mean pairwise distance diversity proxy)
        spread_better_total += int((pass2_spread > p1_spread).sum())
        # lower igd/mdh are better
        igd_better_total += int((pass2_igd < p1_igd).sum())
        mdh_better_total += int((pass2_mdh < p1_mdh).sum())

    if hv_cmp_total == 0 and spread_cmp_total == 0 and igd_cmp_total == 0 and mdh_cmp_total == 0:
        return pd.DataFrame(
            [
                {
                    "hv_better_pct": np.nan,
                    "spread_better_pct": np.nan,
                    "igd_better_pct": np.nan,
                    "mdh_better_pct": np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "hv_better_pct": float(100.0 * hv_better_total / hv_cmp_total)
                if hv_cmp_total > 0
                else np.nan,
                "spread_better_pct": float(100.0 * spread_better_total / spread_cmp_total)
                if spread_cmp_total > 0
                else np.nan,
                "igd_better_pct": float(100.0 * igd_better_total / igd_cmp_total)
                if igd_cmp_total > 0
                else np.nan,
                "mdh_better_pct": float(100.0 * mdh_better_total / mdh_cmp_total)
                if mdh_cmp_total > 0
                else np.nan,
            }
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge Pareto frontiers per dataset from pass_1 outputs "
            "(pass_1_outputs/<dataset>/*.csv, excluding combined_frontiers/ and pass3/)."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        default="pass_1_outputs",
        help="Root directory with pass_1 outputs (default: pass_1_outputs).",
    )
    parser.add_argument(
        "--out-dir",
        default="pass_2_outputs",
        help="Directory to write combined frontiers (default: pass_2_outputs).",
    )
    parser.add_argument(
        "--data-roots",
        nargs="*",
        default=["moot/behavior_data", "moot/optimize/behavior_data", "."],
        help="Directories to search for source dataset CSV files (for IGD reference).",
    )
    parser.add_argument(
        "--pass1-summary",
        default=None,
        help=(
            "Path to pass_1 summary CSV for HV comparison "
            "(default: <outputs-dir>/summary.csv)."
        ),
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_pass1_frontier_csvs(outputs_dir)
    if not files:
        raise FileNotFoundError(
            f"No pass_1 frontier CSVs found under {outputs_dir}/*/*.csv "
            f"(excluding combined_frontiers/, pass3/, and summary.csv)."
        )

    groups = {}
    for f in files:
        dataset_name = f.parent.name
        groups.setdefault(dataset_name, []).append(f)

    summary_rows = []

    for dataset_name, paths in groups.items():
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
                # Skip mismatched schema within same dataset group.
                continue

            all_rows.append(df.copy())

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

        dataset_out_dir = out_dir / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_out_dir / "combined_frontier.csv"
        pareto.to_csv(out_path, index=False)

        frontier_values = to_numeric_objectives(pareto, objective_cols)
        reference_values = None
        source_path = find_source_csv(f"{dataset_name}.csv", args.data_roots)
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
                "dataset": dataset_name,
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

    pass1_summary_path = (
        Path(args.pass1_summary) if args.pass1_summary else (outputs_dir / "summary.csv")
    )
    if pass1_summary_path.exists():
        pass1_summary = pd.read_csv(pass1_summary_path)
        hv_comparison = compare_pass2_vs_pass1_totals(summary, pass1_summary)
        comparison_path = out_dir / "hv_comparison_vs_pass1.csv"
        hv_comparison.to_csv(comparison_path, index=False)
        print(f"Wrote HV comparison vs pass_1: {comparison_path}")
        if not hv_comparison.empty:
            print(hv_comparison.to_string(index=False))
    else:
        print(f"Skipping HV comparison: pass_1 summary not found at {pass1_summary_path}")


if __name__ == "__main__":
    main()

