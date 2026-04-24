import argparse
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from get_metrics import calculate_pareto_metrics


def parse_moot_columns(df: pd.DataFrame):
    y_cols = [c for c in df.columns if c.endswith(("+", "-"))]
    # Exclude helper/metadata columns from pass outputs.
    x_cols = [c for c in df.columns if not c.endswith(("X", "+", "-")) and not c.startswith("_")]
    if not y_cols:
        raise ValueError("No objective columns found (expected +/- suffix).")
    if not x_cols:
        raise ValueError("No decision columns found.")
    return x_cols, y_cols


def is_moot_numeric(col_name: str) -> bool:
    return len(col_name) > 0 and col_name[0].isupper()


def build_mixed_x(df: pd.DataFrame, x_cols):
    num_cols = [c for c in x_cols if is_moot_numeric(c)]
    cat_cols = [c for c in x_cols if not is_moot_numeric(c)]

    X_num = None
    X_cat = None

    if num_cols:
        x_num = df[num_cols].copy()
        for c in num_cols:
            x_num[c] = pd.to_numeric(x_num[c], errors="coerce")
        x_num = x_num.fillna(x_num.median(numeric_only=True))
        mins = x_num.min()
        maxs = x_num.max()
        denom = (maxs - mins).replace(0, 1.0)
        x_num = (x_num - mins) / denom
        X_num = x_num.to_numpy(dtype=np.float32)

    if cat_cols:
        x_cat = df[cat_cols].copy().astype(str).fillna("?")
        for c in cat_cols:
            x_cat[c] = pd.Categorical(x_cat[c]).codes
        X_cat = x_cat.to_numpy(dtype=np.int32)

    return X_num, X_cat, num_cols, cat_cols


def build_objective_matrix(df: pd.DataFrame, y_cols):
    y = df[y_cols].copy()
    for c in y_cols:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    y = y.dropna(subset=y_cols)
    y_for_pymoo = y.copy()
    for c in y_cols:
        if c.endswith("+"):
            y_for_pymoo[c] = -y_for_pymoo[c]
    return y.index.to_numpy(), y.values, y_for_pymoo.values


def to_numeric_objectives(df, obj_cols):
    out = df[obj_cols].copy()
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=obj_cols)
    return out.to_numpy(dtype=float)


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


class MOOTNearestNeighborProblem(Problem):
    def __init__(self, X_num, X_cat, Y_lookup):
        self.X_num = X_num
        self.X_cat = X_cat
        self.Y_lookup = Y_lookup

        self.n_num = 0 if X_num is None else X_num.shape[1]
        self.n_cat = 0 if X_cat is None else X_cat.shape[1]

        xl_num = np.zeros(self.n_num, dtype=float)
        xu_num = np.ones(self.n_num, dtype=float)

        if self.n_cat > 0:
            cat_max = X_cat.max(axis=0).astype(float)
            xl_cat = np.zeros(self.n_cat, dtype=float)
            xu_cat = cat_max
        else:
            xl_cat = np.array([], dtype=float)
            xu_cat = np.array([], dtype=float)

        xl = np.concatenate([xl_num, xl_cat])
        xu = np.concatenate([xu_num, xu_cat])

        super().__init__(n_var=self.n_num + self.n_cat, n_obj=Y_lookup.shape[1], xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_candidates = x.shape[0]
        nn_idx = np.empty(n_candidates, dtype=int)
        for i in range(n_candidates):
            dist = np.zeros(len(self.Y_lookup), dtype=np.float32)
            if self.n_num > 0:
                cand_num = x[i, : self.n_num]
                dist += ((self.X_num - cand_num) ** 2).sum(axis=1)
            if self.n_cat > 0:
                cand_cat = np.rint(x[i, self.n_num :]).astype(np.int32)
                dist += (self.X_cat != cand_cat).sum(axis=1)
            nn_idx[i] = np.argmin(dist)
        out["F"] = self.Y_lookup[nn_idx]


def nearest_uniform_n_points(n_obj, target):
    if n_obj <= 1:
        return max(2, target)
    values = []
    p = 1
    while p <= 200:
        n_points = comb(n_obj + p - 1, p)
        values.append(n_points)
        if n_points >= target:
            break
        p += 1
    best = min(values, key=lambda x: (abs(x - target), -x))
    return max(2, int(best))


def build_algorithms(problem, pop_size):
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=10)
    pop_size_ref = max(pop_size, len(ref_dirs))
    return [
        NSGA2(pop_size=pop_size),
        MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=max(2, min(15, len(ref_dirs))),
            prob_neighbor_mating=(0.7 if len(ref_dirs) > 15 else 0.5),
        ),
        NSGA3(ref_dirs=ref_dirs, pop_size=pop_size_ref),
        UNSGA3(ref_dirs=ref_dirs, pop_size=pop_size_ref),
        RNSGA3(
            ref_points=ref_dirs[: min(2, len(ref_dirs))],
            pop_per_ref_point=nearest_uniform_n_points(
                problem.n_obj,
                max(2, int(np.ceil(pop_size / max(1, min(2, len(ref_dirs)))))),
            ),
        ),
        RVEA(ref_dirs=ref_dirs, pop_size=pop_size_ref),
        AGEMOEA(pop_size=pop_size),
        AGEMOEA2(pop_size=pop_size),
        CTAEA(ref_dirs=ref_dirs),
    ]


def merge_frontiers(input_dir: Path):
    files = sorted(input_dir.glob("*/*.csv"))
    groups = {}
    for f in files:
        # Skip aggregate summary from pass_1.
        if f.name.lower() == "summary.csv":
            continue
        dataset_name = f.parent.name
        groups.setdefault(dataset_name, []).append(f)

    merged = {}
    for dataset_name, paths in groups.items():
        chunks = [pd.read_csv(p) for p in paths]
        merged_df = pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)
        merged[dataset_name] = merged_df
    return merged


def normalize_dataset_name(dataset_value):
    return Path(str(dataset_value)).stem


def compare_pass3_vs_pass1_per_algo(pass3_summary: pd.DataFrame, pass1_summary: pd.DataFrame):
    """
    Return percent-better metrics per pass_3 algorithm against pass_1 algorithms:
    - hv_better_pct: percent of pairwise comparisons where pass_3 hv is higher
    - spread_better_pct: percent of pairwise comparisons where pass_3 spread is higher
    - igd/mdh_better_pct: percent of pairwise comparisons where pass_3 is lower
    """
    if pass3_summary.empty or pass1_summary.empty:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "hv_better_pct",
                "spread_better_pct",
                "igd_better_pct",
                "mdh_better_pct",
            ]
        )

    required_cols = {"dataset", "hv", "spread", "igd", "mdh"}
    if not required_cols.issubset(set(pass3_summary.columns)):
        raise ValueError(
            "pass_3 summary must contain 'dataset', 'hv', 'spread', 'igd', and 'mdh' columns."
        )
    if not required_cols.issubset(set(pass1_summary.columns)):
        raise ValueError(
            "pass_1 summary must contain 'dataset', 'hv', 'spread', 'igd', and 'mdh' columns."
        )
    if "algorithm" not in pass3_summary.columns:
        raise ValueError("pass_3 summary must contain 'algorithm' column.")

    p3 = pass3_summary.copy()
    p1 = pass1_summary.copy()
    p3["dataset_key"] = p3["dataset"].map(normalize_dataset_name)
    p1["dataset_key"] = p1["dataset"].map(normalize_dataset_name)
    p3["hv"] = pd.to_numeric(p3["hv"], errors="coerce")
    p1["hv"] = pd.to_numeric(p1["hv"], errors="coerce")
    p3["spread"] = pd.to_numeric(p3["spread"], errors="coerce")
    p1["spread"] = pd.to_numeric(p1["spread"], errors="coerce")
    p3["igd"] = pd.to_numeric(p3["igd"], errors="coerce")
    p1["igd"] = pd.to_numeric(p1["igd"], errors="coerce")
    p3["mdh"] = pd.to_numeric(p3["mdh"], errors="coerce")
    p1["mdh"] = pd.to_numeric(p1["mdh"], errors="coerce")

    rows = []
    for algorithm, group in p3.groupby("algorithm"):
        hv_better_total = 0
        spread_better_total = 0
        igd_better_total = 0
        mdh_better_total = 0
        hv_cmp_total = 0
        spread_cmp_total = 0
        igd_cmp_total = 0
        mdh_cmp_total = 0

        for _, p3_row in group.iterrows():
            p3_hv = p3_row["hv"]
            p3_spread = p3_row["spread"]
            p3_igd = p3_row["igd"]
            p3_mdh = p3_row["mdh"]
            if pd.isna(p3_hv) or pd.isna(p3_spread) or pd.isna(p3_igd) or pd.isna(p3_mdh):
                continue

            p1_dataset = p1[p1["dataset_key"] == p3_row["dataset_key"]]
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
            hv_better_total += int((p3_hv > p1_hv).sum())
            spread_better_total += int((p3_spread > p1_spread).sum())
            igd_better_total += int((p3_igd < p1_igd).sum())
            mdh_better_total += int((p3_mdh < p1_mdh).sum())

        if hv_cmp_total == 0 and spread_cmp_total == 0 and igd_cmp_total == 0 and mdh_cmp_total == 0:
            rows.append(
                {
                    "algorithm": algorithm,
                    "hv_better_pct": np.nan,
                    "spread_better_pct": np.nan,
                    "igd_better_pct": np.nan,
                    "mdh_better_pct": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "algorithm": algorithm,
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
            )

    return pd.DataFrame(rows).sort_values("algorithm").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pass 3: optimize over merged pass-1 frontiers (no pairwise merge)."
    )
    parser.add_argument(
        "--input-dir",
        default="pass_1_outputs",
        help="Pass-1 outputs root (expects pass_1_outputs/<dataset>/<algorithm>.csv; skips summary.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="pass_3_outputs",
        help="Directory for pass-3 generated frontiers in pass_1 format.",
    )
    parser.add_argument("--pop-size", type=int, default=40, help="Population size.")
    parser.add_argument("--n-gen", type=int, default=20, help="Generations.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--data-roots",
        nargs="*",
        default=["moot/behavior_data", "moot/optimize/behavior_data", "."],
        help="Directories to search for source dataset CSV files (for IGD reference).",
    )
    parser.add_argument(
        "--algorithm-id",
        type=int,
        default=None,
        help="Run one algorithm by id; default runs all.",
    )
    parser.add_argument(
        "--pass1-summary",
        default=None,
        help=(
            "Path to pass_1 summary CSV for comparison "
            "(default: <input-dir>/summary.csv)."
        ),
    )
    args = parser.parse_args()

    merged_by_dataset = merge_frontiers(Path(args.input_dir))
    if not merged_by_dataset:
        raise FileNotFoundError(
            f"No frontier csv files found in {args.input_dir}/*/*.csv "
            "(after skipping combined_frontiers/ and summary.csv)."
        )

    metrics_rows = []

    for dataset_name, df in merged_by_dataset.items():
        print(f"\nDataset: {dataset_name} | merged rows: {len(df)}")
        x_cols, y_cols = parse_moot_columns(df)
        maximize_mask = np.array([c.endswith("+") for c in y_cols], dtype=bool)
        X_num, X_cat, num_cols, cat_cols = build_mixed_x(df, x_cols)

        valid_idx, _, Y_pymoo = build_objective_matrix(df, y_cols)
        df_valid = df.loc[valid_idx].reset_index(drop=True)
        if len(df_valid) == 0:
            print("  Skipping: no valid rows after objective numeric coercion.")
            continue

        # Rebuild x from valid rows to align with Y lookup indexing.
        X_num, X_cat, num_cols, cat_cols = build_mixed_x(df_valid, x_cols)
        problem = MOOTNearestNeighborProblem(X_num, X_cat, Y_pymoo)

        algorithms = build_algorithms(problem, args.pop_size)
        algo_ids = [args.algorithm_id] if args.algorithm_id is not None else list(range(len(algorithms)))

        dataset_output_paths = []
        dataset_metric_rows = []
        try:
            for aid in algo_ids:
                if aid < 0 or aid >= len(algorithms):
                    raise ValueError(f"Invalid algorithm_id={aid}. Valid ids: 0..{len(algorithms)-1}")
                algorithm = algorithms[aid]
                algorithm_name = algorithm.__class__.__name__
                print(f"  Running {aid}: {algorithm_name}")

                out_dataset_dir = Path(args.out_dir) / dataset_name
                out_dataset_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dataset_dir / f"{algorithm_name}.csv"

                if out_path.exists():
                    print(f"    Skipping existing output for {dataset_name}/{algorithm_name}: {out_path}")
                    final_rows = pd.read_csv(out_path)
                    frontier_values = to_numeric_objectives(final_rows, y_cols)
                    if len(frontier_values) == 0:
                        print("    Warning: existing output has no valid objective rows; skipping metrics row.")
                        continue

                    reference_values = None
                    source_path = find_source_csv(f"{dataset_name}.csv", args.data_roots)
                    if source_path is not None:
                        source_df = pd.read_csv(source_path)
                        if all(c in source_df.columns for c in y_cols):
                            source_values = to_numeric_objectives(source_df, y_cols)
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
                    dataset_metric_rows.append(
                        {
                            "algorithm": algorithm_name,
                            "dataset": f"{dataset_name}.csv",
                            "input_files": 1,
                            "merged_rows": int(len(final_rows)),
                            "pareto_rows": int(len(final_rows)),
                            "reference_rows": int(len(reference_values)) if reference_values is not None else 0,
                            "hv": metrics["hv"],
                            "spread": metrics["spread"],
                            "igd": metrics["igd"],
                            "mdh": metrics["mdh"],
                            "output_file": str(out_path),
                        }
                    )
                    continue

                start = time.time()
                res = minimize(
                    problem,
                    algorithm,
                    get_termination("n_gen", args.n_gen),
                    seed=args.seed,
                    verbose=False,
                )
                runtime = time.time() - start

                final_idx = []
                for i in range(res.X.shape[0]):
                    dist = np.zeros(len(df_valid), dtype=np.float32)
                    if X_num is not None:
                        cand_num = res.X[i, : len(num_cols)]
                        dist += ((X_num - cand_num) ** 2).sum(axis=1)
                    if X_cat is not None:
                        cand_cat = np.rint(res.X[i, len(num_cols) :]).astype(np.int32)
                        dist += (X_cat != cand_cat).sum(axis=1)
                    final_idx.append(np.argmin(dist))

                final_rows = df_valid.iloc[np.array(final_idx)].copy().reset_index(drop=True)
                final_rows["_source_row"] = np.array(final_idx)
                final_rows = final_rows.drop_duplicates(subset=["_source_row"]).reset_index(drop=True)
                final_rows.to_csv(out_path, index=False)
                dataset_output_paths.append(out_path)
                print(f"    wrote {len(final_rows)} rows -> {out_path} ({runtime:.1f}s)")

                # Metrics for pass-3 generated frontier
                frontier_values = to_numeric_objectives(final_rows, y_cols)
                reference_values = None
                source_path = find_source_csv(f"{dataset_name}.csv", args.data_roots)
                if source_path is not None:
                    source_df = pd.read_csv(source_path)
                    if all(c in source_df.columns for c in y_cols):
                        source_values = to_numeric_objectives(source_df, y_cols)
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
                dataset_metric_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "dataset": f"{dataset_name}.csv",
                        "input_files": 1,
                        "merged_rows": int(res.X.shape[0]),
                        "pareto_rows": int(len(final_rows)),
                        "reference_rows": int(len(reference_values)) if reference_values is not None else 0,
                        "hv": metrics["hv"],
                        "spread": metrics["spread"],
                        "igd": metrics["igd"],
                        "mdh": metrics["mdh"],
                        "output_file": str(out_path),
                    }
                )
            metrics_rows.extend(dataset_metric_rows)
        except Exception as exc:
            print(
                f"  Discarding dataset {dataset_name}: algorithm failed with "
                f"{type(exc).__name__}: {exc}"
            )
            for path in dataset_output_paths:
                try:
                    if path.exists():
                        path.unlink()
                        print(f"  Removed partial output: {path}")
                except Exception as cleanup_exc:
                    print(
                        f"  Warning: failed to remove partial output {path}: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            continue

    # Save pass-3 metrics summary
    summary_path = Path(args.out_dir) / "summary.csv"
    summary_df = pd.DataFrame(metrics_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote pass-3 metrics summary: {summary_path}")
    if not summary_df.empty:
        print(summary_df.sort_values(["dataset", "algorithm"]).to_string(index=False))

    pass1_summary_path = (
        Path(args.pass1_summary) if args.pass1_summary else (Path(args.input_dir) / "summary.csv")
    )
    if pass1_summary_path.exists():
        pass1_summary = pd.read_csv(pass1_summary_path)
        comparison_df = compare_pass3_vs_pass1_per_algo(summary_df, pass1_summary)
        comparison_path = Path(args.out_dir) / "hv_comparison_vs_pass1.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nWrote pass_3 vs pass_1 comparison: {comparison_path}")
        if not comparison_df.empty:
            print(comparison_df.to_string(index=False))
    else:
        print(f"\nSkipping pass_3 vs pass_1 comparison: missing {pass1_summary_path}")


if __name__ == "__main__":
    main()

