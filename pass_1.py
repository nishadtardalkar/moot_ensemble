import time
import os
import argparse
import glob
from pathlib import Path
from math import comb
import numpy as np
import pandas as pd

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions

from get_metrics import calculate_pareto_metrics


# ---------------------------------
# Pareto metrics (same helpers as pass_2)
# ---------------------------------
def parse_objective_columns(columns):
    obj_cols = [c for c in columns if c.endswith(("+", "-"))]
    maximize_mask = np.array([c.endswith("+") for c in obj_cols], dtype=bool)
    return obj_cols, maximize_mask


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


def normalize_dataset_name(dataset_value):
    return Path(str(dataset_value)).stem


def parse_algorithm_name(path_value):
    return Path(str(path_value)).stem


def to_numeric_objectives(df, obj_cols):
    out = df[obj_cols].copy()
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=obj_cols)
    return out.to_numpy(dtype=float)


# ---------------------------------
# CONFIG
# ---------------------------------
POP_SIZE = 40  # start smaller for big datasets
N_GEN = 20  # start smaller for big datasets
SEED = 1
OUTPUT_ROOT = Path("pass_1_outputs")

# CSV stems (filename without .csv) to skip. Also use --skip-datasets on the CLI.
SKIP_DATASETS = frozenset()


# ---------------------------------
# MOOT helpers
# ---------------------------------
def parse_moot_columns(df: pd.DataFrame):
    """
    MOOT convention:
    - y/objective columns end with '+' or '-'
    - x/decision columns are columns not ending in X, +, or -
    - columns starting with uppercase are numeric in the ezr tooling
    """
    y_cols = [c for c in df.columns if c.endswith(("+", "-"))]
    x_cols = [c for c in df.columns if not c.endswith(("X", "+", "-"))]

    if not y_cols:
        raise ValueError(
            "No MOOT objective columns found. Expected headers ending in '+' or '-'."
        )

    if not x_cols:
        raise ValueError("No MOOT decision columns found.")

    return x_cols, y_cols


def is_moot_numeric(col_name: str) -> bool:
    """ezr convention: uppercase first letter => numeric."""
    return len(col_name) > 0 and col_name[0].isupper()


def build_mixed_x(df: pd.DataFrame, x_cols):
    """
    Keep numeric and categorical columns separately.

    Numeric:
      - converted to float
      - missing values filled
      - min-max normalized to [0, 1]

    Categorical:
      - converted to integer codes
      - later compared using Hamming distance (0 if same, 1 if different)
    """
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
    """
    pymoo minimizes all objectives.
    MOOT:
    - '-' means minimize already
    - '+' means maximize, so negate for pymoo
    """
    y = df[y_cols].copy()

    for c in y_cols:
        y[c] = pd.to_numeric(y[c], errors="coerce")

    if y.isnull().any().any():
        raise ValueError("Objective columns contain non-numeric or missing values.")

    y_for_pymoo = y.copy()
    for c in y_cols:
        if c.endswith("+"):
            y_for_pymoo[c] = -y_for_pymoo[c]

    return y.values, y_for_pymoo.values


# ---------------------------------
# Load MOOT dataset
# ---------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MOOT optimization (all algorithms by default). "
            "With no CSV path and no MOOT_CSV_PATH env, discovers inputs via --glob."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help=(
            "Optional path to one CSV. If omitted and MOOT_CSV_PATH is unset, "
            "uses --glob / --min-rows discovery instead."
        ),
    )
    parser.add_argument(
        "--glob",
        dest="input_glob",
        default="moot/optimize/**/*.csv",
        help="When no input CSV is set, discover CSVs with this pattern (default: moot/optimize/**/*.csv).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="When using discovery, skip CSVs where len(df) <= this value (default: 10000).",
    )
    parser.add_argument(
        "--algorithm-id",
        type=int,
        default=None,
        help="Run only this algorithm (0..8). Default: run all algorithms.",
    )
    parser.add_argument(
        "--data-roots",
        nargs="*",
        default=["moot/behavior_data", "moot/optimize/behavior_data", "."],
        help="Directories to search for source dataset CSV (for IGD reference).",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="*",
        default=["Marketing_Analytics.csv"],
        help=(
            "Extra dataset names to skip: CSV stem(s) or path(s); "
            "merged with SKIP_DATASETS in pass_1.py."
        ),
    )
    parser.add_argument(
        "--compare-algorithm",
        default=None,
        help=(
            "Optional algorithm name (e.g., NSGA2) to compare against all other pass_1 "
            "algorithms using pass_3-style pairwise better-percentage metrics."
        ),
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help=(
            "Only run comparison from an existing pass_1 summary CSV. "
            "Skips all optimization runs."
        ),
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help=(
            "Summary CSV to use for --comparison-only "
            "(default: pass_1_outputs/summary.csv)."
        ),
    )
    return parser.parse_args()


def merged_skip_stems(args):
    stems = {s.lower() for s in SKIP_DATASETS}
    for x in args.skip_datasets or []:
        stems.add(Path(x).stem.lower())
    return stems


def path_is_skipped(path, skip_stems):
    return Path(path).stem.lower() in skip_stems


# ---------------------------------
# Problem adapter
# ---------------------------------
class MOOTNearestNeighborProblem(Problem):
    """
    NSGA-II searches in a mixed space:
    - numeric variables in [0,1]
    - categorical variables as integer codes

    Nearest-neighbor distance:
    - numeric part: squared Euclidean distance
    - categorical part: Hamming distance (0 if same, 1 if different)
    """

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

        super().__init__(
            n_var=self.n_num + self.n_cat,
            n_obj=Y_lookup.shape[1],
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_candidates = x.shape[0]
        nn_idx = np.empty(n_candidates, dtype=int)

        for i in range(n_candidates):
            dist = np.zeros(len(self.Y_lookup), dtype=np.float32)

            if self.n_num > 0:
                cand_num = x[i, : self.n_num]
                dist += ((self.X_num - cand_num) ** 2).sum(axis=1)

            if self.n_cat > 0:
                cand_cat = x[i, self.n_num :]
                cand_cat = np.rint(cand_cat).astype(np.int32)
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


def metrics_summary_row(
    csv_path: str,
    data_roots,
    run_tag: str,
    out_path: Path,
    final_rows: pd.DataFrame,
    merged_rows: int,
):
    obj_cols_metrics, maximize_mask_metrics = parse_objective_columns(
        final_rows.columns
    )
    frontier_values = None
    if obj_cols_metrics and len(final_rows) > 0:
        fv = to_numeric_objectives(final_rows, obj_cols_metrics)
        if len(fv) > 0:
            frontier_values = fv

    if frontier_values is None:
        return None

    reference_values = None
    source_path = find_source_csv(Path(csv_path).name, data_roots)
    if source_path is not None:
        source_df = pd.read_csv(source_path)
        if all(c in source_df.columns for c in obj_cols_metrics):
            source_values = to_numeric_objectives(source_df, obj_cols_metrics)
            if len(source_values) > 0:
                source_min = source_values.copy()
                source_min[:, maximize_mask_metrics] = -source_min[
                    :, maximize_mask_metrics
                ]
                source_nd = non_dominated_mask(source_min)
                reference_values = source_values[source_nd]

    metrics = calculate_pareto_metrics(
        frontier=frontier_values,
        reference_front=reference_values,
        maximize_mask=maximize_mask_metrics,
        normalize=True,
    )

    return {
        "algorithm": parse_algorithm_name(out_path),
        "dataset": f"{run_tag}.csv",
        "input_files": 1,
        "merged_rows": int(merged_rows),
        "pareto_rows": int(len(final_rows)),
        "reference_rows": int(len(reference_values))
        if reference_values is not None
        else 0,
        "hv": metrics["hv"],
        "spread": metrics["spread"],
        "igd": metrics["igd"],
        "mdh": metrics["mdh"],
        "output_file": str(out_path),
    }


def compare_pass1_algorithm_vs_others(
    pass1_summary: pd.DataFrame, target_algorithm: str
):
    """
    For one pass_1 algorithm, compute percent-better metrics against all other
    pass_1 algorithms on the same datasets:
    - hv_better_pct: target hv is higher
    - spread_better_pct: target spread is higher
    - igd/mdh_better_pct: target is lower
    """
    required_cols = {"dataset", "hv", "spread", "igd", "mdh", "output_file"}
    if not required_cols.issubset(set(pass1_summary.columns)):
        raise ValueError(
            "pass_1 summary must contain 'dataset', 'hv', 'spread', 'igd', "
            "'mdh', and 'output_file' columns."
        )

    p1 = pass1_summary.copy()
    if "algorithm" not in p1.columns:
        p1["algorithm"] = p1["output_file"].map(parse_algorithm_name)

    p1["dataset_key"] = p1["dataset"].map(normalize_dataset_name)
    p1["hv"] = pd.to_numeric(p1["hv"], errors="coerce")
    p1["spread"] = pd.to_numeric(p1["spread"], errors="coerce")
    p1["igd"] = pd.to_numeric(p1["igd"], errors="coerce")
    p1["mdh"] = pd.to_numeric(p1["mdh"], errors="coerce")

    target_rows = p1[p1["algorithm"] == target_algorithm]
    if target_rows.empty:
        return pd.DataFrame(
            [
                {
                    "algorithm": target_algorithm,
                    "hv_better_pct": np.nan,
                    "spread_better_pct": np.nan,
                    "igd_better_pct": np.nan,
                    "mdh_better_pct": np.nan,
                    "comparisons": 0,
                }
            ]
        )

    hv_better_total = 0
    spread_better_total = 0
    igd_better_total = 0
    mdh_better_total = 0
    hv_cmp_total = 0
    spread_cmp_total = 0
    igd_cmp_total = 0
    mdh_cmp_total = 0

    for _, row in target_rows.iterrows():
        dataset_rows = p1[p1["dataset_key"] == row["dataset_key"]]
        others = dataset_rows[dataset_rows["algorithm"] != target_algorithm]

        other_hv = others["hv"].dropna()
        other_spread = others["spread"].dropna()
        other_igd = others["igd"].dropna()
        other_mdh = others["mdh"].dropna()
        if other_hv.empty or other_spread.empty or other_igd.empty or other_mdh.empty:
            continue

        if pd.isna(row["hv"]) or pd.isna(row["spread"]) or pd.isna(row["igd"]) or pd.isna(row["mdh"]):
            continue

        hv_cmp_total += int(len(other_hv))
        spread_cmp_total += int(len(other_spread))
        igd_cmp_total += int(len(other_igd))
        mdh_cmp_total += int(len(other_mdh))
        hv_better_total += int((row["hv"] > other_hv).sum())
        spread_better_total += int((row["spread"] > other_spread).sum())
        igd_better_total += int((row["igd"] < other_igd).sum())
        mdh_better_total += int((row["mdh"] < other_mdh).sum())

    comparisons = min(hv_cmp_total, spread_cmp_total, igd_cmp_total, mdh_cmp_total)
    if comparisons == 0:
        return pd.DataFrame(
            [
                {
                    "algorithm": target_algorithm,
                    "hv_better_pct": np.nan,
                    "spread_better_pct": np.nan,
                    "igd_better_pct": np.nan,
                    "mdh_better_pct": np.nan,
                    "comparisons": 0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "algorithm": target_algorithm,
                "hv_better_pct": float(100.0 * hv_better_total / hv_cmp_total),
                "spread_better_pct": float(
                    100.0 * spread_better_total / spread_cmp_total
                ),
                "igd_better_pct": float(100.0 * igd_better_total / igd_cmp_total),
                "mdh_better_pct": float(100.0 * mdh_better_total / mdh_cmp_total),
                "comparisons": int(comparisons),
            }
        ]
    )


def compare_all_pass1_algorithms_vs_others(pass1_summary: pd.DataFrame):
    p1 = pass1_summary.copy()
    if "algorithm" not in p1.columns:
        if "output_file" not in p1.columns:
            raise ValueError(
                "pass_1 summary must contain either 'algorithm' or 'output_file' column."
            )
        p1["algorithm"] = p1["output_file"].map(parse_algorithm_name)

    algorithms = sorted(a for a in p1["algorithm"].dropna().unique())
    if not algorithms:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "hv_better_pct",
                "spread_better_pct",
                "igd_better_pct",
                "mdh_better_pct",
                "comparisons",
            ]
        )

    rows = [
        compare_pass1_algorithm_vs_others(pass1_summary, algorithm)
        for algorithm in algorithms
    ]
    return pd.concat(rows, ignore_index=True)


def write_summary_and_optional_comparison(summary, args):
    summary_path = OUTPUT_ROOT / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote combined summary: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))

    if args.compare_algorithm:
        comparison = compare_pass1_algorithm_vs_others(summary, args.compare_algorithm)
        comparison_path = OUTPUT_ROOT / "comparison_vs_other_algorithms.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Wrote pass_1 algorithm comparison: {comparison_path}")
        print(comparison.to_string(index=False))
    elif args.comparison_only:
        comparison = compare_all_pass1_algorithms_vs_others(summary)
        comparison_path = OUTPUT_ROOT / "comparison_vs_other_algorithms.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Wrote pass_1 all-algorithms comparison: {comparison_path}")
        print(comparison.to_string(index=False))


def run_comparison_only(args):
    summary_path = Path(args.summary_path) if args.summary_path else (OUTPUT_ROOT / "summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found for comparison-only mode: {summary_path}")

    summary = pd.read_csv(summary_path)
    if args.compare_algorithm:
        comparison = compare_pass1_algorithm_vs_others(summary, args.compare_algorithm)
        label = "algorithm"
    else:
        comparison = compare_all_pass1_algorithms_vs_others(summary)
        label = "all-algorithms"
    comparison_path = OUTPUT_ROOT / "comparison_vs_other_algorithms.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Loaded existing summary: {summary_path}")
    print(f"Wrote pass_1 {label} comparison: {comparison_path}")
    print(comparison.to_string(index=False))


def run_single_dataset(csv_path: str, data_roots, args):
    df = pd.read_csv(csv_path)

    x_cols, y_cols = parse_moot_columns(df)
    X_num, X_cat, num_cols, cat_cols = build_mixed_x(df, x_cols)
    Y_original, Y_pymoo = build_objective_matrix(df, y_cols)

    print("Loaded:", csv_path)
    print("Rows:", len(df))

    if len(df) < 1000:
        print(f"Skipping dataset {csv_path} with less than 1000 rows: {len(df)} rows")
        return []

    if len(y_cols) < 2:
        print(
            f"Skipping dataset {csv_path} with less than 2 objectives: {len(y_cols)} objectives"
        )
        return []

    problem = MOOTNearestNeighborProblem(X_num, X_cat, Y_pymoo)

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=10)
    pop_size_ref = max(POP_SIZE, len(ref_dirs))

    algorithms = [
        NSGA2(pop_size=POP_SIZE),
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
                max(2, int(np.ceil(POP_SIZE / max(1, min(2, len(ref_dirs)))))),
            ),
        ),
        RVEA(ref_dirs=ref_dirs, pop_size=pop_size_ref),
        AGEMOEA(pop_size=POP_SIZE, eliminate_duplicates=True),
        AGEMOEA2(pop_size=POP_SIZE, eliminate_duplicates=True),
        CTAEA(ref_dirs=ref_dirs),
    ]

    if args.algorithm_id is not None:
        algo_ids = [args.algorithm_id]
    elif os.environ.get("MOOT_ALGORITHM_ID") is not None:
        algo_ids = [int(os.environ["MOOT_ALGORITHM_ID"])]
    else:
        algo_ids = list(range(len(algorithms)))

    termination = get_termination("n_gen", N_GEN)
    run_tag = Path(csv_path).stem

    summary_rows = []

    written_dataset_files = []

    for aid in algo_ids:
        if aid < 0 or aid >= len(algorithms):
            raise ValueError(
                f"Invalid algorithm_id={aid}. Valid IDs: 0..{len(algorithms)-1}"
            )

        algorithm = algorithms[aid]
        algorithm_name = algorithm.__class__.__name__
        print(f"\n--- Algorithm {aid}: {algorithm_name} ---")

        out_dir = OUTPUT_ROOT / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{algorithm_name}.csv"

        if out_path.exists():
            print(f"Skipping existing output for {run_tag}/{algorithm_name}: {out_path}")
            try:
                final_rows = pd.read_csv(out_path)
                row = metrics_summary_row(
                    csv_path,
                    data_roots,
                    run_tag,
                    out_path,
                    final_rows,
                    merged_rows=len(final_rows),
                )
                if row is not None:
                    summary_rows.append(row)
            except Exception as exc:
                print(
                    f"Warning: could not summarize existing output {out_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue

        try:
            start = time.time()
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=SEED,
                verbose=False,
            )
            runtime = time.time() - start

            final_idx = []
            for i in range(res.X.shape[0]):
                dist = np.zeros(len(df), dtype=np.float32)

                if X_num is not None:
                    cand_num = res.X[i, : len(num_cols)]
                    dist += ((X_num - cand_num) ** 2).sum(axis=1)

                if X_cat is not None:
                    cand_cat = res.X[i, len(num_cols) :]
                    cand_cat = np.rint(cand_cat).astype(np.int32)
                    dist += (X_cat != cand_cat).sum(axis=1)

                final_idx.append(np.argmin(dist))

            final_idx = np.array(final_idx)

            final_rows = df.iloc[final_idx].copy().reset_index(drop=True)

            final_rows["_source_row"] = final_idx
            final_rows = final_rows.drop_duplicates(subset=["_source_row"]).reset_index(
                drop=True
            )

            final_rows.to_csv(out_path, index=False)
            written_dataset_files.append(out_path)
            print(f"Wrote {len(final_rows)} rows -> {out_path} ({runtime:.1f}s)")

            row = metrics_summary_row(
                csv_path,
                data_roots,
                run_tag,
                out_path,
                final_rows,
                merged_rows=int(res.X.shape[0]),
            )
            if row is not None:
                summary_rows.append(row)
        except Exception as exc:
            print(
                f"Discarding dataset {run_tag}: algorithm {algorithm_name} failed with "
                f"{type(exc).__name__}: {exc}"
            )
            for path in written_dataset_files:
                try:
                    if path.exists():
                        path.unlink()
                        print(f"Removed partial output: {path}")
                except Exception as cleanup_exc:
                    print(
                        f"Warning: failed to remove partial output {path}: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            return []

    return summary_rows


def main():
    args = parse_args()
    data_roots = args.data_roots
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.comparison_only:
        run_comparison_only(args)
        return

    all_summary_rows = []

    skip_stems = merged_skip_stems(args)
    csv_path = args.csv_file or os.environ.get("MOOT_CSV_PATH")
    if csv_path is not None:
        if path_is_skipped(csv_path, skip_stems):
            print(f"Skipping excluded dataset: {csv_path}")
        else:
            all_summary_rows.extend(run_single_dataset(csv_path, data_roots, args))
        summary = pd.DataFrame(all_summary_rows)
        write_summary_and_optional_comparison(summary, args)
        return

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    for path in paths:
        if path_is_skipped(path, skip_stems):
            print(f"Skipping excluded dataset: {path}")
            continue
        df_len = len(pd.read_csv(path))
        if df_len <= args.min_rows:
            continue
        print("--------------------------------")
        all_summary_rows.extend(run_single_dataset(path, data_roots, args))
        print()

    summary = pd.DataFrame(all_summary_rows)
    write_summary_and_optional_comparison(summary, args)


if __name__ == "__main__":
    main()
