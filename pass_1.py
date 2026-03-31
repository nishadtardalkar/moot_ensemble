import time
import os
import argparse
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


# ---------------------------------
# CONFIG
# ---------------------------------
CSV_PATH = (
    "moot/behavior_data/all_players.csv"  # default; can be overridden by args/env
)
POP_SIZE = 40  # start smaller for big datasets
N_GEN = 20  # start smaller for big datasets
SEED = 1


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
        description="Run MOOT optimization on a CSV dataset."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=os.environ.get("MOOT_CSV_PATH", CSV_PATH),
        help="Path to CSV file (or env MOOT_CSV_PATH)",
    )
    parser.add_argument(
        "algorithm_id",
        nargs="?",
        type=int,
        default=int(os.environ.get("MOOT_ALGORITHM_ID", "0")),
        help="Algorithm ID (0=NSGA2, 1=MOEA/D) (or env MOOT_ALGORITHM_ID)",
    )
    return parser.parse_args()


args = parse_args()
CSV_PATH = args.csv_file
ALGORITHM_ID = args.algorithm_id

df = pd.read_csv(CSV_PATH)

x_cols, y_cols = parse_moot_columns(df)
X_num, X_cat, num_cols, cat_cols = build_mixed_x(df, x_cols)
Y_original, Y_pymoo = build_objective_matrix(df, y_cols)

print("Loaded:", CSV_PATH)
print("Rows:", len(df))

if len(df) < 1000:
    print(f"Skipping dataset {CSV_PATH} with less than 1000 rows: {len(df)} rows")
    exit(0)

if len(y_cols) < 2:
    print(
        f"Skipping dataset {CSV_PATH} with less than 2 objectives: {len(y_cols)} objectives"
    )
    exit(0)


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


problem = MOOTNearestNeighborProblem(X_num, X_cat, Y_pymoo)

ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=10)
pop_size_ref = max(POP_SIZE, len(ref_dirs))


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
    AGEMOEA(pop_size=POP_SIZE),
    AGEMOEA2(pop_size=POP_SIZE),
    CTAEA(ref_dirs=ref_dirs),
]

if ALGORITHM_ID < 0 or ALGORITHM_ID >= len(algorithms):
    raise ValueError(
        f"Invalid algorithm_id={ALGORITHM_ID}. Valid IDs: 0..{len(algorithms)-1}"
    )

algorithm = algorithms[ALGORITHM_ID]
termination = get_termination("n_gen", N_GEN)

start = time.time()
res = minimize(
    problem,
    algorithm,
    termination,
    seed=SEED,
    verbose=False,
)
runtime = time.time() - start

# ---------------------------------
# Output folder + run tag
# ---------------------------------
algorithm_name = algorithm.__class__.__name__
out_dir = Path("outputs") / algorithm_name
out_dir.mkdir(parents=True, exist_ok=True)
run_tag = f"{CSV_PATH.split('\\')[-1].split('.')[0]}"


# ---------------------------------
# Recover nearest real rows for final solutions
# ---------------------------------
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

# deduplicate repeated nearest-neighbor picks
final_rows["_source_row"] = final_idx
final_rows = final_rows.drop_duplicates(subset=["_source_row"]).reset_index(drop=True)

# Store rows and y's together (objectives are columns in final_rows already)
final_rows.to_csv(out_dir / f"{run_tag}.csv", index=False)
