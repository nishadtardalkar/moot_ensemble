import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

import os
print("Running from:", os.getcwd())

# ---------------------------------
# CONFIG
# ---------------------------------
CSV_PATH = "moot/optimize/behavior_data/all_players.csv"   # change to your MOOT file
POP_SIZE = 40     # start smaller for big datasets
N_GEN = 20        # start smaller for big datasets
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
        raise ValueError("No MOOT objective columns found. Expected headers ending in '+' or '-'.")

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
df = pd.read_csv(CSV_PATH)

x_cols, y_cols = parse_moot_columns(df)
X_num, X_cat, num_cols, cat_cols = build_mixed_x(df, x_cols)
Y_original, Y_pymoo = build_objective_matrix(df, y_cols)

print("Loaded:", CSV_PATH)
print("Decision columns:", x_cols)
print("Objective columns:", y_cols)
print("Rows:", len(df))
print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))
print("Numeric matrix shape:", None if X_num is None else X_num.shape)
print("Categorical matrix shape:", None if X_cat is None else X_cat.shape)


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
                cand_num = x[i, :self.n_num]
                dist += ((self.X_num - cand_num) ** 2).sum(axis=1)

            if self.n_cat > 0:
                cand_cat = x[i, self.n_num:]
                cand_cat = np.rint(cand_cat).astype(np.int32)
                dist += (self.X_cat != cand_cat).sum(axis=1)

            nn_idx[i] = np.argmin(dist)

        out["F"] = self.Y_lookup[nn_idx]


problem = MOOTNearestNeighborProblem(X_num, X_cat, Y_pymoo)

algorithm = NSGA2(pop_size=POP_SIZE)
#ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

'''algorithm = MOEAD(
    ref_dirs=ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7
)'''
termination = get_termination("n_gen", N_GEN)

start = time.time()
res = minimize(
    problem,
    algorithm,
    termination,
    seed=SEED,
    verbose=True,
)
runtime = time.time() - start


# ---------------------------------
# Recover nearest real rows for final solutions
# ---------------------------------
final_idx = []

for i in range(res.X.shape[0]):
    dist = np.zeros(len(df), dtype=np.float32)

    if X_num is not None:
        cand_num = res.X[i, :len(num_cols)]
        dist += ((X_num - cand_num) ** 2).sum(axis=1)

    if X_cat is not None:
        cand_cat = res.X[i, len(num_cols):]
        cand_cat = np.rint(cand_cat).astype(np.int32)
        dist += (X_cat != cand_cat).sum(axis=1)

    final_idx.append(np.argmin(dist))

final_idx = np.array(final_idx)

final_rows = df.iloc[final_idx].copy().reset_index(drop=True)
final_objectives = pd.DataFrame(Y_original[final_idx], columns=y_cols)

# deduplicate repeated nearest-neighbor picks
final_rows["_source_row"] = final_idx
final_rows = final_rows.drop_duplicates(subset=["_source_row"]).reset_index(drop=True)

# ---------------------------------
# Metrics: Hypervolume + Spread
# ---------------------------------

# take the objective values from the rows we found
F_metrics = final_rows[y_cols].copy().to_numpy(dtype=float)

# convert to minimization form
for j, c in enumerate(y_cols):
    if c.endswith("+"):
        F_metrics[:, j] = -F_metrics[:, j]

# -------- NORMALIZE OBJECTIVES --------
mins = F_metrics.min(axis=0)
maxs = F_metrics.max(axis=0)

denom = (maxs - mins)
denom[denom == 0] = 1  # avoid divide-by-zero

F_norm = (F_metrics - mins) / denom

# hypervolume reference point (slightly worse than worst point)
ref_point = np.ones(F_norm.shape[1]) * 1.1

hv = HV(ref_point=ref_point)
hypervolume = hv(F_norm)



def spread_mean_distance(F):
    """
    Spread metric based on mean pairwise distance.

    F should be:
    - normalized
    - minimization form
    """

    n = len(F)

    if n < 2:
        return np.nan

    # compute full distance matrix
    dist_matrix = np.sqrt(((F[:, None, :] - F[None, :, :])**2).sum(axis=2))

    # take upper triangle (ignore diagonal)
    i, j = np.triu_indices(n, k=1)
    distances = dist_matrix[i, j]

    return float(np.mean(distances))

# spread (use normalized front)
spread = spread_mean_distance(F_norm)


print("\nPareto metrics:")
print("Hypervolume:", round(float(hypervolume), 6))
print("Spread:", "N/A for >2 objectives" if np.isnan(spread) else round(float(spread), 6))
print("Non-dominated final rows used for metrics:", len(F_norm))


# ---------------------------------
# Optional: plot if exactly 2 objectives
# ---------------------------------
'''if len(y_cols) == 2:
    plot_idx = final_rows["_source_row"].to_numpy()
    print(final_rows["_source_row"].shape)
    F_plot = Y_original[plot_idx]

    plt.figure(figsize=(6, 5))
    plt.scatter(F_plot[:, 0], F_plot[:, 1])
    plt.xlabel(y_cols[0])
    plt.ylabel(y_cols[1])
    plt.title("NSGA-II on MOOT dataset")
    plt.tight_layout()
    plt.show()'''

# ---------------------------------
# Plot full frontier (optimizer solutions)
# ---------------------------------
if len(y_cols) == 2:

    F = res.F.copy()

    # convert back from minimization form
    for j, c in enumerate(y_cols):
        if c.endswith("+"):
            F[:, j] = -F[:, j]

    plt.figure(figsize=(6,5))

    # optimizer frontier
    plt.scatter(F[:,0], F[:,1], s=20, label="Optimizer Frontier")

    # dataset rows we matched
    dataset_points = final_rows[y_cols].to_numpy()
    #plt.scatter(dataset_points[:,0], dataset_points[:,1],s=60, color="red", label="Dataset Matches")

    plt.xlabel(y_cols[0])
    plt.ylabel(y_cols[1])
    plt.title("MOEA/D Pareto Frontier")
    plt.legend()
    plt.tight_layout()
    plt.show()
