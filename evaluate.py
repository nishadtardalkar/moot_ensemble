from pymoo.core.problem import ElementwiseProblem
from dataloader import DataLoader


class Evaluator(ElementwiseProblem):
    def __init__(
        self,
        dataloader: DataLoader,
    ):
        self.dataloader = dataloader
        super().__init__(
            n_var=len(dataloader.numeric_input_cols)
            + len(dataloader.symbolic_input_cols),
            n_obj=len(dataloader.objective_cols),
            n_ieq_constr=0,
        )

    def distX(self, x, p):
        dist = 0
        for col in self.dataloader.numeric_input_cols:
            dist += (x[col] - p[col]) ** 2
        for col in self.dataloader.symbolic_input_cols:
            dist += int(x[col] != p[col])
        return dist

    def min_distX(self, x):
        dists = [self.distX(x, p) for p in self.dataloader.df.itertuples()]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)
        return min_dist, min_idx

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.dataloader.normalize_numeric_x(x)
        _, row_idx = self.min_distX(x)
        row = self.dataloader.df.iloc[row_idx]
        out["F"] = [
            row[self.dataloader.df.columns[col]]
            for col in self.dataloader.objective_cols
        ]
