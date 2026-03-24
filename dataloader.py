import pandas as pd


class DataLoader:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.objective_cols = []
        self.objective_signs = []
        self.numeric_input_cols = []
        self.symbolic_input_cols = []
        self.col_means = {}
        self.col_stds = {}
        for idx, col in enumerate(self.df.columns):
            if col.endswith("+") or col.endswith("-"):
                self.objective_cols.append(idx)
                self.objective_signs.append(col[-1])
            else:
                if col[0].isupper():
                    self.numeric_input_cols.append(idx - len(self.objective_cols))
                else:
                    self.symbolic_input_cols.append(idx - len(self.objective_cols))
        for col in self.numeric_input_cols:
            self._normalize_numeric_column(self.df.columns[col])

    def _normalize_numeric_column(self, col):
        self.col_means[col] = self.df[col].mean()
        self.col_stds[col] = self.df[col].std()
        self.df[col] = (self.df[col] - self.col_means[col]) / self.col_stds[col]

    def normalize_numeric_x(self, x):
        for col in self.numeric_input_cols:
            x[col] = (x[col] - self.col_means[self.df.columns[col]]) / self.col_stds[
                self.df.columns[col]
            ]
        return x


if __name__ == "__main__":
    dataloader = DataLoader("moot/behavior_data/all_players.csv")
    print(dataloader.df.head())
