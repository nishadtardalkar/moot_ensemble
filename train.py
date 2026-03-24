from dataloader import DataLoader
from evaluate import Evaluator
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class Trainer:
    def __init__(self, dataloader: DataLoader):
        self.evaluator = Evaluator(dataloader)
        self.dataloader = dataloader

    def train(self):
        algorithm = NSGA2(pop_size=100, n_offsprings=100, n_gen=100)
        result = minimize(
            self.evaluator,
            algorithm,
            termination=get_termination("n_gen", 100),
        )
        return result


if __name__ == "__main__":
    dataloader = DataLoader("moot/behavior_data/all_players.csv")
    trainer = Trainer(dataloader)
    result = trainer.train()
    print(result.X)
    print(result.F)
