from typing import Any
import pandas as pd
import numpy as np

import torch
from torch import nn

from gpytorch.likelihoods import GaussianLikelihood

from clsp.dsl import DSL
from clsp.enumeration import Tree, interpret_term
from clsp.types import Constructor, Literal, Param, LVar, Type
from clsp.search import SimpleEA, SimpleBO, GraphGP, RandomWalkKernel, Enumerate, GenerateAndTest

def load_iris(path):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    return train_input, train_labels

class Simple_DNN_Repository:

    def __init__(self, learning_rates: list[float],
                 dimensions: list[int],
                 max_hidden: int,
                 batch_sizes: list[int],
                 epochs: list[int],
                 dataset: torch.utils.data.TensorDataset):
        self.learning_rates = learning_rates
        self.dimensions = dimensions
        self.max_hidden = max_hidden
        self.batch_size = batch_sizes
        self.epochs = epochs
        self.train_dataset = dataset

    def delta(self) -> dict[str, list[Any]]:
        return {
            "bool": [True, False],
            "learning_rate": self.learning_rates,
            "dimension": self.dimensions,
            "hidden": list(range(0, self.max_hidden + 1, 1)),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def gamma(self):
        return {
            "Layer": DSL()
            .Use("n", "dimension")
            .Use("bias", "bool")  # Constructor("Bias", LVar("n")))
            .Use("af", Constructor("activation_function"))
            .In(Constructor("layer", LVar("n"))),

            # "Bias": DSL()
            # .Use("n", "dimension")
            # .In(Constructor("Bias", LVar("n"))),

            "Model": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("l", Constructor("layer", LVar("out")))
            .In(
                Constructor("model",
                            Constructor("input", LVar("in"))
                            & Constructor("output", LVar("out")))
                & Constructor("hidden", Literal(0, "hidden"))
            ),

            "Model_cons": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("neurons", "dimension")
            .Use("n", "hidden")
            .Use("m", "hidden")
            .As(lambda n: n-1)
            .Use("layer", Constructor("layer", LVar("neurons")))
            .Use("model",
                 Constructor("model",
                             Constructor("input", LVar("neurons"))
                             & Constructor("output", LVar("out")))
                 & Constructor("hidden", LVar("m"))
                 )
            .In(
                Constructor("model",
                            Constructor("input", LVar("in"))
                            & Constructor("output", LVar("out"))
                            )
                & Constructor("hidden", LVar("n"))
            ),

            "ReLu": Constructor("activation_function"),

            "ELU": Constructor("activation_function"),

            "Sigmoid": Constructor("activation_function"),

            "MSE": Constructor("loss_function"),

            "CrossEntropy": Constructor("loss_function"),

            "L1": Constructor("loss_function"),

            "DataLoader": DSL()
            .Use("bs", "batch_size")
            .In(Constructor("data", Constructor("batch_size", LVar("bs")))),

            "Adagrad": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("optimizer", Constructor("learning_rate", LVar("lr")))),

            "Adam": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("optimizer", Constructor("learning_rate", LVar("lr")))),

            "SGD": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("optimizer", Constructor("learning_rate", LVar("lr")))),

            "System": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("n", "hidden")
            .Use("lr", "learning_rate")
            .Use("ep", "epochs")
            .Use("bs", "batch_size")
            .Use("data", Constructor("data", Constructor("batch_size", LVar("bs"))))
            .Use("m", Constructor("model", Constructor("input", LVar("in")) & Constructor("output", LVar("out"))) & Constructor("hidden", LVar("n")))
            .Use("opt", Constructor("optimizer", Constructor("learning_rate", LVar("lr"))))
            .Use("l", Constructor("loss_function"))
            .In(
                Constructor("system",
                            Constructor("input_dim", LVar("in"))
                            & Constructor("output_dim", LVar("out"))
                            )
                & Constructor("learning_rate", LVar("lr"))
                & Constructor("hidden_layer", LVar("n"))
                & Constructor("epochs", LVar("ep"))
                & Constructor("batch_size", LVar("bs"))
            ),
        }

    @staticmethod
    def train_loop(dataloader, model, loss_fn, opti, batch_size):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        # print(f"Model structure: {model}\n\n")
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.float()
            y = y.float()
            pred = model(X)
            loss = loss_fn(pred, y)

            # build the optimizer
            optimizer = opti(model.parameters())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    @staticmethod
    def test_loop(dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                X = X.float()
                y = y.float()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 100 * correct, test_loss

    def system(self, dataloader, model, loss_fn, optimizer, batch_size, epochs):
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs} \n -------------------------------")
            self.train_loop(dataloader, model, loss_fn, optimizer, batch_size)
            #  self.test_loop(dataloader, model, loss_fn)
        return self.test_loop(dataloader, model, loss_fn)

    def torch_algebra(self):
        return {
            "System": (lambda i, o, n, lr, ep, bs, data, m, opt, l: self.system(data, m, l, opt, bs, ep)),
            "Layer": (lambda n, b, af: (b, af)),
            "Model": (lambda i, o, l: nn.Sequential(nn.Linear(i, o, l[0]), nn.Softmax(dim=1))),  # , l[1])),
            "Model_cons": (lambda i, o, neurons, n, m, l, model:
                           nn.Sequential(nn.Linear(i, neurons, l[0]), l[1]).extend(model)),
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Sigmoid": nn.Sigmoid(),
            "MSE": nn.MSELoss(),
            "CrossEntropy": nn.CrossEntropyLoss(),
            "L1": nn.L1Loss(),
            "Adagrad": (lambda lr, params: torch.optim.Adagrad(params, lr=lr)),
            "Adam": (lambda lr, params: torch.optim.Adam(params, lr=lr)),
            "SGD": (lambda lr, params: torch.optim.SGD(params, lr=lr)),
            "DataLoader": (lambda bs: torch.utils.data.DataLoader(self.train_dataset, batch_size=bs)),
        }

train_input, train_labels = load_iris('./data/iris.csv')
x_train_tensor = torch.tensor(train_input)
y_train_tensor = torch.tensor(train_labels)

dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

repo = Simple_DNN_Repository([0.01], [3, 4] + list(range(5, 30, 5)), 5, [8], [100], dataset)

target = (Constructor("system",
                      Constructor("input_dim", Literal(4, "dimension"))
                      & Constructor("output_dim", Literal(3, "dimension"))
                      )
          & Constructor("learning_rate", Literal(0.01, "learning_rate"))
          #& Constructor("hidden_layer", Literal(3, "hidden"))
          #& Constructor("epochs", Literal(100, "epochs"))
          #& Constructor("batch_size", Literal(8, "batch_size"))
          )

model = GraphGP(
        train_x=list(),
        train_y=torch.tensor([]),
        likelihood=GaussianLikelihood(),
        kernel=RandomWalkKernel()
    )

test = SimpleBO(model, SimpleEA(repo.gamma(), repo.delta(), target, generations=5), repo.gamma(), repo.delta(), target, cost=50, init_population_size=50) # GenerateAndTest(repo.gamma(), repo.delta(), target)


opt = test.search_max(lambda t: torch.tensor(interpret_term(t, repo.torch_algebra())[0]))

print(interpret_term(opt, repo.torch_algebra()))


# generate and test for 100 terms: (85.33333333333334, 0.7050976565009669)
