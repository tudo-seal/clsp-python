from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections import deque
from typing import  Protocol, runtime_checkable, Any, TypeVar, Generic


from .enumeration import Tree, enumerate_terms
from .fcl import Contains, FiniteCombinatoryLogic
from .grammar import ParameterizedTreeGrammar, RHSRule
from .subtypes import Subtypes
from .types import Literal, Type, Param

from copy import deepcopy
# TODO edit pyproject.toml, or use setup.py or whatever to manage dependencies
import torch
import gpytorch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch import Module
#import networkx as nx
from botorch import fit_gpytorch_mll

from grakel import Graph
from grakel.kernels import (
    RandomWalk,
)

@runtime_checkable
class Comparable(Protocol):
    @abstractmethod
    def __lt__(self: V, other: V) -> bool:
        pass

    @abstractmethod
    def __eq__(self: V, other: object) -> bool:
        pass


T = TypeVar("T", bound=Hashable)
V = TypeVar("V", bound=Comparable)  # codomain of fitness-function. needs to be a poset! which means that compare methods are defined.

"""
This module contains classes and functions for treating
the inhabitation results (parameterized tree grammars/ sets of horn clauses)  as search spaces.
Searching will be enabled by genetic operations on combinatory terms.
Besides enumerating terms, different sampling methods are implemented.
"""


class Sample(ABC, Generic[T]):
    """
    Abstract base class for sampling methods.
    """

    def __init__(self, gamma: Mapping[T, Param | Type], delta: Mapping[str, Iterable[Any] | Contains], target: Type,
                 subtypes: Subtypes = Subtypes({})):
        self.gamma: Mapping[T, Param | Type] = gamma
        self.delta: Mapping[str, Iterable[Any] | Contains] = delta
        self.subtypes: Subtypes = subtypes
        self.fcl = FiniteCombinatoryLogic(gamma, subtypes, delta)
        self.target: Type = target
        self.grammar: ParameterizedTreeGrammar[Type, T] = self.fcl.inhabit(self.target)

    @abstractmethod
    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        """
        Sample a number of trees from the grammar.
        """

        raise NotImplementedError("sample() must be implemented")


class Enumerate(Sample[T]):
    """
    Enumeration as a sampling method.
    """

    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        """
        Sample a number of trees from the grammar.
        """
        return enumerate_terms(self.target, self.grammar, max_count=size)


class SampleFromEnumeration(Sample[T]):
    """
    Sample from a finite enumeration of the grammar.
    """

    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        overfitted = list(enumerate_terms(self.target, self.grammar, max_count=size * 10))
        length = len(overfitted)
        if size > length:
            size = length
        initial = random.sample(overfitted, size)
        return initial


class RandomSample(Sample[T]):
    """
    Random sampling method.
    This sampling methods annotates the grammar with the minimum tree depth of each rule.
    In presence of term predicates in gamma, the minimum tree depth is an over-approximation of the expected tree depth
    and therefore corresponds to a lower bound of the expected tree depth.
    """

    def __init__(self, gamma: Mapping[T, Param | Type], delta: Mapping[str, Iterable[Any] | Contains], target: Type,
                 subtypes: Subtypes = Subtypes({}),
                 tree_depth_delta: int = -1, max_tree_depth: int = -1
                 ):
        super().__init__(gamma, delta, target, subtypes)
        self.min_size: int = self.grammar.minimum_tree_depth(self.target)
        if tree_depth_delta == -1:
            self.tree_depth_delta = 100
        else:
            self.tree_depth_delta = tree_depth_delta
        if max_tree_depth == -1:
            self.max_tree_depth = self.min_size + self.tree_depth_delta
        else:
            self.max_tree_depth = max_tree_depth
        if self.max_tree_depth < self.min_size:
            raise ValueError(f"max_tree_depth {self.max_tree_depth} is less than minimum tree depth {self.min_size}")
        rules, symbol_depths = self.grammar.annotations()
        self.rules: dict[Type, deque[tuple[RHSRule[Type, T], int]]] = rules
        self.symbol_depths: dict[Type, int] = symbol_depths
        self.cost = 10

    def build_tree(self, nt: Type, cs: int, candidate: RHSRule[Type, T]) -> Tree[Type, T] | None:
        if not list(candidate.non_terminals()):
            if cs + 1 > self.max_tree_depth:
                return None
            # rule only derives terminals, therefore all_args has no nonterminals, but to silence the type checker:
            # TODO: how to avoid this unnecessary iteration of all_args AND make mypy happy?
            params: list[Literal] = [lit for lit in candidate.all_args()if isinstance(lit, Literal)]
            cands: tuple[Tree[Type, T], ...] = tuple(
                map(lambda p: Tree(p.value, derived_from=nt, rhs_rule=candidate, is_literal=True), params))
            if candidate.check([]):
                return Tree(candidate.terminal, cands, derived_from=nt, rhs_rule=candidate, is_literal=True)
            else:
                return None
        else:
            # rule derives non-terminals
            children: tuple[Tree[Type, T], ...] = ()
            substitution: dict[str, Tree[Type, T]] = {}
            interleave: Callable[[Mapping[str, Tree[Type, T]]], tuple[Tree[Type, T], ...]] = lambda subs: tuple(
                subs[t] if isinstance(t, str) else t for t in [
                    Tree(p.value, derived_from=nt, rhs_rule=candidate, is_literal=True)
                    if isinstance(p, Literal)
                    else p.name
                    for p in candidate.parameters
                ]
            )
            for _ in range(self.cost):
                for var, child_nt in candidate.binder.items():
                    child_depth = self.symbol_depths.get(child_nt)
                    if child_depth is None:
                        child_depth = self.max_tree_depth
                    if cs + child_depth <= self.max_tree_depth:
                        new_cs = cs + child_depth
                        child_tree: Tree[Type, T] | None = self.sample_random_term(child_nt, new_cs)
                        if child_tree is not None:
                            children = children + (child_tree,)
                            substitution[var] = child_tree
                        else:
                            return None
                    else:
                        return None
                if all(predicate.eval(substitution) for predicate in candidate.predicates):
                    return Tree(
                        candidate.terminal,
                        interleave(substitution),
                        candidate.variable_names,
                        derived_from=nt,
                        rhs_rule=candidate,
                        is_literal=False
                    )
            return None

    def sample_random_term(self, nt: Type, cs: int) -> Tree[Type, T] | None:
        applicable: list[tuple[RHSRule[Type, T], int]] = []
        #for (lhs, rhs), n in self.rules:
        #    new_cs = cs + n
        #    if lhs == nt and new_cs <= self.max_tree_depth:
        #        applicable.append(rhs)
        for rhs, n in self.rules[nt]:
            new_cs = cs + n
            if new_cs <= self.max_tree_depth:
                applicable.append((rhs, new_cs))

        while applicable:
            candidate, next_cs = random.choice(applicable)
            tree = self.build_tree(nt, next_cs, candidate)
            if tree is not None:
                return tree
            applicable.remove((candidate, next_cs))
        return None

    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        """
        Sample a number of trees from the grammar.
        """
        sample: list[Tree[Type, T]] = []
        for _ in range(size):
            nt = self.target
            cs = 0
            term: Tree[Type, T] | None = self.sample_random_term(nt, cs)
            if term is not None:
                sample.append(term)
        return sample


class Search(Sample[T], Generic[T, V]):
    """
    Abstract base class for search strategies.
    """

    @abstractmethod
    def search_max(self, fitness: Callable[[Tree[Type, T]], V]) -> Tree[Type, T]:
        """
        Search for a tree with maximum fitness.
        """
        raise NotImplementedError("search_max() must be implemented")

    @abstractmethod
    def search_fittest(self, fitness: Callable[[Tree[Type, T]], V], size: int) -> Iterable[Tree[Type, T]]:
        """
        Return an iterable of the trees with maximum fitness in the population.
        The iterable is sorted in descending order of fitness, such that the tree with
        maximum fitness is first.
        """
        raise NotImplementedError("search_fittest() must be implemented")


class GenerateAndTest(Search[T, V]):
    """
    Generate and test search method.
    """

    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        """
        Sample a number of trees from the grammar.
        """
        return enumerate_terms(self.target, self.grammar, max_count=size)

    def search_max(self, fitness: Callable[[Tree[Type, T]], V]) -> Tree[Type, T]:
        """
        Search for a tree with maximum fitness.
        """
        return max(self.sample(size=1000), key=fitness)

    def search_fittest(self, fitness: Callable[[Tree[Type, T]], V], size: int) -> Iterable[Tree[Type, T]]:
        """
        Return an iterable of the trees with maximum fitness in the population.
        The iterable is sorted in descending order of fitness, such that the tree with
        maximum fitness is first.
        """
        return sorted(self.sample(size=size), key=fitness, reverse=True)


class SelectionStrategy(ABC, Generic[T, V]):
    """
    Abstract base class for selection strategies.
    """
    def __init__(self, population_size: int | None = None):
        self.size: int = population_size if population_size is not None else 0

    @abstractmethod
    def select(self, evaluated_trees: Mapping[Tree[Type, T], V]) -> Sequence[Tree[Type, T]]:
        """
        Select a number of trees from the population based on their fitness.
        """
        raise NotImplementedError("select() must be implemented")


class TournamentSelection(SelectionStrategy[T, V]):
    """
    Tournament selection strategy.
    """

    def __init__(self, tournament_size: int = 3, population_size: int | None = None):
        super().__init__(population_size)
        self.tournament_size = tournament_size

    def select(self, evaluated_trees: Mapping[Tree[Type, T], V]) -> Sequence[Tree[Type, T]]:
        """
        Select a number of trees from the population based on their fitness.
        """
        if self.size is None or self.size > len(evaluated_trees):
            self.size = len(evaluated_trees)
        selected: set[Tree[Type, T]] = set()
        while len(selected) < self.size:
            tournament = random.sample(list(evaluated_trees.items()), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.add(winner)
        return list(selected)


class EvolutionaryAlgorithm(Search[T, V]):
    """
    Abstract class for evolutionary algorithms.
    Since different SelectionStrategies need different hyperparameters, the concrete selection strategy is an argument
    for the constructor and not handled via multiple inheritance.
    """

    def __init__(self, gamma: Mapping[T, Param | Type], delta: Mapping[str, Iterable[Any] | Contains], target: Type,
                 subtypes: Subtypes = Subtypes({}),
                 selection_strategy: SelectionStrategy[T, V] = TournamentSelection(), generations: int = 5):
        super().__init__(gamma, delta, target, subtypes)
        self.selection_strategy = selection_strategy
        self.generations = generations


class SimpleEA(EvolutionaryAlgorithm[T, V], RandomSample[T]):
    """
    This class implements a very simple evolutionary algorithm with tournament selection.
    """

    def search_max(self, fitness: Callable[[Tree[Type, T]], V]) -> Tree[Type, T]:
        return list(self.search_fittest(fitness, 100))[0]

    def search_fittest(self, fitness: Callable[[Tree[Type, T]], V], size: int) -> Iterable[Tree[Type, T]]:
        # let the [preserved_fittest] fittest individuals survive
        preserved_fittest: int = 3
        self.selection_strategy.size = size
        # Create the initial population
        population: list[Tree[Type, T]] = list(self.sample(size))
        # Run the genetic algorithm for a number of generations
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            # Select the best individuals for reproduction
            selected: Sequence[Tree[Type, T]] = self.selection_strategy.select(
                {tree: fitness(tree) for tree in population})
            # Create the next generation
            next_generation: list[Tree[Type, T]] = []
            pair_length = len(selected) if len(selected) % 2 == 0 else len(selected) - 1
            for i in range(0, pair_length, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                # Perform crossover and mutation to create offspring
                offspring1: Tree[Type, T] | None = parent1.crossover(parent2, self.grammar)
                if offspring1 is None:
                    offspring1 = parent1
                offspring2: Tree[Type, T] | None = offspring1.mutate(self.grammar)
                if offspring2 is None:
                    offspring2 = parent2
                next_generation.append(offspring1)
                next_generation.append(offspring2)
            # Preserve the fittest individuals from the current generation
            next_generation.extend(sorted(population, key=fitness, reverse=True)[:preserved_fittest])
            # Replace the old population with the new one
            population = next_generation
        # Sort the final population by fitness
        population = sorted(population, key=fitness, reverse=True)
        return population

class SIGP(ExactGP):
    """
    A reimplementation of gpytorch's ExactGP that allows for tree inputs.
    The inputs to this class is a Sequence[nx.Graph] instance, which will be transformed into a
    sequence of networkx graphs.

    train_targets need to be a torch.Tensor.

    This class follows is originally from https://github.com/leojklarner/gauche, but is tailored to our CLS usecase.

    In the longer term, if ExactGP can be refactored such that the validation checks ensuring
    that the inputs are torch.Tensors are optional, this class should subclass ExactGP without
    performing those checks.
    """

    def __init__(self, train_inputs: list[Graph], train_targets: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood):
        if (
            train_inputs is not None
            and type(train_inputs) is list[Graph]
        ):
            train_inputs = (train_inputs,)
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("SIGP can only handle Gaussian likelihoods")

        if not isinstance(train_targets, torch.Tensor):
            raise RuntimeError("SIGP can only handle torch.Tensor train_targets.")

        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(
                (
                    i.unsqueeze(-1)   # this case will never be entered, so maybe we just skip it?
                    if torch.is_tensor(i) and i.ndimension() == 1
                    else i
                )
                for i in train_inputs
            )
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    def __call__(self, *args, **kwargs):
        train_inputs = (
            self.train_inputs if self.train_inputs is not None else []
        )

        inputs = [
            (
                i.unsqueeze(-1)
                if torch.is_tensor(i) and i.ndimension() == 1
                else i
            )
            for i in args
        ]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            res = super(ExactGP, self).__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif (
            settings.prior_mode.on()
            or self.train_inputs is None
            or self.train_targets is None
        ):
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "SIGP.forward must return a MultivariateNormal"
                    )
            return full_output

        # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super(ExactGP, self).__call__(
                    *train_inputs, **kwargs
                )

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            if torch.is_tensor(train_inputs[0]):
                batch_shape = train_inputs[0].shape[:-2]
                for train_input, input in zip(train_inputs, inputs):
                    # Make sure the batch shapes agree for training/test data
                    if batch_shape != train_input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(
                            batch_shape, train_input.shape[:-2]
                        )
                        train_input = train_input.expand(
                            *batch_shape, *train_input.shape[-2:]
                        )
                    if batch_shape != input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(
                            batch_shape, input.shape[:-2]
                        )
                        train_input = train_input.expand(
                            *batch_shape, *train_input.shape[-2:]
                        )
                        input = input.expand(*batch_shape, *input.shape[-2:])
                    full_inputs.append(torch.cat([train_input, input], dim=-2))
            else:
                # from IPython.core.debugger import set_trace; set_trace()
                full_inputs = deepcopy(train_inputs)
                full_inputs[0].append(inputs[0])

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "SIGP.forward must return a MultivariateNormal"
                    )
            full_mean, full_covar = (
                full_output.loc,
                full_output.lazy_covariance_matrix,
            )

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size(
                [
                    joint_shape[0] - self.prediction_strategy.train_shape[0],
                    *tasks_shape,
                ]
            )

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(
                    full_mean, full_covar
                )

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(
                *batch_shape, *test_shape
            ).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

class GraphKernel(Module):
    """
    A base class supporting external graph kernels.
    The external kernel must have a method `fit_transform`, which, when
    evaluated on an `Inputs` instance `X`, returns a scaled kernel matrix
    v * k(X, X).

    As gradients are not propagated through to the external kernel, outputs are
    cached to avoid repeated computation.
    """

    def __init__(
        self,
        dtype=torch.float,
    ) -> None:
        super().__init__()
        self.node_label = None
        self.edge_label = None
        self._scale_variance = torch.nn.Parameter(
            torch.tensor([0.1], dtype=dtype)
        )

    def scale(self, S: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self._scale_variance) * S

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scale(self.kernel(X))

    def kernel(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

class RandomWalkKernel(GraphKernel):
    """
    A GraKel wrapper for the random walk kernel.
    This kernel only works on unlabelled graphs.
    See RandomWalkLabeledKernel for labelled graphs.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/random_walk.html
    for more details.
    """

    def __init__(self, dtype=torch.float):
        super().__init__(dtype=dtype)

    # @lru_cache(maxsize=5)
    def kernel(self, X: list[Graph], **grakel_kwargs) -> torch.Tensor:
        # extract required data from the networkx graphs
        # constructed with the Graphein utilities
        # this is cheap and will be cached
        # print(X)
        #X = graph_from_networkx(
        #    X, node_labels_tag=self.node_label, edge_labels_tag=self.edge_label
        #)
        # print(X)
        return torch.tensor(
            RandomWalk(**grakel_kwargs).fit_transform(X)
        ).float()


class GraphGP(SIGP):
    def __init__(
        self,
        train_x: list[Graph],
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: GraphKernel,
    ):
        """
        A subclass of the SIGP class that allows us to use kernels over
        discrete inputs with GPyTorch and BoTorch machinery.

        Parameters:
        -----------
        train_x: list of Graphs
            The training inputs for the model. These are graph objects.
        train_y: torch.Tensor
            The training labels for the model.
        likelihood: gpytorch.likelihoods.Likelihood
            The likelihood function for the model.
        kernel: GraphKernel
            The kernel function for the model.
        **kernel_kwargs:
            The keyword arguments for the kernel function.
        """

        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covariance = kernel

    def forward(self, x):
        """
        A forward pass through the model.
        """
        mean = self.mean(torch.zeros(len(x), 1)).float()
        covariance = self.covariance(x)

        # because graph kernels operate over discrete inputs it might be beneficial
        # to add some jitter for numerical stability
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class BayesianOptimization(Search[T,V]):
    """
    Bayesian optimization for searching trees.
    """

    def __init__(self, model: GraphGP,
                 acquisition_function: Callable[tuple[GraphGP, Tree[Type, T]], V],
                 acquisition_optimizer: Search,
                 gamma: Mapping[T, Type], delta: Mapping[str, Iterable[Any] | Contains], target: Type,
                 subtypes: Subtypes = Subtypes({})):
        super().__init__(gamma, delta, target, subtypes)
        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_optimizer = acquisition_optimizer
        self.train_x: list[Graph] = model.train_inputs
        self.train_y: torch.Tensor = torch.tensor(model.train_targets)  # nope, type mismatch!


    def toTensor(self, y: V) -> torch.Tensor:
        """
        Convert the fitness value to a tensor.
        """
        if isinstance(y, torch.Tensor):
            return y
        elif isinstance(y, (int, float)):
            return torch.tensor([y], dtype=torch.float)
        else:
            raise ValueError(f"Cannot convert {y} to tensor")

def tree_expected_improvement(model: GraphGP, tree: Tree[Type, T]) -> torch.Tensor:
    """
    Compute the negative expected improvement of a tree with respect to the model.
    """
    # xi: float: manual exploration-exploitation trade-off parameter.
    xi: float = 0.0
    x = Graph(tree.to_adjacency_dict())
    from torch.distributions import Normal
    try:
        mu, cov = model.predict(x)
    except:
        return torch.tensor(-1.)  # in case of error. return ei of -1
    std = torch.sqrt(torch.diag(cov))
    mu_star = torch.max(model.y_)
    gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
    u = (mu - mu_star - xi) / std
    ucdf = gauss.cdf(u)
    updf = torch.exp(gauss.log_prob(u))
    ei = std * updf + (mu - mu_star - xi) * ucdf
    return ei

def tree_augmented_expected_improvement(model: GraphGP, tree: Tree[Type, T]) -> torch.Tensor:
    """
    Compute the negative expected improvement of a tree with respect to the model.
    """
    # xi: float: manual exploration-exploitation trade-off parameter.
    xi: float = 0.0
    x = Graph(tree.to_adjacency_dict())
    from torch.distributions import Normal
    try:
        mu, cov = model.predict(x)
    except:
        return torch.tensor(-1.)  # in case of error. return ei of -1
    std = torch.sqrt(torch.diag(cov))
    mu_star = torch.max(model.y_)
    gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
    u = (mu - mu_star - xi) / std
    ucdf = gauss.cdf(u)
    updf = torch.exp(gauss.log_prob(u))
    ei = std * updf + (mu - mu_star - xi) * ucdf
    sigma_n = model.likelihood # type???
    ei *= (1. - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(sigma_n + torch.diag(cov))) # mypy complains about the type of sigma_n!
    return ei

def propose_location(ei_func, surrogate_model: GraphGP, candidates: list[Tree[Type, T]], top_n: int):
    """top_n: return the top n candidates wrt the acquisition function."""
    eis = torch.tensor([ei_func(surrogate_model, candidate) for candidate in candidates])
    _, indicies = eis.topk(top_n)
    xs = [candidates[int(i)] for i in indicies]
    return xs

TRAIN_EPOCHS = 20
LR = 1e-3

class SimpleBO(BayesianOptimization, RandomSample):
    """
    Simple Bayesian optimization for searching trees.
    """
    def initialize_model(self, train_x, train_obj, state_dict=None):
        """
        Initialise model and loss function.

        Args:
            train_x: tensor of inputs
            train_obj: tensor of outputs
            state_dict: current state dict used to speed up fitting

        Returns: mll object, model object
        """

        # define model for objective
        model = GraphGP(
            train_x,
            train_obj,
            likelihood=self.model.likelihood,
            kernel=self.model.covariance,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        model.load_state_dict(self.model.state_dict())

        return mll, model

    def search_max(self, fitness: Callable[[Tree[Type, T]], V]) -> Tree[Type, T]:
        """
        Simple Bayesian Optimization loop.
        """
        init_population = self.sample(5)
        evaluated_trees: dict[Tree[Type, T], V] = {tree: fitness(tree) for tree in init_population}
        # initialize the model with the initial population
        # Define the marginal log likelihood used to optimise the model hyperparameters
        likelihood = self.model.likelihood
        for tree in evaluated_trees.keys():
            print(f'as tree: {tree}')
            print(f'as dict {tree.to_adjacency_dict()}')
        train_x = list(self.train_x) + [Graph(tree.to_adjacency_dict()) for tree in evaluated_trees.keys()]
        train_y = torch.cat((self.train_y, torch.tensor([self.toTensor(y) for y in evaluated_trees.values()])))
        # print(train_x)
        # print(train_y)
        mll_ei, model_ei = self.initialize_model(train_x=train_x, train_obj=train_y) #self.initialize_model(evaluated_trees.keys(), evaluated_trees.values())

        x_next = max(evaluated_trees, key=lambda x: evaluated_trees[x])

        for i in range(100):
            # Use the BoTorch utility for fitting GPs in order
            # to use the LBFGS-B optimiser (recommended), but did not work properly with the non-tensorial inputs
            # fit_gpytorch_model(mll_ei)
            optim = torch.optim.Adam(model_ei.parameters(), lr=LR)
            for j in range(TRAIN_EPOCHS):
                optim.zero_grad()
                output = model_ei(train_x)
                loss = -mll_ei(output, train_y)
                loss.backward()
                optim.step()

            # Get the next point to sample
            x_next: Tree[Type, T] = self.acquisition_optimizer.search_max(
                lambda tree: tree_expected_improvement(model_ei, tree)
            )
            # Evaluate the next point
            y_next: V = fitness(x_next)
            train_x.append(Graph(x_next.to_adjacency_dict()))
            train_y = torch.cat([train_y, self.toTensor(y_next)])
            # Add the new point to the model
            mll_ei, model_ei = self.initialize_model(evaluated_trees.keys(), evaluated_trees.values(),
                                                     model_ei.state_dict())

        self.model = model_ei
        self.train_x = train_x
        self.train_y = train_y
        return x_next

    def search_fittest(self, fitness: Callable[[Tree[Type, T]], V], size: int) -> Iterable[Tree[Type, T]]:
        pass