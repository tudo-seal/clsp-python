import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections import deque
from typing import Any, TypeVar

from .enumeration import Tree, enumerate_terms
from .fcl import Contains, FiniteCombinatoryLogic
from .grammar import ParameterizedTreeGrammar, RHSRule
from .subtypes import Subtypes
from .types import Literal

NT = TypeVar("NT")  # non-terminals
T = TypeVar("T", covariant=True, bound=Hashable)
V = TypeVar("V")  # codomain of fitness-function. needs to be a poset! which means that compare methods are defined.

"""
This module contains classes and functions for treating 
the inhabitation results (parameterized tree grammars/ sets of horn clauses)  as search spaces.
Searching will be enabled by genetic operations on combinatory terms.
Besides enumerating terms, different sampling methods are implemented.
"""


class Sample(ABC):
    """
    Abstract base class for sampling methods.
    """

    def __init__(self, gamma: Mapping[T, NT], delta: Mapping[str, Iterable[Any] | Contains], target: NT,
                 subtypes: Subtypes = Subtypes({})):
        self.gamma: Mapping[T, NT] = gamma
        self.delta: Mapping[str, Iterable[Any] | Contains] = delta
        self.subtypes: Subtypes = subtypes
        self.fcl = FiniteCombinatoryLogic(gamma, subtypes, delta)
        self.target: NT = target
        self.grammar: ParameterizedTreeGrammar[NT, T] = self.fcl.inhabit(self.target)

    @abstractmethod
    def sample(self, size: int) -> Iterable[Tree[NT, T]]:
        """
        Sample a number of trees from the grammar.
        """

        raise NotImplementedError("sample() must be implemented")


class Enumerate(Sample):
    """
    Enumeration as a sampling method.
    """

    def sample(self, size: int) -> Iterable[Tree[NT, T]]:
        """
        Sample a number of trees from the grammar.
        """
        return enumerate_terms(self.target, self.grammar, max_count=size)


class SampleFromEnumeration(Sample):
    """
    Sample from a finite enumeration of the grammar.
    """

    def sample(self, size: int) -> Iterable[Tree[NT, T]]:
        overfitted = list(enumerate_terms(self.target, self.grammar, max_count=size * 10))
        length = len(overfitted)
        if size > length:
            size = length
        initial = random.sample(overfitted, size)
        return initial


class RandomSample(Sample):
    """
    Random sampling method.
    This sampling methods annotates the grammar with the minimum tree depth of each rule.
    In presence of term predicates in gamma, the minimum tree depth is an over-approximation of the expected tree depth
    and therefore corresponds to a lower bound of the expected tree depth.
    """

    def __init__(self, gamma: Mapping[T, NT], delta: Mapping[str, Iterable[Any] | Contains], target: NT,
                 subtypes: Subtypes = Subtypes({}),
                 tree_depth_delta=None, max_tree_depth=None
                 ):
        super().__init__(gamma, delta, target, subtypes)
        self.min_size: int = self.grammar.minimum_tree_depth(self.target)
        if tree_depth_delta is None:
            self.tree_depth_delta = 100
        else:
            self.tree_depth_delta = tree_depth_delta
        if max_tree_depth is None:
            self.max_tree_depth = self.min_size + self.tree_depth_delta
        else:
            self.max_tree_depth = max_tree_depth
        if self.max_tree_depth < self.min_size:
            raise ValueError(f"max_tree_depth {self.max_tree_depth} is less than minimum tree depth {self.min_size}")
        rules, symbol_depths = self.grammar.annotations()
        self.rules: dict[NT, deque[tuple[RHSRule[NT, T], int]]] = rules
        self.symbol_depths: dict[NT, int] = symbol_depths
        self.cost = 10

    def build_tree(self, nt: NT, cs: int, candidate: RHSRule[NT, T]) -> Tree[NT, T] | None:
        if not list(candidate.non_terminals()):
            if cs + 1 > self.max_tree_depth:
                return None
            # rule only derives terminals
            params: list[Literal] = list(candidate.all_args())
            children: tuple[Tree[NT, T], ...] = tuple(
                map(lambda p: Tree(p.value, derived_from=nt, rhs_rule=candidate, is_literal=True), params))
            if candidate.check([]):
                return Tree(candidate.terminal, children, derived_from=nt, rhs_rule=candidate, is_literal=True)
            else:
                return None
        else:
            # rule derives non-terminals
            children: tuple[Tree[NT, T], ...] = ()
            substitution: dict[str, Tree[NT, T]] = {}
            interleave: Callable[[Mapping[str, Tree[NT, T]]], tuple[Tree[NT, T], ...]] = lambda subs: tuple(
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
                        child_tree = self.sample_random_term(child_nt, new_cs)
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

    def sample_random_term(self, nt: NT, cs: int) -> Tree[NT, T] | None:
        applicable: list[tuple[RHSRule[NT, T], int]] = []
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

    def sample(self, size: int) -> Iterable[Tree[NT, T]]:
        """
        Sample a number of trees from the grammar.
        """
        sample: list[Tree[NT, T]] = []
        for _ in range(size):
            nt = self.target
            cs = 0
            term = self.sample_random_term(nt, cs)
            if term is not None:
                sample.append(term)
        return sample


class Search(Sample):
    """
    Abstract base class for search strategies.
    """

    @abstractmethod
    def search_max(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        """
        Search for a tree with maximum fitness.
        """
        raise NotImplementedError("search_max() must be implemented")

    @abstractmethod
    def search_min(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        """
        Search for a tree with minimum fitness.
        """
        raise NotImplementedError("search_min() must be implemented")

    @abstractmethod
    def search_fittest(self, fitness: Callable[[Tree[NT, T]], V], size: int) -> Iterable[Tree[NT, T]]:
        """
        Return an iterable of the trees with maximum fitness in the population.
        The iterable is sorted in descending order of fitness, such that the tree with
        maximum fitness is first.
        """
        raise NotImplementedError("search_fittest() must be implemented")


class GenerateAndTest(Search):
    """
    Generate and test search method.
    """

    def sample(self, size: int) -> Iterable[Tree[NT, T]]:
        """
        Sample a number of trees from the grammar.
        """
        return enumerate_terms(self.target, self.grammar, max_count=size)

    def search_max(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        """
        Search for a tree with maximum fitness.
        """
        return max(self.sample(size=1000), key=fitness)

    def search_min(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        """
        Search for a tree with minimum fitness.
        """
        return min(self.sample(size=1000), key=fitness)

    def search_fittest(self, fitness: Callable[[Tree[NT, T]], V], size: int) -> Iterable[Tree[NT, T]]:
        """
        Return an iterable of the trees with maximum fitness in the population.
        The iterable is sorted in descending order of fitness, such that the tree with
        maximum fitness is first.
        """
        return sorted(self.sample(size=size), key=fitness, reverse=True)


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies.
    """
    def __init__(self, population_size: int):
        self.size = population_size

    @abstractmethod
    def select(self, evaluated_trees: Mapping[Tree[NT, T], V]) -> Sequence[Tree[NT, T]]:
        """
        Select a number of trees from the population based on their fitness.
        """
        raise NotImplementedError("select() must be implemented")


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.
    """

    def __init__(self, tournament_size: int = 3, population_size: int = None):
        self.tournament_size = tournament_size
        self.size = population_size

    def select(self, evaluated_trees: Mapping[Tree[NT, T], V]) -> Sequence[Tree[NT, T]]:
        """
        Select a number of trees from the population based on their fitness.
        """
        if self.size is None or self.size > len(evaluated_trees):
            self.size = len(evaluated_trees)
        selected: set[Tree[NT, T]] = set()
        while len(selected) < self.size:
            tournament = random.sample(list(evaluated_trees.items()), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.add(winner)
        return list(selected)


class EvolutionaryAlgorithm(Search):
    """
    Abstract class for evolutionary algorithms.
    Since different SelectionStrategies need different hyperparameters, the concrete selection strategy is an argument
    for the constructor and not handled via multiple inheritance.
    """

    def __init__(self, gamma: Mapping[T, NT], delta: Mapping[str, Iterable[Any] | Contains], target: NT,
                 subtypes: Subtypes = Subtypes({}),
                 selection_strategy: SelectionStrategy = TournamentSelection(), generations: int = 5):
        super().__init__(gamma, delta, target, subtypes)
        self.selection_strategy = selection_strategy
        self.generations = generations


class SimpleEA(EvolutionaryAlgorithm, RandomSample):
    """
    This class implements a very simple evolutionary algorithm with tournament selection.
    """

    def search_max(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        return list(self.search_fittest(fitness, 100))[0]

    def search_min(self, fitness: Callable[[Tree[NT, T]], V]) -> Tree[NT, T]:
        return self.search_max(lambda x: -fitness(x))

    def search_fittest(self, fitness: Callable[[Tree[NT, T]], V], size: int) -> Iterable[Tree[NT, T]]:
        # let the [preserved_fittest] fittest individuals survive
        preserved_fittest: int = 3
        self.selection_strategy.size = size
        # Create the initial population
        population: Iterable[Tree[NT, T]] = self.sample(size)
        # Run the genetic algorithm for a number of generations
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            # Select the best individuals for reproduction
            selected: Sequence[Tree[NT, T]] = self.selection_strategy.select(
                {tree: fitness(tree) for tree in population})
            # Create the next generation
            next_generation = []
            pair_length = len(selected) if len(selected) % 2 == 0 else len(selected) - 1
            for i in range(0, pair_length, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                # Perform crossover and mutation to create offspring
                offspring1 = parent1.crossover(parent2, self.grammar)
                offspring2 = offspring1.mutate(self.grammar)
                next_generation.append(offspring1)
                next_generation.append(offspring2)
            # Preserve the fittest individuals from the current generation
            next_generation.extend(sorted(population, key=fitness, reverse=True)[:preserved_fittest])
            # Replace the old population with the new one
            population = next_generation
        # Sort the final population by fitness
        population = sorted(population, key=fitness, reverse=True)
        return population
