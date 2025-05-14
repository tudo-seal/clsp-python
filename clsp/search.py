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


class Sample(ABC, Generic[T, V]):
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


class Enumerate(Sample[T, V]):
    """
    Enumeration as a sampling method.
    """

    def sample(self, size: int) -> Iterable[Tree[Type, T]]:
        """
        Sample a number of trees from the grammar.
        """
        return enumerate_terms(self.target, self.grammar, max_count=size)


class SampleFromEnumeration(Sample[T, V]):
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


class RandomSample(Sample[T, V]):
    """
    Random sampling method.
    This sampling methods annotates the grammar with the minimum tree depth of each rule.
    In presence of term predicates in gamma, the minimum tree depth is an over-approximation of the expected tree depth
    and therefore corresponds to a lower bound of the expected tree depth.
    """

    def __init__(self, gamma: Mapping[T, Param | Type], delta: Mapping[str, Iterable[Any] | Contains], target: Type,
                 subtypes: Subtypes = Subtypes({}),
                 tree_depth_delta: int =-1, max_tree_depth: int =-1
                 ):
        super().__init__(gamma, delta, target, subtypes)
        self.min_size: int = self.grammar.minimum_tree_depth(self.target)
        if tree_depth_delta is -1:
            self.tree_depth_delta = 100
        else:
            self.tree_depth_delta = tree_depth_delta
        if max_tree_depth is -1:
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
            params: list[Literal] = [l for l in candidate.all_args()if isinstance(l, Literal)]
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


class Search(Sample[T, V]):
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


class SimpleEA(EvolutionaryAlgorithm[T, V], RandomSample[T, V]):
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
