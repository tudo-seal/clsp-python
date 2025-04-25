import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import truediv
from typing import Any, Generic, Optional, TypeVar, overload
from .enumeration import Tree, enumerate_terms
from .grammar import ParameterizedTreeGrammar

NT = TypeVar("NT")  # non-terminals
T = TypeVar("T", covariant=True, bound=Hashable)
V = TypeVar("V")  # codomain of fitness-function. needs to be a poset!


@dataclass
class Fitness(Generic[NT, T, V]):
    fit: Callable[[Tree[NT, T]], V]
    name: str = field(default="fit")
    ordering: Callable[[V, V], bool] = field(default=lambda x, y: x < y)

    def eval(self, tree: Tree[NT, T]) -> V:
        return self.fit(tree)

    def __str__(self) -> str:
        return f"{self.name}"


# Create the initial population
# TODO: this is not a good way to create the initial population. It should be done by sampling from the grammar
def enumerate_initial_population(target: NT, grammar: ParameterizedTreeGrammar[NT, T], size: int) -> Sequence[Tree[NT, T]]:
    overfitted = list(enumerate_terms(target, grammar, max_count=100000))
    length = len(overfitted)
    if size > length:
        size = length
    initial = random.sample(overfitted, size)
    return initial

def create_initial_population(target: NT, grammar: ParameterizedTreeGrammar[NT, T], pop_size: int, tree_depth_delta=None, max_tree_depth=None) -> Sequence[Tree[NT, T]]:
    if tree_depth_delta is None:
        tree_depth_delta = 100
    min_size: int = grammar.minimum_tree_depth(target)
    if max_tree_depth is None:
        max_tree_depth = min_size + tree_depth_delta
    if max_tree_depth < min_size:
        raise ValueError(f"max_tree_depth {max_tree_depth} is less than minimum tree depth {min_size}")
    initial_population: Sequence[Tree[NT, T]] = []
    for _ in range(pop_size):
        nt = target
        cs = 0
        while True:
            rules, _ = grammar.annotations()
            applicable = []
            for (lhs, rhs), n in rules.items():
                if lhs == nt and cs + n <= max_tree_depth:
                    applicable.append(rhs)
            candidate = random.choice(applicable)
    # TODO: implement randomized top-down enumeration
    return []

# Tournament selection function using tournament selection
def tournament_selection(population: Sequence[Tree[NT, T]], fitness: Fitness[NT, T, V], tournament_size=3, select=None) -> Sequence[Tree[NT, T]]:
    if select is None:
        select = len(population)
    selected = []
    for _ in range(select):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=fitness.eval)
        selected.append(winner)
    return selected


def tournament_selection_curried(pop_fit: Sequence[tuple[Tree[NT, T], V]], tournament_size=3, select=None) -> Sequence[Tree[NT, T]]:
    if select is None:
        select = len(pop_fit)
    selected = []
    for _ in range(select):
        tournament = random.sample(pop_fit, tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# evolutionary algorithm for searching the fittest tree (local maximum of the fitness function) using tournament selection
def tournament_search(target: NT, grammar: ParameterizedTreeGrammar[NT, T], fitness: Fitness[NT, T, V],
                   population_size: int, generations: int, tournament_size=3, preserved_fittest=3) -> Sequence[Tree[NT, T]]:
    # Create the initial population
    population = enumerate_initial_population(target, grammar, population_size)

    # Run the genetic algorithm for a number of generations
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        # Select the best individuals for reproduction
        selected = tournament_selection(population, fitness, tournament_size=3, select=population_size)
        # Create the next generation
        next_generation = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1]
            # Perform crossover and mutation to create offspring
            offspring1 = parent1.crossover(parent2, grammar)
            offspring2 = offspring1.mutate(grammar)
            next_generation.append(offspring1)
            next_generation.append(offspring2)
        # Preserve the fittest individuals from the current generation
        next_generation.extend(sorted(population, key=lambda x: fitness.eval(x), reverse=True)[:preserved_fittest])
        # Replace the old population with the new one
        population = next_generation
    # Sort the final population by fitness
    population = sorted(population, key=fitness.eval, reverse=True)
    return population

