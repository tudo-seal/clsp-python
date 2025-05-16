from collections.abc import Callable, Iterable, Mapping
import timeit
from itertools import product
from typing import Any
import torch
from gpytorch.likelihoods import GaussianLikelihood


from clsp.dsl import DSL
from clsp.enumeration import Tree, interpret_term

from clsp.types import Constructor, Literal, Param, LVar, Type

from clsp.search import SimpleEA, SimpleBO, tree_expected_improvement, GraphGP, RandomWalkKernel, tree_expected_improvement, Enumerate

import numpy as np

from grakel import Graph
from grakel.kernels import (
    RandomWalk,
)
from grakel.graph import is_edge_dictionary

def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def is_free(pos: tuple[int, int]) -> bool:
    col, row = pos
    SEED = 0
    if row == col:
        return True
    else:
        return pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0


def getpath(
    path: Tree[Any, Any]
) -> Iterable[tuple[int, int]]:
    position_arg = path.parameters["b"].root
    while path.root != "START":
        if isinstance(position_arg, tuple):
            yield position_arg
        position_arg = path.parameters["a"].root
        path = path.parameters["pos"]


def main(solutions: int = 10000, output: bool = True) -> float:
    SIZE = 5
    U: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => UP({b})"
    D: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => DOWN({b})"
    L: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => LEFT({b})"
    R: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (LVar(ab)))

    repo: Mapping[
        Callable[[int, int, str], str] | str,
        Param | Type,
    ] = {
        U: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0], a[1] - 1))
        .Use("pos", pos("a"))
        .In(pos("b")),
        D: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0], a[1] + 1))
        .Use("pos", pos("a"))
        .In(pos("b")),
        L: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0] - 1, a[1]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        R: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0] + 1, a[1]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    # literals = {"int": list(range(SIZE))}
    literals = {"int2": list(product(range(SIZE), range(SIZE)))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free((row, col)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))


    start = timeit.default_timer()

    def shortest_loop_free_path(tree: Tree[Any, Any]) -> int:
        path = list(getpath(tree))
        length = len(path)
        if length != len(set(path)):
            return -100000000
        else:
            return length*(-1)

    def longest_loop_free_path(tree: Tree[Any, Any]) -> int:
        path = list(getpath(tree))
        length = len(path)
        if length != len(set(path)):
            return -1
        else:
            return length

    #tournament_search = SimpleEA(repo, literals, fin, generations=4).search_fittest

    test_trees: list[Tree[Type, Any]] = list(Enumerate(repo, literals, fin).sample(100))
    evaluations: list[int] = [longest_loop_free_path(t) for t in test_trees]

    X: list[Graph] = [Graph(e, l) for t in test_trees for e, l, _ in (t.to_labeled_adjacency_dict(),)]
    Y: list[int] = evaluations

    class MyGraph(Graph):
        def __repr__(self) -> str:
            output = ["#vertices"]
            output += [','.join(map(str, self.get_vertices(self._format)))]

            output += ["#edges"]
            output += ['\n'.join(map(lambda x: str(x[0]) + ',' + str(x[1]), self.get_edges(self._format)))]

            def list_repr(x):
                # convert numpy to list
                if type(x) in [np.array, np.ndarray]:
                    x = x.tolist()
                elif isinstance(x, Iterable):
                    x = list(x)
                else:
                    return str(x)
                return '[' + ','.join(map(str, x)) + ']'

            if bool(self.node_labels):
                output += ["#node_labels"]
                output += ['\n'.join(map(lambda x: str(x[0]) + '->' + list_repr(x[1]), self.node_labels.items()))]

            if bool(self.edge_labels):
                output += ["#edge_labels"]
                output += ['\n'.join(map(lambda x: str(x[0][0]) + ',' + str(x[0][1]) + '->' + list_repr(x[1]),
                                         self.edge_labels.items()))]

            return '\n'.join(output)

    for g in test_trees:
        edges, labels, _ = g.to_labeled_adjacency_dict()
        print(edges)
        print(labels)
        print(is_edge_dictionary(edges))
        #print(Graph(d, graph_format="dictionary"))  # der Fehler, der hier kommt ist ein Grakel-Bug und kein Fehler meinerseits!!!!! -.-
        print(MyGraph(edges, node_labels=labels, graph_format="dictionary"))  # der Fehler, der hier kommt ist ein Grakel-Bug und kein Fehler meinerseits!!!!! -.-

    #print(X)
    test = RandomWalkKernel()
    print(test.kernel(X))


    # TODO model = GraphGP()

    model = GraphGP(
        train_x=list(),
        train_y=torch.empty((1,)),
        likelihood=GaussianLikelihood(),
        kernel=RandomWalkKernel()
    )
    bo_search = SimpleBO(model, tree_expected_improvement, SimpleEA(repo, literals, fin, generations=4), repo, literals, fin).search_max


    #final_population = list(tournament_search(longest_loop_free_path, 100))
    final_population = list(bo_search(longest_loop_free_path))

    #for term in final_population[:10]:
    #    positions = list(getpath(term))
    #    t = interpret_term(term)
    #    print(t)
    #    print(f"Path length: {len(positions)}")
    #    print("#######################################")

    print("maximum:")
    winner = final_population[0]
    print(interpret_term(winner))
    win_path = list(getpath(winner))
    print(f"Path: {win_path}")
    print(f"Path set length: {len(set(win_path))}")
    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
