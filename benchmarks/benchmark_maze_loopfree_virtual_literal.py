import timeit
from collections.abc import Callable, Iterable, Mapping, Container
from functools import partial
from itertools import chain, combinations, product
from typing import Any, cast
from clsp.dsl import DSL
from clsp.tree import Tree
from clsp.synthesizer import Synthesizer, Specification, ParameterSpace
from clsp.types import Constructor, Literal, Var, Type


def startc(visited: set[tuple[int, int]]) -> str:
    return "START"


def visited(path: Tree[Any]) -> set[tuple[int, int]]:
    if path.root == startc:
        return {(0, 0)}
    return {cast(tuple[int, int], path.parameters["a"].root)} | visited(path.parameters["pos"])


def powerset(s: list[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    return list(map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))))


def is_free(size: int, pos: tuple[int, int]) -> bool:
    """
    Create a maze in the form:
    XXX...XXX
    X       X
    X X...X X
    . ..... .
    X X...X X
    X       X
    X X...XXX
    X       X
    XXX...XXX
    """

    col, row = pos
    if row in [0, size - 1, size - 3]:
        return True
    else:
        if row == size - 2 and col == size - 1:
            return False
        if col in [0, size - 1]:
            return True
        return False


def main(SIZE: int = 2000, output: bool = False) -> float:
    U: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => UP({b})"
    D: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => DOWN({b})"
    L: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => LEFT({b})"
    R: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (Var(ab)))
    vis: Callable[[str], Type] = lambda ab: Constructor("vis", (Var(ab)))

    repo: Mapping[
        Callable[..., str] | str,
        Specification,
    ] = {
        U: DSL()
        .Parameter("old_visited", "power_int2")
        .Parameter("b", "int2")
        .ParameterConstraint(lambda vars: vars["b"] not in vars["old_visited"])
        .Parameter("a", "int2", lambda vars: [(vars["b"][0], vars["b"][1] + 1)])
        .Parameter("new_visited", "power_int2", lambda vars: [vars["old_visited"] | {vars["b"]}])
        .Argument("pos", pos("a") & vis("new_visited"))
        .Suffix(pos("b") & vis("old_visited")),
        D: DSL()
        .Parameter("old_visited", "power_int2")
        .Parameter("b", "int2")
        .ParameterConstraint(lambda vars: vars["b"] not in vars["old_visited"])
        .Parameter("a", "int2", lambda vars: [(vars["b"][0], vars["b"][1] - 1)])
        .Parameter("new_visited", "power_int2", lambda vars: [vars["old_visited"] | {vars["b"]}])
        .Argument("pos", pos("a") & vis("new_visited"))
        .Suffix(pos("b") & vis("old_visited")),
        L: DSL()
        .Parameter("old_visited", "power_int2")
        .Parameter("b", "int2")
        .ParameterConstraint(lambda vars: vars["b"] not in vars["old_visited"])
        .Parameter("a", "int2", lambda vars: [(vars["b"][0] + 1, vars["b"][1])])
        .Parameter("new_visited", "power_int2", lambda vars: [vars["old_visited"] | {vars["b"]}])
        .Argument("pos", pos("a") & vis("new_visited"))
        .Suffix(pos("b") & vis("old_visited")),
        R: DSL()
        .Parameter("old_visited", "power_int2")
        .Parameter("b", "int2")
        .ParameterConstraint(lambda vars: vars["b"] not in vars["old_visited"])
        .Parameter("a", "int2", lambda vars: [(vars["b"][0] - 1, vars["b"][1])])
        .Parameter("new_visited", "power_int2", lambda vars: [vars["old_visited"] | {vars["b"]}])
        .Argument("pos", pos("a") & vis("new_visited"))
        .Suffix(pos("b") & vis("old_visited")),
        startc: DSL()
        .Parameter("visited", "power_int2")
        .ParameterConstraint(lambda vars: (0, 0) not in vars["visited"])
        .Suffix("pos" @ (Literal((0, 0), "int2")) & vis("visited")),
    }

    # literals = {"int": list(range(SIZE))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(SIZE, (col, row)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")
    positions = list(filter(partial(is_free, SIZE), product(range(SIZE), range(SIZE))))

    class Powerset(Container):
        def __init__(self, s: Iterable[tuple[int, int]]):
            self.s = s

        def __contains__(self, item: object) -> bool:
            if not isinstance(item, frozenset):
                return False
            return item.issubset(self.s)

    power_positions = Powerset(positions)
    literals: ParameterSpace = {"int2": positions, "power_int2": power_positions}

    fin = ("pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))) & (
        "vis" @ Literal(frozenset(), "power_int2")
    )

    synthesizer: Synthesizer[Callable[[int, int, str], str] | str] = Synthesizer(
        repo, literals
    )

    start = timeit.default_timer()
    grammar = synthesizer.constructSolutionSpace(fin)

    if output:
        for tree in grammar.enumerate_trees(fin, 3):
            t = tree.interpret()
            if output:
                print(t)

    print(timeit.default_timer() - start)
    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
