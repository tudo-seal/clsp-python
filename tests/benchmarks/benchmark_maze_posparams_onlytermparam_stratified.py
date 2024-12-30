from collections.abc import Callable, Mapping
from functools import cache
import timeit
from typing import Optional
from clsp.dsl import DSL
from clsp.enumeration import Tree, enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, LVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


@cache
def current_position(pos: Tree[str]) -> tuple[int, int]:
    curr_pos = (0, 0)
    while pos.root != "START":
        if pos.root == "D":
            curr_pos = (curr_pos[0], curr_pos[1] + 1)
        elif pos.root == "U":
            curr_pos = (curr_pos[0], curr_pos[1] - 1)
        elif pos.root == "L":
            curr_pos = (curr_pos[0] - 1, curr_pos[1])
        elif pos.root == "R":
            curr_pos = (curr_pos[0] + 1, curr_pos[1])
        pos = pos.parameters["pos"]
    return curr_pos


@cache
def is_free(pos: tuple[int, int], SIZE: int, next_step: Optional[tuple[int, int]] = None) -> bool:
    col, row = pos
    if next_step is not None:
        col = col + next_step[0]
        row = row + next_step[1]
    if col < 0 or row < 0 or col >= SIZE or row >= SIZE:
        return False
    SEED = 0
    if row == col:
        return True
    else:
        return pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0


def main(SIZE: int = 9, output: bool = True) -> float:
    repo: Mapping[
        str,
        Param | Type,
    ] = {
        "U": DSL()
        .Use("new_size", "int")
        .Use("size", "int")
        .As(lambda new_size: new_size - 1)
        .Use("pos", Constructor("pos", LVar("size")))
        .With(lambda pos: is_free(current_position(pos), SIZE, next_step=(0, -1)))
        .In(Constructor("pos", LVar("new_size"))),
        "D": DSL()
        .Use("new_size", "int")
        .Use("size", "int")
        .As(lambda new_size: new_size - 1)
        .Use("pos", Constructor("pos", LVar("size")))
        .With(lambda pos: is_free(current_position(pos), SIZE, next_step=(0, 1)))
        .In(Constructor("pos", LVar("new_size"))),
        "L": DSL()
        .Use("new_size", "int")
        .Use("size", "int")
        .As(lambda new_size: new_size - 1)
        .Use("pos", Constructor("pos", LVar("size")))
        .With(lambda pos: is_free(current_position(pos), SIZE, next_step=(-1, 0)))
        .In(Constructor("pos", LVar("new_size"))),
        "R": DSL()
        .Use("new_size", "int")
        .Use("size", "int")
        .As(lambda new_size: new_size - 1)
        .Use("pos", Constructor("pos", LVar("size")))
        .With(lambda pos: is_free(current_position(pos), SIZE, next_step=(1, 0)))
        .In(Constructor("pos", LVar("new_size"))),
        "START": "pos" @ (Literal(0, "int")),
        "END": DSL()
        .Use("size", "int")
        .Use("pos", Constructor("pos", LVar("size")))
        .With(lambda pos: current_position(pos) == (SIZE - 1, SIZE - 1))
        .In(Constructor("end")),
    }

    literals = {"int": set(range(SIZE * SIZE))}
    # literals = {"int2": list(filter(is_free, product(range(SIZE), range(SIZE))))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free((col, row), SIZE):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = Constructor("pos", Literal(4, "int"))
    fin = Constructor("end")

    fcl: FiniteCombinatoryLogic[str] = FiniteCombinatoryLogic(repo, literals=literals)

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    interpretation = {
        "U": lambda new_size, size, pos: f"{pos} => UP",
        "D": lambda new_size, size, pos: f"{pos} => DOWN",
        "L": lambda new_size, size, pos: f"{pos} => LEFT",
        "R": lambda new_size, size, pos: f"{pos} => RIGHT",
        "END": lambda size, pos: f"{pos} => END",
    }

    for term in enumerate_terms(fin, grammar, 10):
        t = interpret_term(term, interpretation)
        if output:
            print(t)

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
