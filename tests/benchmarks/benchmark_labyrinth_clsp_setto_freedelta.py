from collections.abc import Callable, Mapping
from abc import ABC, abstractmethod
import timeit
from typing import Any, Generic, TypeVar
from itertools import product

from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, LVar, Type


Carrier = TypeVar("Carrier")


class MazeSig(Generic[Carrier], ABC):
    @abstractmethod
    def up(self, a: tuple[int, int], b: tuple[int, int], p: Carrier) -> Carrier: ...
    @abstractmethod
    def down(self, a: tuple[int, int], b: tuple[int, int], p: Carrier) -> Carrier: ...
    @abstractmethod
    def left(self, a: tuple[int, int], b: tuple[int, int], p: Carrier) -> Carrier: ...
    @abstractmethod
    def right(self, a: tuple[int, int], b: tuple[int, int], p: Carrier) -> Carrier: ...
    @abstractmethod
    def start(self) -> Carrier: ...

    def as_dict(self) -> dict[str, Any]:
        return {"U": self.up, "D": self.down, "L": self.left, "R": self.right, "START": self.start}


class MazeString(MazeSig[str]):
    def up(self, a: tuple[int, int], b: tuple[int, int], p: str) -> str:
        return f"{p} => UP({b})"

    def down(self, a: tuple[int, int], b: tuple[int, int], p: str) -> str:
        return f"{p} => DOWN({b})"

    def left(self, a: tuple[int, int], b: tuple[int, int], p: str) -> str:
        return f"{p} => LEFT({b})"

    def right(self, a: tuple[int, int], b: tuple[int, int], p: str) -> str:
        return f"{p} => RIGHT({b})"

    def start(self) -> str:
        return "START"


class MazePoints(MazeSig[list[tuple[int, int]]]):
    def up(
        self, a: tuple[int, int], b: tuple[int, int], p: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return p + [b]

    def down(
        self, a: tuple[int, int], b: tuple[int, int], p: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return p + [b]

    def left(
        self, a: tuple[int, int], b: tuple[int, int], p: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return p + [b]

    def right(
        self, a: tuple[int, int], b: tuple[int, int], p: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return p + [b]

    def start(self) -> list[tuple[int, int]]:
        return [(0, 0)]


def main(SIZE: int = 10, output: bool = True) -> float:
    def is_free(col: int, row: int) -> bool:
        SEED = 0
        if row == col:
            return True
        else:
            return pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0

    pos: Callable[[str], Type] = lambda p: Constructor("pos", LVar(p))

    repo: Mapping[
        str,
        Param | Type,
    ] = {
        "U": DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .As(lambda a: (a[0], a[1] - 1))
        .In(pos("a") ** pos("b")),
        "D": DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .As(lambda a: (a[0], a[1] + 1))
        .In(pos("a") ** pos("b")),
        "L": DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .As(lambda a: (a[0] - 1, a[1]))
        .In(pos("a") ** pos("b")),
        "R": DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .As(lambda a: (a[0] + 1, a[1]))
        .In(pos("a") ** pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    literals = {
        "int2": [pos for pos in product(range(SIZE), range(SIZE)) if is_free(pos[0], pos[1])]
    }

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(col, row):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))

    fcl: FiniteCombinatoryLogic[str] = FiniteCombinatoryLogic(repo, literals=literals)

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        t = interpret_term(term, MazeString().as_dict())
        p = interpret_term(term, MazePoints().as_dict())
        if output:
            print(t)
            for row in range(SIZE):
                for col in range(SIZE):
                    if (col, row) in p:
                        print("X", end="")
                    elif is_free(col, row):
                        print("-", end="")
                    else:
                        print("#", end="")
                print("")

    return timeit.default_timer() - start


if __name__ == "__main__":
    main(20)
