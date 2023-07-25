from collections.abc import Mapping
import itertools
import timeit
from dataclasses import dataclass, field
from typing import Any

from cls import (
    Type,
    Constructor,
    Product,
    Omega,
    Arrow,
    Intersection,
    FiniteCombinatoryLogic,
    enumerate_terms,
    interpret_term,
    Subtypes,
)


# pseudo-random labyrinth
def is_free(row: int, col: int) -> bool:
    SEED = 0
    if row == col:
        return True
    else:
        return (
            pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0
        )


def int_to_type(x: int) -> Type[str]:
    return Constructor(str(x))


def free(row: int, col: int) -> Type[str]:
    return Constructor("Free", Product(int_to_type(row), int_to_type(col)))


def pos(row: int, col: int) -> Type[str]:
    return Constructor("Pos", Product(int_to_type(row), int_to_type(col)))


def seen(row: int, col: int) -> Type[str]:
    return Constructor(f"Seen_({row}, {col})")


@dataclass(frozen=True)
class Move:
    direction: str = field(init=True)

    def __call__(self, path: str, position: str) -> str:
        return f"{path} then go {self.direction}"


@dataclass(frozen=True)
class Start:
    def __call__(self) -> str:
        return "start"


def move(
    SIZE: int, drow_from: int, dcol_from: int, drow_to: int, dcol_to: int
) -> Type[str]:
    return Type.intersect(
        [
            Arrow(
                pos(row + drow_from, col + dcol_from),
                Arrow(
                    free(row + drow_to, col + dcol_to),
                    Intersection(
                        pos(row + drow_to, col + dcol_to),
                        seen(row + drow_to, col + dcol_to),
                    ),
                ),
            )
            for row in range(0, SIZE)
            for col in range(0, SIZE)
        ]
        + [
            Arrow(seen(row, col), Arrow(Omega(), seen(row, col)))
            for row in range(0, SIZE)
            for col in range(0, SIZE)
        ]
    )


def labyrinth(SIZE: int = 10, output: bool = True) -> float:
    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(row, col):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    free_fields: Mapping[str, Type[str]] = {
        f"Pos_at_({row}, {col})": free(row, col)
        for row in range(0, SIZE)
        for col in range(0, SIZE)
        if is_free(row, col)
    }

    repository: Mapping[Any, Type[str]] = {
        Start(): Intersection(pos(0, 0), seen(0, 0)),
        Move("up"): move(SIZE, 1, 0, 0, 0),
        Move("down"): move(SIZE, 0, 0, 1, 0),
        Move("left"): move(SIZE, 0, 1, 0, 0),
        Move("right"): move(SIZE, 0, 0, 0, 1),
    } | free_fields

    start = timeit.default_timer()
    gamma = FiniteCombinatoryLogic(repository, Subtypes({}))
    if output:
        print("Time (Constructor): ", timeit.default_timer() - start)
    start = timeit.default_timer()

    # target: BooleanTerm[Type] = Var(pos(SIZE - 1, SIZE - 1)) & ~(Var(seen(1, 1)))
    target = pos(SIZE - 1, SIZE - 1)
    # target: BooleanTerm[Type] = Var(seen(1, 1))

    results = gamma.inhabit(target)
    if output:
        print("Time (Inhabitation): ", timeit.default_timer() - start)
    for t in itertools.islice(enumerate_terms(target, results), 3):
        if output:
            print("Term:")
            print(t)
            print("Interpretation:")
        term = interpret_term(t)
        if output:
            print(term)
            print("")
    return timeit.default_timer() - start


if __name__ == "__main__":
    labyrinth()
