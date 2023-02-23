from dataclasses import dataclass, field
from typing import Callable

from cls_python import *
from cls_python.boolean import BooleanTerm, Var
from cls_python.pfcl import FiniteCombinatoryLogic
from cls_python.enumeration_new import enumerate_terms, interpret_term

labyrinth_free = (
    (True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, False, True, True, True, False, True, True, True, False, True, True, True, False, True, True),
    (True, True, True, True, True, True, True, True, True, False, True, False, False, True, True, False, False, True, True, True, True, True, True, True, True, True, False, False, False, True),
    (True, True, True, True, True, True, True, False, True, False, True, True, True, True, False, True, True, True, True, True, False, False, True, True, False, True, True, False, True, False),
    (True, True, True, True, True, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False),
    (True, True, True, True, True, False, True, True, False, False, True, False, True, False, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, False),
    (True, True, False, False, True, True, True, True, False, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True),
    (True, True, True, False, True, False, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True),
    (True, True, True, False, True, False, False, True, True, False, False, False, True, True, True, True, False, True, True, True, False, True, True, True, True, False, True, True, False, True),
    (True, True, True, False, True, True, False, True, True, False, True, False, True, True, False, False, False, True, False, True, True, True, True, True, False, False, True, False, True, True),
    (True, False, False, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, False, False, True, False, True, True),
    (True, False, True, False, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, False, True, False, True, False, True, True, False, False, True, False),
    (True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, True, False, False, True, False, True, True, True, True, True, True, True, True),
    (True, False, True, True, False, False, True, False, False, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True),
    (True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True),
    (False, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, False, True, True, True, False, True, True, True, True, True, True),
    (True, False, False, True, True, True, False, False, True, True, False, False, True, True, False, True, True, False, False, True, True, True, True, True, True, True, False, True, True, True),
    (True, True, False, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True),
    (True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False, False, False, True, False, True, True, True, True, True, True, True, True, True, True),
    (True, True, True, True, True, False, True, True, True, False, True, False, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, False, True),
    (False, False, True, True, False, True, False, True, True, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True),
    (True, True, True, False, False, False, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, False),
    (False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, False, True),
    (True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True),
    (True, True, True, False, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, False, True, False, True, True),
    (True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, False, True, True, True),
    (True, True, True, False, True, True, False, True, True, True, True, False, True, True, True, True, True, True, False, False, True, True, True, True, False, True, True, True, False, True),
    (False, False, True, True, False, True, True, True, True, False, True, True, True, False, True, True, False, False, True, True, True, True, True, True, True, True, False, False, False, True),
    (True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True),
    (True, True, False, True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, True, False, False, True, False, True, False, True, True, False, True, True),
    (True, True, True, True, True, False, True, True, False, False, False, True, True, False, False, True, False, False, True, True, True, False, True, True, True, True, True, True, True, True)
)

size = 3  # len(labyrinth_free)

def int_to_type(x: int) -> Type:
    return Constructor(str(x))

def free(row: int, col: int) -> Type:
    return Constructor("Free", Product(int_to_type(row), int_to_type(col)))

def pos(row: int, col: int) -> Type:
    return Constructor("Pos", Product(int_to_type(row), int_to_type(col)))

def seen(row: int, col: int) -> Type:
    return Constructor(f"Seen_({row}, {col})")


free_fields = {
    f"Pos_at_({row}, {col})": free(row, col)
    for row in range(0, size) for col in range(0, size) if labyrinth_free[col][row]
}

def move(drow_from: int, dcol_from: int, drow_to: int, dcol_to: int) -> Type:
    return Type.intersect([
        Arrow(pos(row + drow_from, col + dcol_from),
              Arrow(
                  free(row + drow_to, col + dcol_to),
                  Intersection(pos(row + drow_to, col + dcol_to), seen(row + drow_to, col + dcol_to)))
              )
        for row in range(0, size) for col in range(0, size)
    ] + [
        Arrow(seen(row, col),
              Arrow(Omega(), seen(row, col))
              )
        for row in range(0, size) for col in range(0, size)
    ])


@dataclass(frozen=True)
class Move(object):
    direction: field(init=True)

    def __call__(self, path: str, position: str) -> str:
        return f"{path} then go {self.direction}"

@dataclass(frozen=True)
class Start(object):
    def __call__(self) -> str:
        return "start"

repository = {
    Start(): Intersection(pos(0, 0), seen(0, 0)),
    Move("up"): move(0, 1, 0, 0),
    Move("down"): move(0, 0, 0, 1),
    Move("left"): move(1, 0, 0, 0),
    Move("right"): move(0, 0, 1, 0),
    **free_fields
}

import timeit

if __name__ == "__main__":
    start = timeit.default_timer()
    gamma = FiniteCombinatoryLogic(repository, Subtypes({}))
    print('Time (Constructor): ', timeit.default_timer() - start) 
    start = timeit.default_timer()

    target: BooleanTerm[Type] = Var(pos(size - 1, size - 1)) & ~(Var(seen(1, 1)))

    results = gamma.inhabit(target)
    print('Time (Inhabitation): ', timeit.default_timer() - start) 
    for t in enumerate_terms(target, results):
        print("Term:")
        print(t)
        print("Interpretation:")
        print(interpret_term(t))
        print("")
