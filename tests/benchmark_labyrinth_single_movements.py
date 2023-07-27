from dataclasses import dataclass, field
import itertools
import timeit

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


SIZE = 3


def int_to_type(x: int) -> Type:
    return Constructor(str(x))


def free(row: int, col: int) -> Type:
    return Constructor("Free", Product(int_to_type(row), int_to_type(col)))


def pos(row: int, col: int) -> Type:
    return Constructor("Pos", Product(int_to_type(row), int_to_type(col)))


def seen(row: int, col: int) -> Type:
    return Constructor(f"Seen_({row}, {col})")


def single_move(
    row: int, col: int, drow_from: int, dcol_from: int, drow_to: int, dcol_to: int
) -> Type:
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
        ]
        + [
            Arrow(seen(row, col), Arrow(Omega(), seen(row, col)))
            for row in range(0, SIZE)
            for col in range(0, SIZE)
        ]
    )


@dataclass(frozen=True)
class Move(object):
    direction: str = field(init=True)

    def __call__(self, path: str, position: str) -> str:
        return f"{path} then go {self.direction}"


@dataclass(frozen=True)
class Start(object):
    def __call__(self) -> str:
        return "start"


def move(drow_from: int, dcol_from: int, drow_to: int, dcol_to: int) -> Type:
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


def test() -> None:
    for row in range(SIZE):
        for col in range(SIZE):
            if is_free(row, col):
                print("-", end="")
            else:
                print("#", end="")
        print("")

    single_movements = {
        c: t
        for row in range(0, SIZE)
        for col in range(0, SIZE)
        for (c, t) in (
            (Move(f"up_({row}, {col})"), single_move(row, col, 1, 0, 0, 0)),
            (Move(f"down_({row}, {col})"), single_move(row, col, 0, 0, 1, 0)),
            (Move(f"left_({row}, {col})"), single_move(row, col, 0, 1, 0, 0)),
            (Move(f"right_({row}, {col})"), single_move(row, col, 0, 0, 0, 1)),
        )
    }

    free_fields = {
        f"Pos_at_({row}, {col})": free(row, col)
        for row in range(0, SIZE)
        for col in range(0, SIZE)
        if is_free(row, col)
    }

    repository = (
        {
            Start(): Intersection(pos(0, 0), seen(0, 0)),
        }
        | single_movements
        | free_fields
    )

    start = timeit.default_timer()
    gamma = FiniteCombinatoryLogic(repository, Subtypes({}))
    print("Time (Constructor): ", timeit.default_timer() - start)
    start = timeit.default_timer()

    # target: BooleanTerm[Type] = Var(pos(SIZE - 1, SIZE - 1)) & ~(Var(seen(1, 1)))
    target = pos(SIZE - 1, SIZE - 1)

    results = gamma.inhabit(target)
    print("Time (Inhabitation): ", timeit.default_timer() - start)
    for t in itertools.islice(enumerate_terms(target, results), 2):
        print("Term:")
        print(t)
        print("Interpretation:")
        print(interpret_term(t))
        print("")


if __name__ == "__main__":
    test()
