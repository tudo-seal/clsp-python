from collections.abc import Mapping
from cls.dsl import Use
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import Param, SetTo, TVar, Literal


def X2(x: int, y: int, z: int, n: str) -> str:
    return f"X2: x:{x} y:{y} z:{z} ({n})"


def X(x: int, y: int, z: int, n: str) -> str:
    return f"X: x:{x} y:{y} z:{z} ({n})"


def Y(x: int, y: int, z: int) -> str:
    return f"Y: x:{x} y:{y} z:{z}"


def p(vars: Mapping[str, Literal]) -> bool:
    res = vars["x"].value + vars["y"].value == 5
    print(vars)
    print(res)
    return bool(res)


def po(vars: Mapping[str, Literal]) -> int:
    return int(vars["y"].value + 1 * vars["x"].value + 1)


def main() -> None:
    repo = {
        X2: Use[str]("x", int)
        .Use("y", int)
        .With(p)
        .Use("z", int)
        .As(po)
        .In(
            ("c" @ ((TVar[str]("y") * TVar("x") * TVar("z"))))
            ** ("c" @ (TVar[str]("x") * TVar("y") * TVar("z")))
        ),
        X: Param(
            "x",
            int,
            lambda _: True,
            Param(
                "y",
                int,
                p,
                Param(
                    "z",
                    int,
                    SetTo(po),
                    ("c" @ ((TVar[str]("y") * TVar("x") * TVar("z"))))
                    ** ("c" @ (TVar[str]("x") * TVar("y") * TVar("z"))),
                ),
            ),
        ),
        Y: Param(
            "x",
            int,
            lambda _: True,
            Param(
                "y",
                int,
                p,
                Param(
                    "z",
                    int,
                    lambda _: True,
                    "c" @ (TVar[str]("x") * TVar("y") * TVar("z")),
                ),
            ),
        ),
    }
    # print(repo[X])

    tau = "c" @ (Literal(2, int) * Literal(3, int) * Literal(4, int))

    grammar = FiniteCombinatoryLogic(repo, literals={int: [1, 2, 3, 4]}).inhabit(tau)
    # print(grammar.show())
    for term in enumerate_terms(tau, grammar):
        print(interpret_term(term))


if __name__ == "__main__":
    main()
