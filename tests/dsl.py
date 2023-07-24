from cls.dsl import Use
from cls.enumeration import enumerate_terms, enumerate_terms_of_size, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import Omega, Param, SetTo, TVar, Literal


def X2(x, y, z, n):
    return f"X2: x:{x} y:{y} z:{z} ({n})"


def X(x, y, z, n):
    return f"X: x:{x} y:{y} z:{z} ({n})"


def Y(x, y, z):
    return f"Y: x:{x} y:{y} z:{z}"


def p(vars):
    res = vars["x"].value + vars["y"].value == 5
    print(vars)
    print(res)
    return res


def po(vars):
    return vars["y"].value + 1 or vars["x"].value + 1


def main() -> None:
    repo = {
        X2: Use("x", int)
        .Use("y", int)
        .With(p)
        .Use("z", int)
        .As(po)
        .In(
            ("c" @ ((TVar("y") * TVar("x") * TVar("z"))))
            ** ("c" @ (TVar("x") * TVar("y") * TVar("z")))
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
                    ("c" @ ((TVar("y") * TVar("x") * TVar("z"))))
                    ** ("c" @ (TVar("x") * TVar("y") * TVar("z"))),
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
                    "z", int, lambda _: True, "c" @ (TVar("x") * TVar("y") * TVar("z"))
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
