from collections.abc import Callable
from cls import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    inhabit_and_interpret,
)
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import Literal, Param, TVar


def test() -> None:
    a: Type[str] = Constructor("a")
    b: Type[str] = Constructor("b")
    c: Type[str] = Constructor("c")
    d: Type[str] = Literal(3, int)

    X: str = "X"
    Y = lambda x, y: f"Y {x} {y}"
    F: Callable[[str], str] = lambda x: f"F({x})"

    repository: dict[str | Callable[[str], str], Param[str]] = dict(
        {Y: Param("x", int, lambda _: True, Arrow(TVar("x"), c))}
    )
    # for real_result in inhabit_and_interpret(repository, [Literal("3", int)]):
    #     print(real_result)
    result = FiniteCombinatoryLogic(repository, literals={int: [3]}).inhabit(c)
    print(result.show())
    for term in enumerate_terms(c, result):
        print(interpret_term(term))


if __name__ == "__main__":
    test()
