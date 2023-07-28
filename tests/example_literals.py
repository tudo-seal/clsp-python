from cls import (
    Type,
    Constructor,
    Arrow,
)
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import Param, TVar


def test() -> None:
    # a: Type = Constructor("a")
    # b: Type = Constructor("b")
    c: Type = Constructor("c")
    # d: Type = Literal(3, int)

    # X: str = "X"
    Y = lambda x, y: f"Y {x} {y}"
    # F: Callable[[str], str] = lambda x: f"F({x})"

    repository = dict({Y: Param("x", int, lambda _: True, Arrow(TVar("x"), c))})
    # for real_result in inhabit_and_interpret(repository, [Literal("3", int)]):
    #     print(real_result)
    result = FiniteCombinatoryLogic(repository, literals={int: [3]}).inhabit(c)
    print(result.show())
    for term in enumerate_terms(c, result):
        print(interpret_term(term))


if __name__ == "__main__":
    test()
