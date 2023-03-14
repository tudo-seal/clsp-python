from collections.abc import Callable
from bcls import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    Var,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)
from bcls.enumeration import interpret_term


def test() -> None:
    a: Type[str] = Constructor("a")
    b: Type[str] = Constructor("b")
    c: Type[str] = Constructor("c")
    d: Type[str] = Constructor("d")

    X: str = "X"
    Y: str = "Y"
    K: Callable[[str], Callable[[str], str]] = lambda x: (lambda y: x)
    MAP: Callable[[str, Callable[[str], str]], str] = lambda x, f: f(x)

    repository: dict[str | Callable[[str], str], Type[str]] = dict(
        {
            X: a,
            Y: b,
            K: Arrow(a, Arrow(b, c)),
            MAP: Arrow(b, Arrow(Arrow(b, c), c)),
        }
    )
    environment: dict[str, set[str]] = dict()
    subtypes = Subtypes(environment)

    target = Var(c)
    # target = Var(Arrow(b, c))

    fcl = FiniteCombinatoryLogic(repository, subtypes)
    result = fcl.inhabit(target)

    enumerated_result = enumerate_terms(target, result)

    for real_result in enumerated_result:
        print(real_result)
        print(interpret_term(real_result))


if __name__ == "__main__":
    test()
