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
    F: Callable[[str], str] = lambda x: f"F({x})"

    repository: dict[str | Callable[[str], str], Type[str]] = dict(
        {
            X: Intersection(Intersection(a, b), d),
            Y: d,
            F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
        }
    )
    environment: dict[str, set[str]] = dict()
    subtypes = Subtypes(environment)

    target = Var(a) & ~(Var(b) & Var(c))
    # target = Var(b)

    fcl = FiniteCombinatoryLogic(repository, subtypes)
    result = fcl.inhabit(target)

    enumerated_result = enumerate_terms(target, result)

    for real_result in enumerated_result:
        print(interpret_term(real_result))


if __name__ == "__main__":
    test()
