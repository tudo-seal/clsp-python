from collections.abc import Callable
from cls import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)
from cls.enumeration import interpret_term


def test() -> None:
    a: Type = Constructor("a")
    b: Type = Constructor("b")
    c: Type = Constructor("c")
    d: Type = Constructor("d")

    X: str = "X"
    Y: str = "Y"
    F: Callable[[str], str] = lambda x: f"F({x})"

    repository: dict[str | Callable[[str], str], Type] = dict(
        {
            X: Intersection(Intersection(a, b), d),
            Y: d,
            F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
        }
    )
    environment: dict[str, set[str]] = dict()
    subtypes = Subtypes(environment)

    target = Intersection(c, b)
    # target = Var(b)

    fcl = FiniteCombinatoryLogic(repository, subtypes)
    result = fcl.inhabit(target)

    enumerated_result = enumerate_terms(target, result)

    for real_result in enumerated_result:
        print(interpret_term(real_result))


if __name__ == "__main__":
    test()
