from collections.abc import Callable, Mapping
from picls import (
    Type,
    Constructor,
    Arrow,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)
from picls.enumeration import interpret_term


def test() -> None:
    a: Type = Constructor("a")
    b: Type = Constructor("b")
    c: Type = Constructor("c")
    # d: Type = Constructor("d")

    X: str = "X"
    Y: str = "Y"
    K: Callable[[str], Callable[[str], str]] = lambda x: (lambda y: x)

    def K2(x: str, y: str) -> str:
        return x

    MAP: Callable[[str, Callable[[str], str]], str] = lambda x, f: f(x)

    def MAP2(x: str, f: Callable[[str], str]) -> str:
        return f(x)

    repository: Mapping[
        str
        | Callable[[str, str], str]
        | Callable[[str], Callable[[str], str]]
        | Callable[[str, Callable[[str], str]], str],
        Type,
    ] = dict(
        {
            X: a,
            Y: b,
            K: Arrow(a, Arrow(b, c)),
            K2: Arrow(a, Arrow(b, c)),
            MAP: Arrow(b, Arrow(Arrow(b, c), c)),
            MAP2: Arrow(b, Arrow(Arrow(b, c), c)),
        }
    )
    environment: dict[str, set[str]] = dict()
    subtypes = Subtypes(environment)

    target = c
    # target = Var(Arrow(b, c))

    fcl = FiniteCombinatoryLogic(repository, subtypes)
    result = fcl.inhabit(target)

    enumerated_result = enumerate_terms(target, result)

    for real_result in enumerated_result:
        print(real_result)
        print(interpret_term(real_result))


if __name__ == "__main__":
    test()
