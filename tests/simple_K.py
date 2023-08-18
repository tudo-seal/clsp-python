from collections.abc import Callable, Mapping
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic
from picls.types import Arrow, Constructor, Type

X: str = "X"
Y: str = "Y"
K: Callable[[str], Callable[[str], str]] = lambda x: (lambda y: x)


def K2(x: str, y: str) -> str:
    return x


MAP: Callable[[str, Callable[[str], str]], str] = lambda x, f: f(x)


def MAP2(x: str, f: Callable[[str], str]) -> str:
    return f(x)


def main():
    a: Type = Constructor("a")
    b: Type = Constructor("b")
    c: Type = Constructor("c")
    # d: Type = Constructor("d")

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

    target = c
    # target = Var(Arrow(b, c))

    fcl = FiniteCombinatoryLogic(repository, {})
    result = fcl.multiinhabit(target)

    enumerated_result = list(enumerate_terms(target, result))
    interpreted_terms = []
    for real_result in enumerated_result:
        print(real_result)
        interpreted_term = interpret_term(real_result)
        interpreted_terms.append(interpreted_term)


if __name__ == "__main__":
    main()
