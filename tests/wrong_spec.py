from collections.abc import Callable
from bcls import Type, Constructor, Arrow, Intersection, Var, inhabit_and_interpret


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
            Y: Arrow(a, b),
            F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
        }
    )

    for real_result in inhabit_and_interpret(repository, Var(b)):
        print(real_result)


if __name__ == "__main__":
    test()
