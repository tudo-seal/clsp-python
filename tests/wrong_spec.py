from collections.abc import Callable
from picls import Type, Constructor, Arrow, Intersection, inhabit_and_interpret


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
            Y: Arrow(a, b),
            F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
        }
    )

    for real_result in inhabit_and_interpret(repository, b):
        print(real_result)


if __name__ == "__main__":
    test()
