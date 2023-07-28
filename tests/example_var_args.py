from collections.abc import Sequence
from picls import Type, Constructor, Arrow, inhabit_and_interpret


def F(*x: str) -> Sequence[str]:
    return x


def test() -> None:
    a: Type = Constructor("a")
    b: Type = Constructor("b")
    c: Type = Constructor("c")
    # d: Type = Constructor("d")

    X: str = "X"
    Y: str = "Y"

    repository = dict({X: a, Y: b, F: Arrow(a, (Arrow(b, (Arrow(b, c)))))})

    for real_result in inhabit_and_interpret(repository, c):
        print(real_result)


if __name__ == "__main__":
    test()
