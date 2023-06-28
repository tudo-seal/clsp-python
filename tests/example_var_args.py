from cls import Type, Constructor, Arrow, inhabit_and_interpret


def F(*x):
    return x


def test() -> None:
    a: Type[str] = Constructor("a")
    b: Type[str] = Constructor("b")
    c: Type[str] = Constructor("c")
    d: Type[str] = Constructor("d")

    X: str = "X"
    Y: str = "Y"

    repository = dict({X: a, Y: b, F: Arrow(a, (Arrow(b, (Arrow(b, c)))))})

    for real_result in inhabit_and_interpret(repository, c):
        print(real_result)


if __name__ == "__main__":
    test()
