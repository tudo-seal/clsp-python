from typing import Any, TypeVar
from picls import inhabit_and_interpret
from picls.types import Arrow, Constructor, Type

T = TypeVar("T")


def nil() -> list[Any]:
    return list()


def cons(x: T, xs: list[T]) -> list[T]:
    the_list = [x]
    the_list.extend(xs)
    return the_list


def ComponentUsingLists(list_arg: list[Any]) -> str:
    return f"{list_arg}"


A = Constructor("A")
B = Constructor("B")
C = Constructor("C")


def List(a: Type) -> Constructor:
    return Constructor("List", a)


def a_to_b(a: list[Any]) -> int:
    return len(a)


exampleRepo = {
    nil: List(A),
    cons: Arrow(A, Arrow(List(A), List(A))),
    a_to_b: Arrow(A, B),
    lambda _: "b": Arrow(A, B),
    lambda _: 2: Arrow(A, B),
    lambda f, xs: list(map(f, xs)): Arrow(Arrow(A, B), Arrow(List(A), List(B))),
    "a": A,
    ComponentUsingLists: Arrow(List(B), C),
}


def main() -> None:
    r = inhabit_and_interpret(exampleRepo, C)
    for z in r:
        print(z)


if __name__ == "__main__":
    main()
