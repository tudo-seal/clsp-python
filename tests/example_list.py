from bcls import inhabit_and_interpret
from bcls.types import Arrow, Constructor


def nil():
    return list()


def cons(x, xs):
    the_list = [x]
    the_list.extend(xs)
    return the_list


def ComponentUsingLists(list_arg: list):
    return f"{list_arg}"


A = Constructor("A")
B = Constructor("B")
C = Constructor("C")


def List(a):
    return Constructor("List", a)


def a_to_b(a):
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


def main():
    r = inhabit_and_interpret(exampleRepo, C)
    for z in r:
        print(z)


if __name__ == "__main__":
    main()
