"""
If you want to have a variable arity of a combinator, you can type it using an intersection of all
the function types, that correspond to it's intended arity.

Note: While you can have multiple possible arities, each of them need to be fix. In this example
we use the type (A -> C) & (A -> B -> C) to denote a function, that can either be called with
one (A) or with two parameters (A and B).

To implement a combinator, fitting that type, you either have to use the *args construct (See
MultiArgsComponent) or default values (See DefaultArgComponent).

"""
from bcls import (
    Constructor,
    Arrow,
    Intersection,
    inhabit_and_interpret,
)


def MultiArgsComponent(*arguments):
    return (
        f"MultiArgsComponent: I have {len(arguments)} argument(s), they are {arguments}"
    )


def DefaultArgComponent(arg1, arg2=None):
    if arg2 is None:
        return f"DefaultArgComponent: I only got one argument, it is {arg1}"
    else:
        return f"DefaultArgComponent: I got two arguments, {arg1} and {arg2}"


def main():
    repo = {
        MultiArgsComponent: Intersection(
            Arrow(Constructor("A"), Constructor("C")),
            Arrow(Constructor("A"), Arrow(Constructor("B"), Constructor("C"))),
        ),
        DefaultArgComponent: Intersection(
            Arrow(Constructor("A"), Constructor("C")),
            Arrow(Constructor("A"), Arrow(Constructor("B"), Constructor("C"))),
        ),
        "testarg1": Constructor("A"),
        "testarg2": Constructor("B"),
    }
    results = inhabit_and_interpret(repo, Constructor("C"))
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
