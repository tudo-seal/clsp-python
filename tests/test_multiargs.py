"""
If you want to have a variable arity of a combinator, you can type it using an intersection of all
the function types, that correspond to it's intended arity.

Note: While you can have multiple possible arities, each of them need to be fix. In this example
we use the type (A -> C) & (A -> B -> C) to denote a function, that can either be called with
one (A) or with two parameters (A and B).

To implement a combinator, fitting that type, you either have to use the *args construct (See
MultiArgsComponent) or default values (See DefaultArgComponent).

"""
import logging
from typing import Optional
import unittest
from clsp import (
    Constructor,
    Arrow,
    Intersection,
    inhabit_and_interpret,
)


def MultiArgsComponent(*arguments: str) -> str:
    return (
        f"MultiArgsComponent: I have {len(arguments)} argument(s), they are {arguments}"
    )


def DefaultArgComponent(arg1: str, arg2: Optional[str] = None) -> str:
    if arg2 is None:
        return f"DefaultArgComponent: I only got one argument, it is {arg1}"
    else:
        return f"DefaultArgComponent: I got two arguments, {arg1} and {arg2}"


class Test(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
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
        self.results = list(inhabit_and_interpret(repo, Constructor("C")))

        for result in self.results:
            self.logger.info(result)

    def test_length(self) -> None:
        self.assertEqual(4, len(self.results))

    def test_elements(self) -> None:
        self.assertIn(
            "DefaultArgComponent: I only got one argument, it is testarg1", self.results
        )
        self.assertIn(
            "MultiArgsComponent: I have 1 argument(s), they are ('testarg1',)",
            self.results,
        )
        self.assertIn(
            "MultiArgsComponent: I have 2 argument(s), they are ('testarg1', 'testarg2')",
            self.results,
        )
        self.assertIn(
            "DefaultArgComponent: I got two arguments, testarg1 and testarg2",
            self.results,
        )


if __name__ == "__main__":
    unittest.main()
