# Demonstration of the DSL for assigning values to lietral variables

#TODO more tests on different variants how values can be created, also infinite contains

import logging
import unittest
from clsp.dsl import DSL
from clsp.synthesizer import Contains, Synthesizer
from clsp.types import Type, Constructor, Var, Literal, Omega

class TestDSLUse(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_param(self) -> None:
        # literal varibles can be assigned computed values
        def X(x: bool, y: bool, z: bool) -> str:
            return f"X {x} {y} {z}"

        componentSpecifications = {
            X: DSL()
            .Parameter("x", "bool")
            .ParameterConstraint(lambda vars: vars["x"] == True) # x is True
            .Parameter("y", "bool", lambda vars: [False]) # y is False
            .Parameter("z", "bool", lambda vars: [vars["x"]]) # z is equal to x
            .Suffix(
                Constructor("a", Var("x"))
                & Constructor("b", Var("y"))
                & Constructor("c", Var("z"))
            )
        }

        def xyz(x: bool | None, y: bool | None, z: bool | None) -> Type:
            return (
                Constructor("a", Omega() if x is None else Literal(x, "bool"))
                & Constructor("b", Omega() if y is None else Literal(y, "bool"))
                & Constructor("c", Omega() if z is None else Literal(z, "bool"))
            )

        synthesizer = Synthesizer(componentSpecifications, parameterSpace={"bool": [True, False]})

        for x in [True, False, None]:
            for y in [True, False, None]:
                for z in [True, False, None]:
                    target = xyz(x, y, z)
                    grammar = synthesizer.constructSolutionSpace(target)
                    result = {tree.interpret() for tree in grammar.enumerate_trees(target)}
                    if x == False or y == True or z == False:
                        self.assertEqual(len(result), 0)
                    else:
                        self.assertEqual(len(result), 1)
                        self.assertTrue(result.issubset({"X True False True"}))

    def test_multi_values1(self) -> None:
        # a literal varible can be assigned multiple computed values
        def X(a: int, b: int) -> str:
            return f"X {a} {b}"

        parameterSpace = {"int": [0, 1, 2, 3]}
        componentSpecifications = {
            X: DSL()
            .Parameter("a", "int") # a in [0, 1, 2, 3]
            .Parameter("b", "int", lambda vars: [vars["a"] - 1, vars["a"] + 1]) # b in [a-1, a+1]
            .Suffix(Constructor("c", Var("a")))
        }

        synthesizer = Synthesizer(componentSpecifications, parameterSpace)
        target = Constructor("c", Literal(0, "int"))

        result = synthesizer.constructSolutionSpace(target)
        self.assertEqual(
            list(tree.interpret() for tree in result.enumerate_trees(target)), ["X 0 1"]
        )

    def test_multi_values2(self) -> None:
        # a literal varible can be assigned multiple computed values
        def X(a: int, b: int) -> str:
            return f"X {a} {b}"

        parameterSpace = {"int": [0, 1, 2, 3]}
        componentSpecifications = {
            X: DSL()
            .Parameter("a", "int")
            .Parameter("b", "int", lambda vars: [vars["a"] - 1, vars["a"] + 1])
            .Suffix(Constructor("c", Var("a")))
        }

        synthesizer = Synthesizer(componentSpecifications, parameterSpace)
        target = Constructor("c", Literal(1, "int"))

        result = synthesizer.constructSolutionSpace(target)
        self.assertSetEqual(
            set(tree.interpret() for tree in result.enumerate_trees(target)), {"X 1 2", "X 1 0"}
        )

    def test_infinite_values(self) -> None:
        # the number of values for a literal variable can be infinite
        class Nat(Contains):
            # represents the set of (arbitrary large) natural numbers
            def __contains__(self, value: object) -> bool:
                return isinstance(value, int) and value >= 0

        def X(x: int, y: int, b: str) -> str:
            return f"X {x} ({b})"

        parameterSpace = {"nat": Nat()}
        target = "c" @ Literal(3, "nat")

        componentSpecifications = {
            X: DSL()
            .Parameter("a", "nat") # a in [0, 1, 2, ...]
            .Parameter("b", "nat", lambda vars: [vars["a"] - 1]) # b in [a-1]
            .Suffix(("c" @ Var("b")) ** ("c" @ Var("a"))), # c(b) -> c(a)
            "ZERO": "c" @ Literal(0, "nat"), # c(0)
        }

        synthesizer = Synthesizer(componentSpecifications, parameterSpace)
        grammar = synthesizer.constructSolutionSpace(target)

        self.assertEqual(["X 3 (X 2 (X 1 (ZERO)))"], [tree.interpret() for tree in grammar.enumerate_trees(target)])


if __name__ == "__main__":
    unittest.main()
