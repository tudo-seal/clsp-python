import logging
import unittest
from clsp.dsl import DSL, Requires
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import (
    Arrow,
    Intersection,
    Param,
    SetTo,
    Constructor,
    Product,
    Literal,
    LVar,
    Omega,
)
from clsp.subtypes import Subtypes

class TestDSL(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.a = Constructor("a")
        self.b = Constructor("b")
        self.c = Constructor("c")

    def test1(self) -> None:
        ty = Intersection(LVar("a"), Constructor("c", LVar("b")))

        path = Literal(4, "int")

        sub = Subtypes({})

        groups = {"a" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test1: {sub.infer_substitution(ty, path, groups)}")

    def test2(self) -> None:
        ty = Intersection(LVar("a"), Literal(3, "int"))

        path = Literal(3, "int")

        sub = Subtypes({})

        groups = {"a" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test2: {sub.infer_substitution(ty, path, groups)}")

    def test3(self) -> None:
        ty = Intersection(LVar("a"), LVar("a'"))

        path = Literal(3, "int")

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test3: {sub.infer_substitution(ty, path, groups)}")

    def test4(self) -> None:
        ty = Intersection(LVar("a"), LVar("a'"))

        path = Literal("s", "str")

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test4: {sub.infer_substitution(ty, path, groups)}")

    def test5(self) -> None:
        ty = Arrow(LVar("a"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test5: {sub.infer_substitution(ty, path, groups)}")


    def test6(self) -> None:
        ty = Arrow(LVar("b"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test6: {sub.infer_substitution(ty, path, groups)}")

    def test7(self) -> None:
        ty = Arrow(Omega(), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test7: {sub.infer_substitution(ty, path, groups)}")

    def test8(self) -> None:
        ty = Arrow(Literal(4, "int"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a" : "int", "a'" : "int", "b" : "str"}
        print(f"{ty} <= {path}")
        print(f"test8: {sub.infer_substitution(ty, path, groups)}")

    def test_for_real(self) -> None:
        ty = Intersection(Constructor("In", LVar("a")), Constructor("Out", LVar("b")))

        path1 = Constructor("In", Literal(3, "int"))
        path2 = Constructor("Out", Literal(4, "int"))

        sub = Subtypes({})

        groups = {"a" : "int", "b" : "int"}
        print(f"{ty} <= {path1}")
        print(f"{ty} <= {path2}")
        print(f"test_for_real: {sub.infer_substitution(ty, path1, groups)} and {sub.infer_substitution(ty, path2, groups)}")

if __name__ == "__main__":
    unittest.main()
