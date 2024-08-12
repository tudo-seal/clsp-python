import logging
import unittest
from clsp.types import (
    Arrow,
    Intersection,
    Constructor,
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

        groups = {"a": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test1: {sub.infer_substitution(ty, path, groups)}")

    def test2(self) -> None:
        ty = Intersection(LVar("a"), Literal(3, "int"))

        path = Literal(3, "int")

        sub = Subtypes({})

        groups = {"a": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test2: {sub.infer_substitution(ty, path, groups)}")

    def test3(self) -> None:
        ty = Intersection(LVar("a"), LVar("a'"))

        path = Literal(3, "int")

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test3: {sub.infer_substitution(ty, path, groups)}")

    def test4(self) -> None:
        ty = Intersection(LVar("a"), LVar("a'"))

        path = Literal("s", "str")

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test4: {sub.infer_substitution(ty, path, groups)}")

    def test5(self) -> None:
        ty = Arrow(LVar("a"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test5: {sub.infer_substitution(ty, path, groups)}")

    def test6(self) -> None:
        ty = Arrow(LVar("b"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test6: {sub.infer_substitution(ty, path, groups)}")

    def test7(self) -> None:
        ty = Arrow(Omega(), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test7: {sub.infer_substitution(ty, path, groups)}")

    def test8(self) -> None:
        ty = Arrow(Literal(4, "int"), LVar("a"))

        path = Arrow(Literal(3, "int"), Literal(3, "int"))

        sub = Subtypes({})

        groups = {"a": "int", "a'": "int", "b": "str"}
        self.logger.info(f"{ty} <= {path}")
        self.logger.info(f"test8: {sub.infer_substitution(ty, path, groups)}")

    def test_for_real(self) -> None:
        ty = Intersection(Constructor("In", LVar("a")), Constructor("Out", LVar("b")))

        path1 = Constructor("In", Literal(3, "int"))
        path2 = Constructor("Out", Literal(4, "int"))

        sub = Subtypes({})

        groups = {"a": "int", "b": "int"}
        self.logger.info(f"{ty} <= {path1}")
        self.logger.info(f"{ty} <= {path2}")
        self.logger.info(
            f"test_for_real: {sub.infer_substitution(ty, path1, groups)} and {sub.infer_substitution(ty, path2, groups)}"
        )


if __name__ == "__main__":
    unittest.main()
