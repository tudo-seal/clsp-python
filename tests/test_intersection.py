from collections.abc import Callable
import logging
import unittest
from cls import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    inhabit_and_interpret,
)


class Test(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
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
                Y: d,
                F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
            }
        )
        self.real_results = [
            res for res in inhabit_and_interpret(repository, [Intersection(a, d), c])
        ]
        for res in self.real_results:
            self.logger.info(res)

    def test_length(self) -> None:
        self.assertEqual(3, len(self.real_results))

    def test_elements(self) -> None:
        self.assertEqual("X", self.real_results[0])
        self.assertIn("F(Y)", self.real_results)
        self.assertIn("F(X)", self.real_results)


if __name__ == "__main__":
    unittest.main()
