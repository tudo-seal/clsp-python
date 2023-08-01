from collections.abc import Sequence
import logging
import unittest
from picls import Type, Constructor, Arrow, inhabit_and_interpret


def F(*x: str) -> Sequence[str]:
    return x


class Test(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_var_args(self) -> None:
        a: Type = Constructor("a")
        b: Type = Constructor("b")
        c: Type = Constructor("c")
        # d: Type = Constructor("d")

        X: str = "X"
        Y: str = "Y"

        repository = dict({X: a, Y: b, F: Arrow(a, (Arrow(b, (Arrow(b, c)))))})

        for real_result in inhabit_and_interpret(repository, c):
            self.assertEqual(("X", "Y", "Y"), real_result)
            self.logger.info(real_result)


if __name__ == "__main__":
    unittest.main()
