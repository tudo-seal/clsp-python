import logging
import unittest
from clsp.enumeration import Tree, enumerate_terms
from clsp.grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule
from clsp.types import Literal


class TestParamGrammar(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.grammar: ParameterizedTreeGrammar[str, str] = ParameterizedTreeGrammar()
        self.grammar.add_rule(
            "X",
            RHSRule({"y": "Y"}, [], "x", [GVar("y"), GVar("y")], ["y", "y"], []),
        )
        self.grammar.add_rule(
            "Y",
            RHSRule({}, [Predicate(lambda _: True, "⊤")], "y1", [Literal(1, "int")], ["n"], []),
        )
        self.grammar.add_rule(
            "Y",
            RHSRule({}, [], "y2", [], [], []),
        )
        self.grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False, "⊥")], "y3", [], [], []))

    def test_grammar(self) -> None:
        self.logger.info(self.grammar.show())
        self.assertEqual(
            "X ~> ∀(y:Y).x(<y>)(<y>)\nY ~> ⊤ ⇛ y1([1, int]) | y2 | ⊥ ⇛ y3",
            self.grammar.show(),
        )

    def test_enum(self) -> None:
        enumeration = enumerate_terms("X", self.grammar)
        expected_results = [
            Tree("x", (Tree("y1", (Tree(1, ()),), ["n"]), Tree("y1", (Tree(1, ()),), ["n"])), ["y", "y"],),
            Tree("x", (Tree("y2", ()), Tree("y2", ())), ["y", "y"]),
        ]
        self.assertCountEqual(enumeration, expected_results)


if __name__ == "__main__":
    unittest.main()
