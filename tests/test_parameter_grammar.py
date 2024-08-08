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
            RHSRule({"y": "Y"}, [], "x", [GVar("y"), GVar("y")], []),
        )
        self.grammar.add_rule(
            "Y",
            RHSRule({}, [Predicate(lambda _: True, "⊤")], "y1", [Literal(3, "int")], []),
        )
        self.grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False, "⊥")], "y2", [], []))

    def test_grammar(self) -> None:
        self.logger.info(self.grammar.show())
        self.assertEqual(
            "X ~> ∀(y:Y).x(<y>)(<y>)\nY ~> ⊤ ⇛ y1([3, int]) | ⊥ ⇛ y2",
            self.grammar.show(),
        )

    def test_enum(self) -> None:
        enumeration = enumerate_terms("X", self.grammar)

        for t in enumeration:
            self.logger.info(t)
            self.assertEqual(Tree("x", (Tree("y1", (Tree(3, ()),)), Tree("y1", (Tree(3, ()),)))), t)


if __name__ == "__main__":
    unittest.main()
