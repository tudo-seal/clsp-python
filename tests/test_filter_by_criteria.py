from __future__ import annotations

import logging
import unittest
from typing import Any

from clsp.enumeration import enumerate_terms, interpret_term, Tree
from clsp.fcl import FiniteCombinatoryLogic, Contains
from clsp.dsl import DSL
from clsp.types import (
    Constructor,
    Literal,
    Param,
    LVar,
    Type,
)


class Part:
    """
    Represents the properties of structural parts, in this case their weight.
    """
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def __call__(self, *collect: Part) -> str:
        return (self.name + " params: " + str(collect)).replace("\\", "")

    def __repr__(self) -> str:
        return self.name


class TestFilterByCriteria(unittest.TestCase):
    """
    Provides a best practice example for efficiently filtering results of an inhabitation request based on
    properties that need to be computed on the terms, meaning they can not be modeled by literal variables.

    Weight is used as an example, as this computes inefficiently when using an infinite literal context, as this
    prevents efficient pre-computation.
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def compute_weight(self, tree: Tree[Any, Part]) -> float:
        """
        Recursively computes the weight of the given term.
        This is used in conjunction with type predicates to ensure that target weight is not exceeded.
        As terms are constructed bottom-up, this leads to efficient filtering on terms.

        :param tree: The uninterpreted term.
        :return: The computed weight of the term.
        """
        match tree.root.name:
            case "Branching Part":
                return (
                    self.compute_weight(tree.parameters["left_part"])
                    + self.compute_weight(tree.parameters["right_part"])
                    + tree.root.weight
                )
            case "Extending Part":
                return self.compute_weight(tree.parameters["next_part"]) + tree.root.weight
            case "Effector":
                return tree.root.weight
            case _:
                raise RuntimeError("Unhandled Part in Repository.")

    def setUp(self) -> None:
        """
        Creates a repository containing branching structural parts (A -> A -> A), extending structural parts (A ->
        A), and effectors (A). The parts have different weights.

        The Predicates prevent each part in the repository from exceeding a weight limit, target_weight (2.0).
        The weight is a literal predicate whose permissible values are floats.

        All combinations of structural parts which form assemblies that weigh less than target_weight are requested.
        The performance is heavily affected by the max_count of the enumeration.

        A perplexing example is choosing a target weight of 5, which leads to 1248 results. Without supplying a
        max_count, the enumeration takes around 10 seconds. If setting max_count to 1248, the enumration only takes
        around 1.6 seconds. Setting the max_count to 1249, the enumeration takes around 10 seconds again.

        :return:
        """
        repo: dict[Part, Type | Param] = {
            Part("Branching Part", 1): DSL()
            .Use("target_weight", "float")
            .Use("left_part", Constructor("Structural") & ("c" @ LVar("target_weight")))
            .Use("right_part", Constructor("Structural") & ("c" @ LVar("target_weight")))
            .With(
                lambda target_weight, left_part, right_part,: target_weight
                > self.compute_weight(left_part) + self.compute_weight(right_part) + 1
            )
            .In(Constructor("Structural") & ("c" @ LVar("target_weight"))),
            Part("Extending Part", 0.5): DSL()
            .Use("target_weight", "float")
            .Use("next_part", Constructor("Structural") & ("c" @ LVar("target_weight")))
            .With(
                lambda target_weight, next_part: target_weight
                > self.compute_weight(next_part) + 0.5
            )
            .In(Constructor("Structural") & ("c" @ LVar("target_weight"))),
            Part("Effector", 0.1): DSL()
            .Use("target_weight", "float")
            .With(lambda target_weight: target_weight > 0.1)
            .In(Constructor("Structural") & ("c" @ LVar("target_weight"))),
        }

        class Float(Contains):
            def __contains__(self, value: object) -> bool:
                return isinstance(value, float) and value >= 0.0

        literals = {"float": Float()}

        fcl = FiniteCombinatoryLogic(repo, literals=literals)
        self.query = Constructor("Structural") & ("c" @ (Literal(2.0, "float")))
        self.grammar = fcl.inhabit(self.query)
        self.terms = list(enumerate_terms(self.query, self.grammar, max_count=1249))

    def test_count(self) -> None:
        """
        Tests if the number of results is as expected.

        :return:
        """
        self.assertEqual(8, len(self.terms))

    def test_elements(self) -> None:
        """
        Tests if the interpreted terms are as expected.

        :return:
        """
        results = [
            "Effector params: (2.0,)",
            "Branching Part params: (2.0, 'Effector params: (2.0,)', 'Effector params: (2.0,)')",
            "Extending Part params: (2.0, 'Effector params: (2.0,)')",
            "Branching Part params: (2.0, \"Extending Part params: (2.0, 'Effector params: (2.0,)')\", 'Effector "
            "params: (2.0,)')",
            "Branching Part params: (2.0, 'Effector params: (2.0,)', \"Extending Part params: (2.0, 'Effector params: "
            "(2.0,)')\")",
            "Extending Part params: (2.0, \"Extending Part params: (2.0, 'Effector params: (2.0,)')\")",
            "Extending Part params: (2.0, \"Branching Part params: (2.0, 'Effector params: (2.0,)', 'Effector params: "
            "(2.0,)')\")",
            "Extending Part params: (2.0, 'Extending Part params: (2.0, \"Extending Part params: (2.0, 'Effector "
            "params: (2.0,)')\")')",
        ]
        for term in self.terms:
            self.logger.info(interpret_term(term))
            self.assertIn(interpret_term(term), results)

    def test_compute_weight(self) -> None:
        """
        Tests if the weights of the terms are feasible.
        Tests if unknown parts lead to an exception.

        :return:
        """
        weights = [0.1, 0.6, 1.1, 1.2, 1.6, 1.7]
        for term in self.terms:
            self.logger.info(term)
            self.assertIn(self.compute_weight(term), weights)

        unhandled_tree: Tree[Any,Part] = Tree(Part("Unhandled Part", 0))
        self.assertRaises(RuntimeError, self.compute_weight, unhandled_tree)


if __name__ == "__main__":
    unittest.main()
