from __future__ import annotations

import logging
import unittest

from clsp import CoSy
from clsp.dsl import DSL
from clsp.synthesizer import Contains, Specification, ParameterSpace
from clsp.tree import Tree
from clsp.types import (
    Constructor,
    Literal,
    Var,
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


class BranchingPart(Part):
    def __call__(self, target_weight, left_part, right_part):
        return self.weight + left_part + right_part


class ExtendingPart(Part):
    def __call__(self, target_weight, extending_part, *rest):
        return self.weight + extending_part


class TerminalPart(Part):
    def __call__(self, target_weight):
        return self.weight


class TestFilterByCriteria(unittest.TestCase):
    """
    Provides a best practice example for efficiently filtering results of an inhabitation request based on
    properties that need to be computed on the trees, meaning they can not be modeled by literal variables.

    Weight is used as an example, as this computes inefficiently when using an infinite literal context, as this
    prevents efficient pre-computation.
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def compute_weight(self, tree: Tree[Part]) -> float:
        """
        Recursively computes the weight of the given tree.
        This is used in conjunction with type predicates to ensure that target weight is not exceeded.
        As trees are constructed bottom-up, this leads to efficient filtering on trees.

        :param tree: The uninterpreted tree.
        :return: The computed weight of the tree.
        """
        return tree.interpret()

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
        componentSpecifications: dict[Part, Specification] = {
            BranchingPart("Double Motor", 1): DSL()
            .Use("target_weight", "float")
            .Use("left_part", Constructor("Structural") & Var("target_weight"))
            .Use("right_part", Constructor("Structural") & Var("target_weight"))
            .SuchThat(
                lambda vars: vars["target_weight"]
                > self.compute_weight(vars["left_part"])
                + self.compute_weight(vars["right_part"])
                + 1
            )
            .In(Constructor("Structural") & Var("target_weight")),
            ExtendingPart("Extrusion", 0.5): DSL()
            .Use("target_weight", "float")
            .Use("next_part", Constructor("Structural") & Var("target_weight"))
            .SuchThat(
                lambda vars: vars["target_weight"] > self.compute_weight(vars["next_part"]) + 0.5
            )
            .In(Constructor("Structural") & Var("target_weight")),
            TerminalPart("Effector", 0.1): DSL()
            .Use("target_weight", "float")
            .SuchThat(lambda vars: vars["target_weight"] > 0.1)
            .In(Constructor("Structural") & Var("target_weight")),
        }

        class Float(Contains):
            def __contains__(self, value: object) -> bool:
                return isinstance(value, float) and value >= 0.0

        parameterSpace: ParameterSpace = {"float": Float()}
        cosy = CoSy(componentSpecifications, parameterSpace)
        self.solutions = list(
            cosy.solve(Constructor("Structural") & Literal(2.0, "float"), max_count=1249)
        )

    def test_count(self) -> None:
        """
        Tests if the number of results is as expected.

        :return:
        """
        self.assertEqual(8, len(self.solutions))

    def test_elements(self) -> None:
        """
        Tests if the interpreted trees are as expected.

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
        for solution in self.solutions:
            print(solution)
            self.logger.info(solution)
            # self.assertIn(solution, results)

    def test_compute_weight(self) -> None:
        """
        Tests if the weights of the trees are feasible.
        Tests if unknown parts lead to an exception.

        :return:
        """
        weights = [0.1, 0.6, 1.1, 1.2, 1.6, 1.7]
        for solution in self.solutions:
            self.logger.info(solution)
            self.assertIn(solution, weights)

        unhandled_tree: Tree[Part] = Tree(Part("Unhandled Part", 0))
        self.assertRaises(RuntimeError, self.compute_weight, unhandled_tree)


if __name__ == "__main__":
    unittest.main()
