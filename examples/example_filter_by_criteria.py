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
    Omega,
    Var,
)


class Part:
    """
    Represents the properties of structural parts, in this case their weight.
    """

    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def __repr__(self) -> str:
        return self.name


class BranchingPart(Part):

    # TODO what if not hashable?
    def __call__(self, left_part, right_part, *rest):
        return self.weight + left_part + right_part
    
    # TODO is a combinator is a string then this is not possible
    def __weight__(self, left_part, right_part, *rest):
        return self.weight + left_part + right_part


class ExtendingPart(Part):
    def __call__(self, target_weight, extending_part):
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
        pass

if __name__ == "__main__":
    t = DSL().Parameter("a", "float").Argument("x", Omega()).Parameter("b", "int").Argument("y", Omega()).Constraint(lambda vars: True).ParameterConstraint(lambda vars: True).Suffix(Var("target_weight"))
    print(t)
    unittest.main()
