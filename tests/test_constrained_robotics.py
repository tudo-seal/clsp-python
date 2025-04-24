from __future__ import annotations

import logging
import unittest

from clsp import Subtypes
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

    def __init__(self, name: str, properties: dict):
        self.name = name
        self.properties = properties

    def __call__(self, *collect: Part) -> str:
        return (self.name + " params: " + str(collect)).replace("\\", "")

    def __repr__(self) -> str:
        return self.name


class TestConstrainedRobotics(unittest.TestCase):
    """
    Provides a best practice example for efficiently filtering results of an inhabitation request based on
    properties that need to be computed on the terms, meaning they can not be modeled by literal variables.

    Weight is used as an example, as this computes inefficiently when using an infinite literal context, as this
    prevents efficient pre-computation.
    """

    STRONG_MOTOR = "Strong Motor"
    WEAK_MOTOR = "Weak Motor"
    BRANCHING_PART = "Branching Part"
    EXTENDING_PART = "Extending Part"
    EFFECTOR = "Effector"

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def compute_weight(self, tree: Tree[Part]) -> float:
        """
        Recursively computes the weight of the given term.
        This is used in conjunction with type predicates to ensure that target weight is not exceeded.
        As terms are constructed bottom-up, this leads to efficient filtering on terms.

        Note that all parameters that are not literals but represent structural parts need to be affixed or suffixed
        with "part".

        :param tree: The uninterpreted term.
        :return: The computed weight of the term.
        """
        return tree.root.properties.get("weight") + sum(
            self.compute_weight(part) for key, part in tree.parameters.items() if "part" in key
        )

    def compute_torque(self, tree: Tree[Part]) -> float:
        """

        :param tree: The uninterpreted term.
        :return: The computed weight of the term.
        """
        return 0.0

    def check_torque_strictly_descending(self, tree: Tree[Part], torque) -> bool:
        pass

    def setUp(self) -> None:
        """

        :return:
        """
        part_properties: dict[str, dict[str, float]] = {
            self.STRONG_MOTOR: {"weight": 1.0, "torque": 3.0, "angle": 70.0, "length": 5.0},
            self.WEAK_MOTOR: {"weight": 0.8, "torque": 1.0, "angle": 90.0, "length": 5.0},
            self.BRANCHING_PART: {"weight": 0.8, "torque": 1.0, "angle": 90.0, "length": 5.0},
            self.EXTENDING_PART: {"weight": 0.5, "length": 50.0},
            self.EFFECTOR: {"weight": 0.1, "torque": 1.0, "length": 5.0},
        }

        weight_algebra = {
            self.STRONG_MOTOR: lambda _, __, ___, next_part: part_properties.get(
                self.STRONG_MOTOR
            ).get("weight")
            + next_part,
            self.WEAK_MOTOR: lambda _, __, ___, next_part: part_properties.get(self.WEAK_MOTOR).get(
                "weight"
            )
            + next_part,
            self.BRANCHING_PART: lambda _, __, ____, _____, left_part, right_part: part_properties.get(
                self.BRANCHING_PART
            ).get(
                "weight"
            )
            + left_part
            + right_part,
            self.EXTENDING_PART: lambda _, __, next_part: part_properties.get(
                self.EXTENDING_PART
            ).get("weight")
            + next_part,
            self.EFFECTOR: lambda _: part_properties.get(self.EFFECTOR).get("weight"),
        }

        max_torque_algebra = {
            self.STRONG_MOTOR: lambda _, __, ___, next_part: max(
                part_properties.get(self.STRONG_MOTOR).get("torque"), next_part
            ),
            self.WEAK_MOTOR: lambda _, __, ___, next_part: max(
                part_properties.get(self.WEAK_MOTOR).get("torque"), next_part
            ),
            self.BRANCHING_PART: lambda _, __, ____, _____, left_part, right_part: max(
                left_part, right_part
            ),
            self.EXTENDING_PART: lambda _, __, next_part: next_part,
            self.EFFECTOR: lambda _: 0.0,
        }

        repo: dict[Part, Type | Param] = {
            #
            ###########################################################################################################
            # [IDENTIFIER]: STRONG MOTOR
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, TORQUE, ANGLE, LENGTH
            ###########################################################################################################
            #
            self.STRONG_MOTOR: DSL()
            .Use("target_weight", "weight")
            .Use("dof", "dofs")
            .Use("new_dof", "dofs")
            .As(lambda dof: dof + 1)
            .Use(
                "next_part",
                Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof")),
            )
            .With(
                lambda target_weight, next_part: target_weight
                > interpret_term(next_part, weight_algebra)
                + part_properties.get(self.STRONG_MOTOR).get("weight")
            )
            .With(
                lambda next_part: interpret_term(next_part, max_torque_algebra)
                <= part_properties.get(self.STRONG_MOTOR).get("torque")
            )
            .In((Constructor("Motor") & ("c" @ LVar("target_weight"))) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: WEAK MOTOR
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, TORQUE, ANGLE, LENGTH
            ###########################################################################################################
            #
            self.WEAK_MOTOR: DSL()
            .Use("target_weight", "weight")
            .Use("dof", "dofs")
            .Use("new_dof", "dofs")
            .As(lambda dof: dof + 1)
            .Use(
                "next_part",
                Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof")),
            )
            .With(
                lambda target_weight, next_part: target_weight
                > interpret_term(next_part, weight_algebra)
                + part_properties.get(self.WEAK_MOTOR).get("weight")
            )
            .With(
                lambda next_part: interpret_term(next_part, max_torque_algebra)
                <= part_properties.get(self.WEAK_MOTOR).get("torque")
            )
            .In((Constructor("Motor") & ("c" @ LVar("target_weight"))) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: BRANCHING PART
            # [CHILDREN]: 2
            # [PROPERTIES]: WEIGHT, LENGTH
            ###########################################################################################################
            #
            self.BRANCHING_PART: DSL()
            .Use("target_weight", "weight")
            .Use("dof_l", "dofs")
            .Use("dof_r", "dofs")
            .Use("new_dof", "dofs")
            .As(lambda dof_l, dof_r: dof_l + dof_r)
            .Use(
                "left_part",
                Constructor("Structural") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof_l")),
            )
            .Use(
                "right_part",
                Constructor("Structural") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof_r")),
            )
            .With(
                lambda target_weight, left_part, right_part,: target_weight
                > interpret_term(left_part, weight_algebra)
                + interpret_term(right_part, weight_algebra)
                + part_properties.get(self.BRANCHING_PART).get("weight")
            )
            .In(Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: EXTENDING PART
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, LENGTH
            ###########################################################################################################
            #
            self.EXTENDING_PART: DSL()
            .Use("target_weight", "weight")
            .Use("dof", "dofs")
            .Use(
                "next_part",
                Constructor("Structural") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof")),
            )
            .With(
                lambda target_weight, next_part: target_weight
                > interpret_term(next_part, weight_algebra)
                + part_properties.get(self.EXTENDING_PART).get("weight")
            )
            .In(Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: EFFECTOR
            # [CHILDREN]: 0
            # [PROPERTIES]: WEIGHT, TORQUE, LENGTH
            ###########################################################################################################
            #
            self.EFFECTOR: DSL()
            .Use("target_weight", "weight")
            .With(
                lambda target_weight: target_weight
                > part_properties.get(self.EFFECTOR).get("weight")
            )
            .In(
                (Constructor("Motor") & ("c" @ LVar("target_weight"))) & ("c" @ Literal(0, "dofs"))
            ),
        }

        class Float(Contains):
            def __contains__(self, value: object) -> bool:
                return isinstance(value, float) and value >= 0.0

        targets = {"weight": 3.2, "dof": 2}
        literals = {
            "weight": Float(),
            "length": Float(),
            "dofs": list(range(targets.get("dof") + 1)),
        }
        environment: dict[str, set[str]] = {"Motor": {"Structural"}, "Inert": {"Structural"}}

        subtypes = Subtypes(environment)
        fcl = FiniteCombinatoryLogic(repo, subtypes=subtypes, literals=literals)
        self.query = (
            Constructor("Motor")
            & ("c" @ (Literal(targets.get("weight"), "weight")))
            & ("c" @ Literal(targets.get("dof"), "dofs"))
        )
        self.grammar = fcl.inhabit(self.query)
        self.terms = list(enumerate_terms(self.query, self.grammar, max_count=20))
        for term in self.terms:
            print(term)

    def test_count(self) -> None:
        """
        Tests if the number of results is as expected.

        :return:
        """
        self.assertEqual(4, len(self.terms))


if __name__ == "__main__":
    unittest.main()
