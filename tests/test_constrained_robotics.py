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


STRONG_MOTOR: str = "Strong Motor"
WEAK_MOTOR: str = "Weak Motor"
BRANCHING_PART: str = "Branching Part"
EXTENDING_PART: str = "Extending Part"
EFFECTOR: str = "Effector"

PART_PROPERTIES: dict[str, dict[str, float]] = {
    STRONG_MOTOR: {"weight": 1.0, "torque": 15.0, "angle": 70.0, "length": 5.0},
    WEAK_MOTOR: {"weight": 0.8, "torque": 4.0, "angle": 90.0, "length": 5.0},
    BRANCHING_PART: {"weight": 0.8, "torque": 1.0, "angle": 90.0, "length": 5.0},
    EXTENDING_PART: {"weight": 0.5, "length": 50.0},
    EFFECTOR: {"weight": 0.1, "torque": 1.0, "length": 5.0},
}

max_torque_algebra = {
    STRONG_MOTOR: lambda _, __, ___, next_part: max(
        PART_PROPERTIES.get(STRONG_MOTOR).get("torque"), next_part
    ),
    WEAK_MOTOR: lambda _, __, ___, next_part: max(
        PART_PROPERTIES.get(WEAK_MOTOR).get("torque"), next_part
    ),
    BRANCHING_PART: lambda _, __, ____, _____, left_part, right_part: max(left_part, right_part),
    EXTENDING_PART: lambda _, __, next_part: next_part,
    EFFECTOR: lambda _: 0.0,
}

center_of_mass_algebra = {
    STRONG_MOTOR: lambda _, __, ___, next_part: compute_center_of_mass_and_weight(
        (
            PART_PROPERTIES.get(STRONG_MOTOR).get("length") / 2,
            PART_PROPERTIES.get(STRONG_MOTOR).get("weight"),
        ),
        (
            PART_PROPERTIES.get(STRONG_MOTOR).get("length") + next_part[0],
            next_part[1],
        ),
    ),
    WEAK_MOTOR: lambda _, __, ___, next_part: compute_center_of_mass_and_weight(
        (
            PART_PROPERTIES.get(WEAK_MOTOR).get("length") / 2,
            PART_PROPERTIES.get(WEAK_MOTOR).get("weight"),
        ),
        (
            PART_PROPERTIES.get(WEAK_MOTOR).get("length") + next_part[0],
            next_part[1],
        ),
    ),
    BRANCHING_PART: lambda _, __, ____, _____, left_part, right_part: compute_center_of_mass_and_weight(
        (
            PART_PROPERTIES.get(BRANCHING_PART).get("length") / 2,
            PART_PROPERTIES.get(BRANCHING_PART).get("weight"),
        ),
        (
            PART_PROPERTIES.get(BRANCHING_PART).get("length") + left_part[0],
            left_part[1],
        ),
        (
            PART_PROPERTIES.get(BRANCHING_PART).get("length") + right_part[0],
            right_part[1],
        ),
    ),
    EXTENDING_PART: lambda _, __, next_part: compute_center_of_mass_and_weight(
        (
            PART_PROPERTIES.get(EXTENDING_PART).get("length") / 2,
            PART_PROPERTIES.get(EXTENDING_PART).get("weight"),
        ),
        (
            PART_PROPERTIES.get(EXTENDING_PART).get("length") + next_part[0],
            next_part[1],
        ),
    ),
    EFFECTOR: lambda _: compute_center_of_mass_and_weight(
        (
            PART_PROPERTIES.get(EFFECTOR).get("length") / 2,
            PART_PROPERTIES.get(EFFECTOR).get("weight"),
        )
    ),
}


def compute_center_of_mass_and_weight(
    *center_weights: list[tuple[float, float]]
) -> tuple[float, float]:
    """

    :param centers:
    :param weights:
    :return:
    """
    sum_weight = sum([weight for _, weight in center_weights])
    return sum([center * weight for center, weight in center_weights]) / sum_weight, sum_weight


def check_torque_constraints(id: str, *trees: Tree) -> bool:
    centers_weights = [interpret_term(tree, center_of_mass_algebra) for tree in trees]
    max_torque = max([interpret_term(tree, max_torque_algebra) for tree in trees])
    for center, weight in centers_weights:
        if "Motor" in id:
            # print(
            #     f"{id}: {PART_PROPERTIES.get(id).get("torque")} Nm, Needed: {weight * (center + PART_PROPERTIES.get(id).get("length")) * 0.0981} Nm"
            # )
            if (
                PART_PROPERTIES.get(id).get("torque")
                < weight * (center + PART_PROPERTIES.get(id).get("length")) * 0.0981
            ):
                return False
            if max_torque > PART_PROPERTIES.get(id).get("torque"):
                return False
    return True


class TestConstrainedRobotics(unittest.TestCase):
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

    def setUp(self) -> None:
        """

        :return:
        """

        repo: dict[str, Type | Param] = {
            #
            ###########################################################################################################
            # [IDENTIFIER]: STRONG MOTOR
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, TORQUE, ANGLE, LENGTH
            ###########################################################################################################
            #
            STRONG_MOTOR: DSL()
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
                > interpret_term(next_part, center_of_mass_algebra)[1]
                + PART_PROPERTIES.get(STRONG_MOTOR).get("weight")
            )
            .With(lambda next_part: check_torque_constraints(STRONG_MOTOR, next_part))
            .In((Constructor("Motor") & ("c" @ LVar("target_weight"))) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: WEAK MOTOR
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, TORQUE, ANGLE, LENGTH
            ###########################################################################################################
            #
            WEAK_MOTOR: DSL()
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
                > interpret_term(next_part, center_of_mass_algebra)[1]
                + PART_PROPERTIES.get(WEAK_MOTOR).get("weight")
            )
            .With(lambda next_part: check_torque_constraints(WEAK_MOTOR, next_part))
            .In((Constructor("Motor") & ("c" @ LVar("target_weight"))) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: BRANCHING PART
            # [CHILDREN]: 2
            # [PROPERTIES]: WEIGHT, LENGTH
            ###########################################################################################################
            #
            BRANCHING_PART: DSL()
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
                > interpret_term(left_part, center_of_mass_algebra)[1]
                + interpret_term(right_part, center_of_mass_algebra)[1]
                + PART_PROPERTIES.get(BRANCHING_PART).get("weight")
            )
            .In(Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("new_dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: EXTENDING PART
            # [CHILDREN]: 1
            # [PROPERTIES]: WEIGHT, LENGTH
            ###########################################################################################################
            #
            EXTENDING_PART: DSL()
            .Use("target_weight", "weight")
            .Use("dof", "dofs")
            .Use(
                "next_part",
                Constructor("Structural") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof")),
            )
            .With(
                lambda target_weight, next_part: target_weight
                > interpret_term(next_part, center_of_mass_algebra)[1]
                + PART_PROPERTIES.get(EXTENDING_PART).get("weight")
            )
            .In(Constructor("Inert") & ("c" @ LVar("target_weight")) & ("c" @ LVar("dof"))),
            #
            ###########################################################################################################
            # [IDENTIFIER]: EFFECTOR
            # [CHILDREN]: 0
            # [PROPERTIES]: WEIGHT, TORQUE, LENGTH
            ###########################################################################################################
            #
            EFFECTOR: DSL()
            .Use("target_weight", "weight")
            .With(lambda target_weight: target_weight > PART_PROPERTIES.get(EFFECTOR).get("weight"))
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
        query = (
            Constructor("Motor")
            & ("c" @ (Literal(targets.get("weight"), "weight")))
            & ("c" @ Literal(targets.get("dof"), "dofs"))
        )
        grammar = fcl.inhabit(query)
        terms = list(enumerate_terms(query, grammar, max_count=20))
        for term in terms:
            print(term)

    def test_count(self) -> None:
        """
        Tests if the number of results is as expected.

        :return:
        """
        self.assertEqual(4, len(self.terms))


if __name__ == "__main__":
    unittest.main()
