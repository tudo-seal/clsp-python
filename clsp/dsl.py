# pylint: disable=invalid-name
"""
This module provides a `DSL` class, which allows users to define specifications
in a declarative manner using a fluent interface.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .types import Literal, LiteralParameter, TermParameter, Predicate, Type, Abstraction, Implication

class DSL:
    """
    A domain-specific language (DSL) to define component specifications.

    This class provides a interface for defining specifications in a declarative manner. It allows
    users to specify the name and group of each parameter, as well as filter.

    Examples:
        DSL()
            .Parameter("x", int)
            .Parameter("y", int, lambda vars: vars["x"] + 1)
            .Parameter("z", str)
            .ParameterConstraint(lambda vars: len(vars["z"]) == vars["x"] + vars["y"])
            .Suffix(<Type using Var("x"), Var("y") and Var("z")>)

        constructs a specification for a function with three parameters:
        - `x`: an integer
        - `y`: an integer, the value of which is `x` + 1
        - `z`: a string, whose length is equal to `x` + `y`
        The `Suffix` method specifies the function, which uses the variables `x`, `y`, and `z`.
    """

    def __init__(self) -> None:
        """
        Initialize the DSL object
        """

        self._result: Callable[[Abstraction | Implication | Type], Abstraction | Implication | Type] = lambda suffix: suffix

    @staticmethod
    def _wrap_constraint(
        constraint: Callable[[Mapping[str, Any]], bool],
    ) -> Callable[[Mapping[str, Literal]], bool]:
        """
        Transforms a constraint, that directly use the values of `vars` to one
        that uses the `Literal` values of `vars` instead.
        """
        return lambda vars: constraint({k: v.value if isinstance(v, Literal) else v for k, v in vars.items()})

    @staticmethod
    def _wrap_sequence(
        group: str,
        values: Callable[[dict[str, Any]], Sequence[Any]],
    ) -> Callable[[dict[str, Literal]], Sequence[Literal]]:
        """
        Transforms a parameterized sequence, that directly uses the values of `vars` to one
        that refers to the `Literal` values of `vars` instead.
        """

        return lambda vars: [Literal(value, group) for value in values({k: v.value for k, v in vars.items()})]
    
    
    def Parameter(self, name: str, group: str, candidates: Callable[[dict[str, Any]], Sequence[Any]] | None = None) -> DSL:
        """
        Introduce a new parameter variable.

        `group` is a string, and an instance of this specification will be generated
        for each valid literal in the corresponding literal group.
        You can use this variable as Var(name) in all `Type`s, after the introduction
        and in all predicates.
        Optionally, you can specify a sequence of candidate values, that will be used to generate
        the literals. This sequence is parameterized by the values of previously
        defined literal variables. This is useful, if you want to restrict the values of a variable
        to a subset of the values in the corresponding literal group.

        :param name: The name of the new variable.
        :type name: str
        :param group: The group of the variable.
        :type group: str
        :param candidates: Parameterized sequence of candidate values, that will be used to generate the literals.
        :type candidates: Callable[[dict[str, Any]], Sequence[Any]] | None
        :return: The DSL object.
        :rtype: DSL
        """
        wrapped_candidates = DSL._wrap_sequence(group, candidates) if candidates else None
        def new_result(suffix: Abstraction | Implication | Type, result=self._result) -> Abstraction | Implication | Type:
            return result(Abstraction(LiteralParameter(name, group, wrapped_candidates), suffix))
        self._result = new_result
        return self
    
    def Argument(self, name: str, specification: Type) -> DSL:
        """
        Introduce a new variable.

        `group` is a `Type`, and an instance will be generated for each tree, satisfying
        the specification given by the type. Since this can only be done in the enumeration step,
        you can only use these variables in predicates, that themselves belong to variables whose `group` is a `Type`.

        :param name: The name of the new variable.
        :type name: str
        :param specification: The type of the variable.
        :type specification: Type
        :return: The DSL object.
        :rtype: DSL
        """
        def new_result(suffix: Abstraction | Implication | Type, result=self._result) -> Abstraction | Implication | Type:
            return result(Abstraction(TermParameter(name, specification), suffix))
        self._result = new_result
        return self
    
    def ParameterConstraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> DSL:
        """
        Constraint on the previously defined parameter variables.

        :param constraint: A constraint deciding, if the currently chosen parameter values are valid.
            The values of variables are passed by a dictionary, where the keys are the names of the
            parameter variables and the values are the corresponding values.
        :type constraint: Callable[[Mapping[str, Any]], bool]
        :return: The DSL object.
        :rtype: DSL
        """
        wrapped_constraint = DSL._wrap_constraint(constraint)
        def new_result(suffix: Abstraction | Implication | Type, result=self._result) -> Abstraction | Implication | Type:
            return result(Implication(Predicate(wrapped_constraint, True), suffix))
        self._result = new_result
        return self
    
    def Constraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> DSL:
        """
        Constraint on the previously defined parameter variables and argument variables.

        :param constraint: A constraint deciding, if the currently chosen values are valid.
            The values of variables are passed by a dictionary, where the keys are the names of the
            variables and the values are the corresponding values.
        :type constraint: Callable[[Mapping[str, Any]], bool]
        :return: The DSL object.
        :rtype: DSL
        """
        wrapped_constraint = DSL._wrap_constraint(constraint)
        def new_result(suffix: Abstraction | Implication | Type, result=self._result) -> Abstraction | Implication | Type:
            return result(Implication(Predicate(wrapped_constraint, False), suffix))
        self._result = new_result
        return self
    
    def Suffix(self, suffix: Type) -> Abstraction | Implication | Type:
        """
        Constructs the final specification wrapping the given `Type` `suffix`.

        :param suffix: The wrapped type.
        :type suffix: Type
        :return: The constructed specification.
        :rtype: Abstraction | Type
        """
        return self._result(suffix)
