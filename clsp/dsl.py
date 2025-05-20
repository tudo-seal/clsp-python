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
            .Use("x", int)
            .Use("y", int, lambda vars: vars["x"] + 1)
            .Use("z", str)
            .SuchThat(lambda vars: len(vars["z"]) == vars["x"] + vars["y"])
            .In(<Type using Var("x"), Var("y") and Var("z")>)

        constructs a specification for a function with three parameters:
        - `x`: an integer
        - `y`: an integer, which is one more than `x`
        - `z`: a string, whose length is equal to the sum of `x` and `y`
        The `In` method specifies the function, which uses the variables `x`, `y`, and `z`.
    """

    def __init__(self) -> None:
        """
        Initialize the DSL object
        """

        self._result = lambda suffix: suffix

    @staticmethod
    def _wrap_predicates(
        predicates: Sequence[Callable[[Mapping[str, Any]], bool]],
    ) -> Callable[[Mapping[str, Literal]], bool]:
        """
        Transforms a sequence of predicate, that directly use the values of `vars` to one
        predicate that uses the `Literal` values of `vars` instead.
        """
        return lambda vars: all(p({k: v.value if isinstance(v, Literal) else v for k, v in vars.items()}) for p in predicates)

    @staticmethod
    def _wrap_constraint(
        constraint: Callable[[Mapping[str, Any]], bool],
    ) -> Callable[[Mapping[str, Literal]], bool]:
        """
        Transforms a sequence of predicate, that directly use the values of `vars` to one
        predicate that uses the `Literal` values of `vars` instead.
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
        Introduce a new variable.

        This can be twofold. Either `group` is a string, or `group` is a `Type`

        If `group` is a string, then an instance of this specification will be generated
        for each valid literal in the corresponding literal group.
        You can use this variable as Var(name) in all `Type`s, after the introduction
        and in all predicates.
        We call these variables "`Literal` variables".
        Optionally, you can specify a sequence of values, that will be used to generate
        the literals. This sequence is parameterized by the values of previously
        defined literal variables. This is useful, if you want to restrict the values of a variable
        to a subset of the values in the corresponding literal group.
        

        If `group` is a `Type`, an instance will be generated for each tree, satisfying
        the specification given by the type. Since this can only be done in the enumeration step,
        you can only use these variables in predicates, that themselves belong to variables whose `group` is a `Type`.
        We call these variables "`Type` variables".

        Using `Type` variables is generally more powerful, but since the corresponding predicates
        can only be evaluated in the enumeration step, this usually is slower.

        :param name: The name of the new variable.
        :type name: str
        :param group: The type of the variable.
        :type group: str | Type
        :param values: Parameterized sequence of values, that will be used to generate the literals.
        :type values: Callable[[dict[str, Any]], Sequence[Any]] | None
        :return: The DSL object.
        :rtype: DSL
        """
        wrapped_candidates = DSL._wrap_sequence(group, candidates) if candidates else None
        self._result = lambda suffix, result=self._result: result(Abstraction(LiteralParameter(name, group, wrapped_candidates), suffix))
        return self
    
    def Argument(self, name: str, specification: Type) -> DSL:
        self._result = lambda suffix, result=self._result: result(Abstraction(TermParameter(name, specification), suffix))
        return self
    
    def ParameterConstraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> DSL:
        wrapped_constraint = DSL._wrap_constraint(constraint)
        self._result = lambda suffix, result=self._result: result(Implication(Predicate(wrapped_constraint, True), suffix))
        return self
    
    def Constraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> DSL:
        """
        Predicate on the previously defined variables.

        If the last variable introduced was a `Type` variable, the predicate has access to all
        previously defined variable. Otherwise it has only access to `Literal` variables.

        :param predicate: A predicate deciding, if the currently chosen values are valid.
            The values of variables are passt by a dictionary, where the keys are the names of the
            variables and the values are the corresponding values.
        :type predicate: Callable[..., bool]
        :return: The DSL object.
        :rtype: DSL
        """
        wrapped_constraint = DSL._wrap_constraint(constraint)
        self._result = lambda suffix, result=self._result: result(Implication(Predicate(wrapped_constraint, False), suffix))
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
