# pylint: disable=invalid-name
"""
This module provides a `DSL` class, which allows users to define specifications
in a declarative manner using a fluent interface. It also includes the `Requires` class,
which is a builder for arrow types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any
from inspect import signature

from .types import Arrow, Literal, Param, SetTo, Type


class DSL:
    """
    A domain-specific language (DSL) to define specification.

    This class provides a interface for defining specifications in a declarative manner. It allows
    users to specify the name and type of each parameter, as well as filter.

    Examples:
        DSL()
            .Use("x", int)
            .Use("y", int)
            .As(lambda x: x + 1)
            .Use("z", str)
            .With(lambda x y z: len(z) == x + y)
            .In(<Type using TVar("x"), TVar("y") and TVar("z")>)

        This corresponds to:
            Param("x", int, lambda _: True,
                Param("y", int, SetTo(lambda vars: vars["x"].value + 1)
                    Param("z", str,
                        lambda vars: len(vars["z"].value) == vars["x"].value + vars["y"].value,
                        <Type using TVar("x"), TVar("y") and TVar("z")>
                        )
                    )
                )
    """

    @staticmethod
    def TRUE(_: Mapping[str, Literal]) -> bool:
        """
        A predicate, that constantly returns True
        """
        return True

    @staticmethod
    def _unwrap_predicate(
        predicate: Callable[..., Any]
    ) -> Callable[[Mapping[str, Literal | Any]], Any]:
        """
        Transforms a DSL-predicate to a predicate, that the `Param`-class can use.

        The names of the parameters from `predicate` are looked up in the dict, that holds the
        values for the variables.

        :param predicate: The DSL predicate function.
        :type predicate: Callable[..., Any]
        :return: The Param predicate function.
        :rtype: Callable[[Mapping[str, Literal | Any]], Any]
        """
        needed_parameters = signature(predicate).parameters
        return lambda vars: predicate(
            **{
                param: v.value if isinstance(v, Literal) else v
                for param, v in vars.items()
                if param in needed_parameters
            }
        )

    @staticmethod
    def _extracted_values(
        predicate: Callable[[Mapping[str, Any]], Any]
    ) -> Callable[[Mapping[str, Literal | Any]], Any]:
        """
        Transforms a predicate, that directly uses the values of `vars` to a `Param`-predicate

        :param predicate: The predicate function, using the values directly.
        :type predicate: Callable[[Mapping[str, Any]], Any]
        :return: The `Param` predicate function.
        :rtype: Callable[[Mapping[str, Literal | Any]], Any]
        """

        return lambda vars: predicate(
            {k: v.value if isinstance(v, Literal) else v for k, v in vars.items()}
        )

    def __init__(self) -> None:
        """
        Initialize the DSL object
        """
        self._accumulator: list[
            tuple[str, Any, Callable[[Mapping[str, Literal]], bool] | SetTo]
        ] = []

    def Use(self, name: str, ty: Any) -> DSL:
        """
        Introduce a new variable.

        If `ty` is a `LiteralType`, you can use its value directly in all filters and in the type at
        the end using `TVar(name)`. If `ty` is a `Type`, you can only use it other filters
        corresponding to `Type` variables, these will be set to Terms inhabiting the Type.

        Using `Type` variables is generally more powerful, but also generally slower.

        :param name: The name of the new variable.
        :type name: str
        :param ty: The type of the variable.
        :type ty: Any
        :return: The DSL object. To concatenate multiple calls.
        :rtype: DSL
        """
        self._accumulator.append((name, ty, DSL.TRUE))
        return self

    def As(self, set_to: Callable[..., Any]) -> DSL:
        """
        Set the previous variable directly to the result of a computation.

        Only available to `Literal` variables. And can only access `Literal` variables.

        :param set_to: The function computing the value for the variable. The names of the
            parameters to this function correspond directly to the names of the variables,
            previously introduced.
        :type set_to: Callable[..., Any]
        :return: The DSL object. To concatenate multiple calls.
        :rtype: DSL
        """
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            SetTo(DSL._unwrap_predicate(set_to)),
        )
        return self

    def AsRaw(self, set_to: Callable[[Mapping[str, Any]], Any]) -> DSL:
        """
        Set the previous variable directly to the result of a computation.

        Similar to `As`, but the `set_to` function gets a dictionary, mapping the variable names
        to their values instead.

        Only available to `Literal` variables. And can only access `Literal` variables.

        :param set_to: The function computing the value for the variable.
        :type set_to: Callable[[Mapping[str, Any]], Any]
        :return: The DSL object. To concatenate multiple calls.
        :rtype: DSL
        """
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            SetTo(DSL._extracted_values(set_to)),
        )
        return self

    def With(self, predicate: Callable[..., Any]) -> DSL:
        """
        Filter on the previously definied variables.

        If the last variable introduced was a variable with a `Type`-type (as opposed to a
        `Literal`-type), the predicate has access to all previously defined variable. Otherwise
        it has only access to `Literal` variables.

        *Note:* Filtering on `Literal` variables is significantly faster than filtering on
        `Type` variable

        :param predicate: A predicate deciding, if the currently chosen values are valid. The names
            of the parameters to this function correspond directly to the names of the variables,
            previously introduced.
        :type predicate: Callable[..., bool]
        :return: The DSL object.
        :rtype: DSL
        """
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            DSL._unwrap_predicate(predicate),
        )
        return self

    def WithRaw(self, predicate: Callable[[Mapping[str, Any]], Any]) -> DSL:
        """
        Filter on the previously definied variables.

        Similar to `With`, but the `pred` function gets a dictionary, mapping the variable names
        to their values instead.

        :param predicate: A predicate deciding, if the currently chosen values are valid.
        :type predicate: Callable[[Mapping[str, Any]], bool]
        :return: The DSL object.
        :rtype: DSL
        """
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            DSL._extracted_values(predicate),
        )
        return self

    def In(self, ty: Type) -> Param | Type:
        """
        Constructs the final specification wrapping the given `Type` `ty`.

        If no variables are declared, this returns `ty` directly, otherwise a `Param`.

        :param ty: The wrapped type.
        :type ty: Type
        :return: The constructed specification.
        :rtype: Param | Type
        """
        return_type: Param | Type = ty
        for spec in reversed(self._accumulator):
            return_type = Param(*spec, return_type)
        return return_type


# pylint: disable=too-few-public-methods
class Requires:
    """
    Builder for arrow types.

    Requires(a, b, c).Provides(d) ~> Arrow(a, Arrow(b, Arrow(c, d)))
    """

    def __init__(self, *arguments: Type) -> None:
        """
        Initializes the Requires object.

        :param arguments: The types of the arguments.
        :type arguments: Type
        """

        self._arguments = list(arguments)

    def Provides(self, target: Type) -> Type:
        """
        Returns the arrow type that represents the Requires object.

        :param target: The type of the target.
        :type target: Type
        :return: The arrow type.
        :rtype: Type
        """
        return reduce(lambda a, b: Arrow(b, a), reversed(self._arguments + [target]))