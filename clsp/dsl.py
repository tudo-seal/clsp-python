# pylint: disable=invalid-name
"""
This module provides a `DSL` class, which allows users to define specifications
in a declarative manner using a fluent interface. It also includes the `Requires` class,
which is a builder for arrow types.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import Any, overload
from inspect import signature
import typing

from .types import Arrow, Literal, Param, SetTo, Type


class DSL:
    """
    A domain-specific language (DSL) to define specification.

    This class provides a interface for defining specifications in a declarative manner. It allows
    users to specify the name and group of each parameter, as well as filter.

    Examples:
        DSL()
            .Use("x", int)
            .Use("y", int)
            .As(lambda x: x + 1)
            .Use("z", str)
            .With(lambda x y z: len(z) == x + y)
            .In(<Type using LVar("x"), LVar("y") and LVar("z")>)

        This corresponds to:
            Param("x", int, lambda _: True,
                Param("y", int, SetTo(lambda vars: vars["x"].value + 1)
                    Param("z", str,
                        lambda vars: len(vars["z"].value) == vars["x"].value + vars["y"].value,
                        <Type using LVar("x"), LVar("y") and LVar("z")>
                        )
                    )
                )

    The DSL object accepts the following configuration options at creation time:

      - cache (Default=False): Safe and reuse the full substitution space for a type. Can be used
            in some instances for performance gains. Should only be uses if a lot of literal
            variables are neither inferred nor computed via `As`.
      - infer (Default=True): Use a limited form of unification to infer the values for literal
            variables in the output type. Should only be deactivated for debug purposes.
    """

    @staticmethod
    def TRUE(_: Mapping[str, Literal]) -> bool:
        """
        A predicate, that constantly returns True
        """
        return True

    @staticmethod
    def _unwrap_predicate(
        predicate: Callable[..., Any],
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
        predicate: Callable[[Mapping[str, Any]], Any],
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

    def __init__(self, /, cache: bool = False, infer: bool = True) -> None:
        """
        Initialize the DSL object
        """
        self.cache = cache
        self.infer = infer
        self._accumulator: list[
            tuple[str, Any, list[Callable[[Mapping[str, Literal]], bool] | SetTo]]
        ] = []

    def Use(self, name: str, group: str | Type) -> DSL:
        """
        Introduce a new variable.

        This can be twofold. Either `group` is a string, or `group` is a `Type`

        If `group` is a string, then an instance of this specification will be generated
        for each valid literal in the corresponding literal group (That satisfy all given
        predicates). You can use this variable as LVar(name) in all `Type`s, after the introduction
        and in all predicates. Corresponding predicates will be evaluated when building the
        repository. We call these variables "`Literal` variables"

        If `group` is a `Type`, an instance will be generated for each combinatory term, satisfying
        the type. Since this can only be done in the enumeration step, you can only use this
        variables in predicates, that themselves belong to variables whose `group` is a `Type`.
        We call these variables "`Type` variables"

        Using `Type` variables is generally more powerful, but since the corresponding predicates
        can only be evaluated in the enumeration step, this is generally much slower.

        :param name: The name of the new variable.
        :type name: str
        :param group: The type of the variable.
        :type group: str | Type
        :return: The DSL object. To concatenate multiple calls.
        :rtype: DSL
        """
        self._accumulator.append((name, group, [DSL.TRUE]))
        return self

    @overload
    def As(
        self,
        set_to: Callable[..., Any],
        /,
        raw: typing.Literal[False] = ...,
        multi_value: typing.Literal[False] = ...,
        override: bool = False,
    ) -> DSL: ...

    @overload
    def As(
        self,
        set_to: Callable[[Mapping[str, Any]], Iterable[Any]],
        /,
        raw: typing.Literal[True] = ...,
        multi_value: typing.Literal[True] = ...,
        override: bool = False,
    ) -> DSL: ...

    @overload
    def As(
        self,
        set_to: Callable[[Mapping[str, Any]], Any],
        /,
        raw: typing.Literal[True] = ...,
        multi_value: typing.Literal[False] = ...,
        override: bool = False,
    ) -> DSL: ...

    @overload
    def As(
        self,
        set_to: Callable[..., Iterable[Any]],
        /,
        raw: typing.Literal[False] = ...,
        multi_value: typing.Literal[True] = ...,
        override: bool = False,
    ) -> DSL: ...

    def As(
        self,
        set_to: Callable[..., Any],
        /,
        raw: bool = False,
        multi_value: bool = False,
        override: bool = False,
    ) -> DSL:
        """
        Set the previous variable directly to the result of a computation.

        Only available to `Literal` variables. And can only access `Literal` variables.

        :param set_to: The function computing the value for the variable. The names of the
            parameters to this function correspond directly to the names of the variables,
            previously introduced.
        :type set_to: Callable[..., Any]
        :param override: Whether the result of the computation should be discarded, if it
            is not in the literal set for the group. Default is False (discard).
        :type override: bool
        :param raw: If True, `set_to` will be called with a dictionary, mapping variable names
            to values, instead of each variable as a parameter.
        :param multi_value: If `multi_value` is True, the function `set_to` needs to return an
            `Iterable`. Potential values for the variable range over those computed elements.
        :type multi_value: bool
        :return: The DSL object. To concatenate multiple calls.
        :rtype: DSL
        """

        unwrapper = DSL._unwrap_predicate if not raw else DSL._extracted_values
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            last_element[2] + [SetTo(unwrapper(set_to), override, multi_value)],
        )
        return self

    @overload
    def AsRaw(
        self,
        set_to: Callable[[Mapping[str, Any]], Any],
        /,
        override: bool = False,
        multi_value: typing.Literal[False] = False,
    ) -> DSL: ...

    @overload
    def AsRaw(
        self,
        set_to: Callable[[Mapping[str, Any]], Iterable[Any]],
        /,
        override: bool = False,
        multi_value: typing.Literal[True] = True,
    ) -> DSL: ...

    def AsRaw(
        self,
        set_to: Callable[[Mapping[str, Any]], Any],
        /,
        override: bool = False,
        multi_value: bool = False,
    ) -> DSL:
        """
        Deprecated, use As(... , raw = True) instead
        """
        if multi_value:  # This is for typing reasons
            self.As(set_to, override=override, raw=True, multi_value=True)
        else:
            self.As(set_to, override=override, raw=True, multi_value=False)
        return self

    @overload
    def With(
        self, predicate: Callable[[Mapping[str, Any]], Any], /, raw: typing.Literal[True] = True
    ) -> DSL: ...

    @overload
    def With(self, predicate: Callable[..., Any], /, raw: typing.Literal[False] = False) -> DSL: ...

    def With(self, predicate: Callable[..., Any], /, raw: bool = False) -> DSL:
        """
        Filter on the previously definied variables.

        If the last variable introduced was a `Type` variable, the predicate has access to all
        previously defined variable. Otherwise it has only access to `Literal` variables.
        Values from `Literal` variables can be used in the predicate directly, values from
        `Type` variables are given as their `Tree[T]`, that can be `interpret`ed in the
        predicate.

        *Note:* Filtering on `Literal` variables is significantly faster than filtering on
        `Type` variable

        :param predicate: A predicate deciding, if the currently chosen values are valid. The names
            of the parameters to this function correspond directly to the names of the variables,
            previously introduced.
        :type predicate: Callable[..., bool]
        :return: The DSL object.
        :rtype: DSL
        """
        unwrapper = DSL._unwrap_predicate if not raw else DSL._extracted_values

        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            last_element[2] + [unwrapper(predicate)],
        )
        return self

    def WithRaw(self, predicate: Callable[[Mapping[str, Any]], Any]) -> DSL:
        """
        Deprecated, use As(... , raw = True) instead
        """
        self.With(predicate, raw=True)
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
            return_type = Param(*spec, cache=self.cache, infer=self.infer, inner=return_type)
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
