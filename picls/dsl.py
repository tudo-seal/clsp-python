from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any
from inspect import signature

from .types import Arrow, Literal, Param, SetTo, Type


def TRUE(_: Mapping[str, Literal]) -> bool:
    return True


def unwrap_predicate(
    predicate: Callable[..., Any]
) -> Callable[[Mapping[str, Literal | Any]], Any]:
    needed_parameters = signature(predicate).parameters
    return lambda vars: predicate(
        **{
            param: v.value if isinstance(v, Literal) else v
            for param, v in vars.items()
            if param in needed_parameters
        }
    )


def extracted_values(
    predicate: Callable[[Mapping[str, Any]], Any]
) -> Callable[[Mapping[str, Literal | Any]], Any]:
    return lambda vars: predicate(
        {k: v.value if isinstance(v, Literal) else v for k, v in vars.items()}
    )


class Use:
    def __init__(self, name: str, typ: Any) -> None:
        self.name = name
        self.type = typ
        self._predicate: Callable[[Mapping[str, Literal]], bool] | SetTo = TRUE
        self._accumulator: list[Use] = []

    def With(self, predicate: Callable[..., Any]) -> Use:
        self._predicate = unwrap_predicate(predicate)
        return self

    def As(self, set_to: Callable[..., Any]) -> Use:
        self._predicate = SetTo(unwrap_predicate(set_to))
        return self

    def Use(self, name: str, ty: Any) -> Use:
        accumulator = self._accumulator + [self]
        inner: Use = Use(name, ty)
        inner._accumulator = accumulator
        return inner

    def In(self, ty: Type) -> Param:
        pi: Param = Param(self.name, self.type, self._predicate, ty)
        for param in reversed(self._accumulator):
            pi = Param(param.name, param.type, param._predicate, pi)
        return pi


class DSL:
    def __init__(self) -> None:
        self._accumulator: list[
            tuple[str, Any, Callable[[Mapping[str, Literal]], bool] | SetTo]
        ] = []

    def Use(self, name: str, ty: Any) -> DSL:
        self._accumulator.append((name, ty, TRUE))
        return self

    def As(self, set_to: Callable[..., Any]) -> DSL:
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            SetTo(unwrap_predicate(set_to)),
        )
        return self

    def AsRaw(self, set_to: Callable[[Mapping[str, Any]], Any]) -> DSL:
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            SetTo(extracted_values(set_to)),
        )
        return self

    def WithRaw(self, predicate: Callable[..., Any]) -> DSL:
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            extracted_values(predicate),
        )
        return self

    def With(self, predicate: Callable[..., Any]) -> DSL:
        last_element = self._accumulator[-1]
        self._accumulator[-1] = (
            last_element[0],
            last_element[1],
            unwrap_predicate(predicate),
        )
        return self

    def In(self, ty: Type) -> Param | Type:
        return_type: Param | Type = ty
        for spec in reversed(self._accumulator):
            return_type = Param(*spec, return_type)
        return return_type


class Requires:
    def __init__(self, *arguments: Type) -> None:
        self._arguments = list(arguments)

    def Provides(self, target: Type) -> Type:
        return reduce(lambda a, b: Arrow(b, a), reversed(self._arguments + [target]))
