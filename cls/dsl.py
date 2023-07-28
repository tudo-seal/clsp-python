from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any

from cls.types import Arrow, Literal, Param, SetTo, Type


def TRUE(_: Mapping[str, Literal]) -> bool:
    return True


class Use:
    def __init__(self, name: str, typ: Any) -> None:
        self.name = name
        self.type = typ
        self._predicate: Callable[[Mapping[str, Literal]], bool] | SetTo = TRUE
        self._accumulator: list[Use] = []

    def With(self, predicate: Callable[[Mapping[str, Literal | Any]], bool]) -> Use:
        self._predicate = predicate
        return self

    def As(self, set_to: Callable[[Mapping[str, Literal]], Any]) -> Use:
        self._predicate = SetTo(set_to)
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


class Requires:
    def __init__(self, *arguments: Type) -> None:
        self._arguments = list(arguments)

    def Provides(self, target: Type) -> Type:
        return reduce(lambda a, b: Arrow(b, a), reversed(self._arguments + [target]))
