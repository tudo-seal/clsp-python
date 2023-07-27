from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Type(ABC):
    is_omega: bool = field(init=True, kw_only=True, compare=False)
    size: int = field(init=True, kw_only=True, compare=False)
    organized: set[Type] = field(init=True, kw_only=True, compare=False)

    def __str__(self) -> str:
        return self._str_prec(0)

    def __mul__(self, other: Type) -> Type:
        return Product(self, other)

    @abstractmethod
    def _organized(self) -> set[Type]:
        pass

    @abstractmethod
    def _size(self) -> int:
        pass

    @abstractmethod
    def _is_omega(self) -> bool:
        pass

    @abstractmethod
    def _str_prec(self, prec: int) -> str:
        pass

    @staticmethod
    def _parens(s: str) -> str:
        return f"({s})"

    @staticmethod
    def intersect(types: Sequence[Type]) -> Type:
        if len(types) > 0:
            rtypes = reversed(types)
            result: Type = next(rtypes)
            for ty in rtypes:
                result = Intersection(ty, result)
            return result
        else:
            return Omega()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["is_omega"]
        del state["size"]
        del state["organized"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__["is_omega"] = self._is_omega()
        self.__dict__["size"] = self._size()
        self.__dict__["organized"] = self._organized()


@dataclass(frozen=True)
class Omega(Type):
    is_omega: bool = field(init=False, compare=False)
    size: bool = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return True

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type]:
        return set()

    def _str_prec(self, prec: int) -> str:
        return "omega"


@dataclass(frozen=True)
class Constructor(Type):
    name: str = field(init=True)
    arg: Type = field(default=Omega(), init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1 + self.arg.size

    def _organized(self) -> set[Type]:
        if len(self.arg.organized) <= 1:
            return {self}
        else:
            return {Constructor(self.name, ap) for ap in self.arg.organized}

    def _str_prec(self, prec: int) -> str:
        if self.arg == Omega():
            return str(self.name)
        else:
            return f"{str(self.name)}({str(self.arg)})"


@dataclass(frozen=True)
class Product(Type):
    left: Type = field(init=True)
    right: Type = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1 + self.left.size + self.right.size

    def _organized(self) -> set[Type]:
        if len(self.left.organized) + len(self.right.organized) <= 1:
            return {self}
        else:
            return set(
                itertools.chain(
                    (Product(lp, Omega()) for lp in self.left.organized),
                    (Product(Omega(), rp) for rp in self.right.organized),
                )
            )

    def _str_prec(self, prec: int) -> str:
        product_prec: int = 9

        def product_str_prec(other: Type) -> str:
            match other:
                case Product(_, _):
                    return other._str_prec(product_prec)
                case _:
                    return other._str_prec(product_prec + 1)

        result: str = (
            f"{product_str_prec(self.left)} * {self.right._str_prec(product_prec + 1)}"
        )
        return Type._parens(result) if prec > product_prec else result


@dataclass(frozen=True)
class Arrow(Type):
    source: Type = field(init=True)
    target: Type = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return self.target.is_omega

    def _size(self) -> int:
        return 1 + self.source.size + self.target.size

    def _organized(self) -> set[Type]:
        if len(self.target.organized) == 0:
            return set()
        elif len(self.target.organized) == 1:
            return {self}
        else:
            return {Arrow(self.source, tp) for tp in self.target.organized}

    def _str_prec(self, prec: int) -> str:
        arrow_prec: int = 8
        result: str
        match self.target:
            case Arrow(_, _):
                result = (
                    f"{self.source._str_prec(arrow_prec + 1)} -> "
                    f"{self.target._str_prec(arrow_prec)}"
                )
            case _:
                result = (
                    f"{self.source._str_prec(arrow_prec + 1)} -> "
                    f"{self.target._str_prec(arrow_prec + 1)}"
                )
        return Type._parens(result) if prec > arrow_prec else result


@dataclass(frozen=True)
class Intersection(Type):
    left: Type = field(init=True)
    right: Type = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return self.left.is_omega and self.right.is_omega

    def _size(self) -> int:
        return 1 + self.left.size + self.right.size

    def _organized(self) -> set[Type]:
        return set.union(self.left.organized, self.right.organized)

    def _str_prec(self, prec: int) -> str:
        intersection_prec: int = 10

        def intersection_str_prec(other: Type) -> str:
            match other:
                case Intersection(_, _):
                    return other._str_prec(intersection_prec)
                case _:
                    return other._str_prec(intersection_prec + 1)

        result: str = (
            f"{intersection_str_prec(self.left)} & {intersection_str_prec(self.right)}"
        )
        return Type._parens(result) if prec > intersection_prec else result
