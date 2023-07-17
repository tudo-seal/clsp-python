from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class Type(ABC, Generic[T]):
    is_omega: bool = field(init=True, kw_only=True, compare=False)
    size: int = field(init=True, kw_only=True, compare=False)
    organized: set[Type[T]] = field(init=True, kw_only=True, compare=False)

    def __str__(self) -> str:
        return self._str_prec(0)

    def __mul__(self, other: Type[T]) -> Type[T]:
        return Product(self, other)

    @abstractmethod
    def _organized(self) -> set[Type[T]]:
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

    @abstractmethod
    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        pass

    @staticmethod
    def _parens(s: str) -> str:
        return f"({s})"

    @staticmethod
    def intersect(types: Sequence[Type[T]]) -> Type[T]:
        if len(types) > 0:
            rtypes = reversed(types)
            result: Type[T] = next(rtypes)
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

    def __pow__(self, other: Type[T]) -> Type[T]:
        return Arrow(self, other)

    def __mul__(self, other: Type[T]) -> Type[T]:
        return Product(self, other)

    def __and__(self, other: Type[T]) -> Type[T]:
        return Intersection(self, other)

    def __rmatmul__(self, name: T) -> Type[T]:
        return Constructor(name, self)


@dataclass(frozen=True)
class Omega(Type[T]):
    is_omega: bool = field(init=False, compare=False)
    size: bool = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

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

    def _organized(self) -> set[Type[T]]:
        return set()

    def _str_prec(self, prec: int) -> str:
        return "omega"

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        return self


@dataclass(frozen=True)
class Constructor(Type[T]):
    name: T = field(init=True)
    arg: Type[T] = field(default=Omega(), init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

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

    def _organized(self) -> set[Type[T]]:
        if len(self.arg.organized) <= 1:
            return {self}
        else:
            return {Constructor(self.name, ap) for ap in self.arg.organized}

    def _str_prec(self, prec: int) -> str:
        if self.arg == Omega():
            return str(self.name)
        else:
            return f"{str(self.name)}({str(self.arg)})"

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        return Constructor(self.name, self.arg.subst(substitution))


@dataclass(frozen=True)
class Product(Type[T]):
    left: Type[T] = field(init=True)
    right: Type[T] = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

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

    def _organized(self) -> set[Type[T]]:
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

        def product_str_prec(other: Type[T]) -> str:
            match other:
                case Product(_, _):
                    return other._str_prec(product_prec)
                case _:
                    return other._str_prec(product_prec + 1)

        result: str = (
            f"{product_str_prec(self.left)} * {self.right._str_prec(product_prec + 1)}"
        )
        return Type[T]._parens(result) if prec > product_prec else result

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        return Product(self.left.subst(substitution), self.right.subst(substitution))


@dataclass(frozen=True)
class Arrow(Type[T]):
    source: Type[T] = field(init=True)
    target: Type[T] = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

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

    def _organized(self) -> set[Type[T]]:
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

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        return Arrow(self.source.subst(substitution), self.target.subst(substitution))


@dataclass(frozen=True)
class Intersection(Type[T]):
    left: Type[T] = field(init=True)
    right: Type[T] = field(init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

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

    def _organized(self) -> set[Type[T]]:
        return set.union(self.left.organized, self.right.organized)

    def _str_prec(self, prec: int) -> str:
        intersection_prec: int = 10

        def intersection_str_prec(other: Type[T]) -> str:
            match other:
                case Intersection(_, _):
                    return other._str_prec(intersection_prec)
                case _:
                    return other._str_prec(intersection_prec + 1)

        result: str = (
            f"{intersection_str_prec(self.left)} & {intersection_str_prec(self.right)}"
        )
        return Type._parens(result) if prec > intersection_prec else result

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        return Intersection(
            self.left.subst(substitution), self.right.subst(substitution)
        )


@dataclass(frozen=True)
class Literal(Type[Any]):
    value: Any
    type: Any
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[Any]] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type[Any]]:
        return {self}

    def _str_prec(self, prec: int) -> str:
        return f"{str(self.value)}@({str(self.type)})"

    def subst(self, substitution: dict[str, Literal]) -> Type[Any]:
        return self


@dataclass(frozen=True)
class TVar(Type[T]):
    name: str
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type[T]] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type[T]]:
        return {self}

    def _str_prec(self, prec: int) -> str:
        return f"<{str(self.name)}>"

    def subst(self, substitution: dict[str, Literal]) -> Type[T]:
        if self.name in substitution:
            return substitution[self.name]
        else:
            return self


@dataclass
class ParamSpec(Generic[T]):
    pass


# @dataclass(frozen=True)
@dataclass
class LitParamSpec(ParamSpec[T]):
    name: str
    type: Any
    predicate: Callable[[dict[str, Any]], bool] | SetTo


@dataclass
class TermParamSpec(ParamSpec[T]):
    name: str
    type: Type[T]
    predicate: Callable[[dict[str, Any]], bool]


@dataclass
class SetTo:
    compute: Callable[[dict[str, Any]], Any]


# @dataclass(frozen=True)
@dataclass
class Param(Generic[T]):
    name: str
    type: Type[T] | Any
    predicate: Callable[[dict[str, Any]], bool] | SetTo
    inner: Param[T] | Type[T]

    def get_lit_spec(self) -> LitParamSpec[T]:
        return LitParamSpec(self.name, self.type, self.predicate)

    def get_term_spec(self) -> TermParamSpec[T]:
        if isinstance(self.predicate, SetTo):
            raise RuntimeError("Term parameters cannot have SetTo")
        return TermParamSpec(self.name, self.type, self.predicate)

    def __str__(self) -> str:
        return f"<{self.name}, {self.type}, {self.predicate}>.{self.inner}"
