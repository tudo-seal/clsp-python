from __future__ import annotations
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Optional

from cls.types import Literal

NT = TypeVar("NT")
T = TypeVar("T")


# @dataclass
# class Binder(Generic[NT]):
#     name: str
#     nonterminal: NT
#
#     def __str__(self) -> str:
#         return f"∀({self.name}: {self.nonterminal})."


@dataclass
class Predicate:
    predicate: Callable[[dict[str, Any]], bool]
    name: str = field(default="P")
    predicate_substs: dict[str, Any] = field(default_factory=dict)

    def eval(self, assign: dict[str, Any]) -> bool:
        return self.predicate(self.predicate_substs | assign)

    def __str__(self) -> str:
        return f"{self.name} ⇛ "


@dataclass(frozen=True)
class GVar:
    name: str

    def __str__(self) -> str:
        return f"<{self.name}>"


@dataclass
class RHSRule(Generic[NT, T]):
    binder: dict[str, NT]
    predicates: list[Predicate]
    terminal: T
    args: list[Literal | GVar]

    def __str__(self) -> str:
        forallstrings = "".join(
            [f"∀({name}:{ty})." for name, ty in self.binder.items()]
        )
        predicatestrings = "".join([str(predicate) for predicate in self.predicates])
        argstring = "".join([f"({str(arg)})" for arg in self.args])
        return f"{forallstrings}{predicatestrings}{str(self.terminal)}{argstring}"


class ParameterizedTreeGrammar(Generic[NT, T]):
    _rules: dict[NT, deque[RHSRule[NT, T]]] = {}

    def get(self, nonterminal: NT) -> Optional[deque[RHSRule[NT, T]]]:
        return self._rules.get(nonterminal)

    def update(self, param: dict[NT, deque[RHSRule[NT, T]]]) -> None:
        self._rules.update(param)

    def __getitem__(self, nonterminal: NT) -> deque[RHSRule[NT, T]]:
        return self._rules[nonterminal]

    def nonterminals(self) -> Iterable[NT]:
        return self._rules.keys()

    def as_tuples(self) -> Iterable[tuple[NT, deque[RHSRule[NT, T]]]]:
        return self._rules.items()

    def __setitem__(self, nonterminal: NT, rhs: deque[RHSRule[NT, T]]) -> None:
        self._rules[nonterminal] = rhs

    def add_rule(self, nonterminal: NT, rule: RHSRule[NT, T]) -> None:
        if nonterminal in self._rules:
            self._rules[nonterminal].append(rule)
        else:
            self._rules[nonterminal] = deque([rule])

    def show(self) -> str:
        return "\n".join(
            f"{str(nt)} ~> {' | '.join([str(subrule) for subrule in rule])}"
            for nt, rule in self._rules.items()
        )


def bind_single(x: NT) -> RHSRule[NT, Any]:
    return RHSRule({"x": x}, [], "x", [])
