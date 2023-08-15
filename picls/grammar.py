from __future__ import annotations
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Optional

from .types import Literal

NT = TypeVar("NT")
T = TypeVar("T")
T2 = TypeVar("T2")


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
    _evaluated: bool = field(default=False)
    _value: bool = field(default=False)

    def eval(self, assign: dict[str, Any]) -> bool:
        if not self._evaluated:
            self._value = self.predicate(self.predicate_substs | assign)
            # TODO: the predicate needs to be reevaluated for different assignments
            # self._evaluated = True

        return self._value

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
    parameters: list[Literal | GVar]
    args: list[NT]

    def __str__(self) -> str:
        forallstrings = "".join(
            [f"∀({name}:{ty})." for name, ty in self.binder.items()]
        )
        predicatestrings = "".join([str(predicate) for predicate in self.predicates])
        paramstring = "".join([f"({str(param)})" for param in self.parameters])
        argstring = "".join([f"({str(arg)})" for arg in self.args])
        return f"{forallstrings}{predicatestrings}{str(self.terminal)}{paramstring}{argstring}"

    def map_terminal(self, f: Callable[[T], T2]) -> RHSRule[NT, T2]:
        return RHSRule(
            self.binder, self.predicates, f(self.terminal), self.parameters, self.args
        )


@dataclass()
class ParameterizedTreeGrammar(Generic[NT, T]):
    _rules: dict[NT, deque[RHSRule[NT, T]]] = field(default_factory=dict)

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

    def map_over_terminals(
        self, f: Callable[[T], T2]
    ) -> ParameterizedTreeGrammar[NT, T2]:
        return ParameterizedTreeGrammar(
            {
                nt: deque(map(lambda rule: rule.map_terminal(f), rules))
                for nt, rules in self._rules.items()
            }
        )

    def show(self) -> str:
        return "\n".join(
            f"{str(nt)} ~> {' | '.join([str(subrule) for subrule in rule])}"
            for nt, rule in self._rules.items()
        )
