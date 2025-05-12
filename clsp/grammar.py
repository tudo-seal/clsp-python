from __future__ import annotations
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Optional
from itertools import chain

from .types import Literal

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


@dataclass(frozen=True)
class RHSRule(Generic[NT, T]):
    binder: dict[str, NT]
    predicates: list[Predicate]

    terminal: T
    parameters: list[Literal | GVar]
    variable_names: list[str]
    args: list[NT]

    def __len__(self) -> int:
        return len(self.parameters) + len(self.args)

    def all_args(self) -> Iterable[NT | Literal]:
        for p in self.parameters:
            if isinstance(p, Literal):
                yield p
            else:
                yield self.binder[p.name]
        yield from self.args

    def check(self, parameters: Iterable[Any]) -> bool:
        """Test if all predicates of a rule are satisfied by the parameters."""
        substitution = {
            param.name: subterm
            for subterm, param in zip(parameters, self.parameters)
            if isinstance(param, GVar)
        }
        return all(predicate.eval(substitution) for predicate in self.predicates)

    def non_terminals(self) -> frozenset[NT]:
        """Set of non-terminals occurring in the body of the rule."""
        return frozenset(chain(self.binder.values(), self.args))

    def __str__(self) -> str:
        forallstrings = "".join([f"∀({name}:{ty})." for name, ty in self.binder.items()])
        predicatestrings = "".join([str(predicate) for predicate in self.predicates])
        paramstring = "".join([f"({str(param)})" for param in self.parameters])
        argstring = "".join([f"({str(arg)})" for arg in self.args])
        return f"{forallstrings}{predicatestrings}{str(self.terminal)}{paramstring}{argstring}"


@dataclass
class ParameterizedTreeGrammar(Generic[NT, T]):
    _rules: dict[NT, deque[RHSRule[NT, T]]] = field(default_factory=lambda: defaultdict(deque))

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
        self._rules[nonterminal].append(rule)

    def show(self) -> str:
        return "\n".join(
            f"{str(nt)} ~> {' | '.join([str(subrule) for subrule in rule])}"
            for nt, rule in self._rules.items()
        )

    def annotations(self) -> tuple[tuple[tuple[tuple[NT, RHSRule[NT, T]], int],...], tuple[tuple[NT, int],...]]:
        """
        Following the grammar based initialization method (GBIM) for context free grammars,
        we annotate terminals, nonterminals and rules with the expected minimum depth of generated terms.
        In contrast to context free grammars, these depths are overapproximation, because we can not include the
        evaluation of predicates in the computation of the expected depth.
        But even this lower bounds of an expected depth should be a good enough approximation to compute an
        initial population of terms with a suitable distribution of term depths.

        The length of a terminal symbol is always 0, therefore we don't need to return annotations for terminals.
        """

        annotated: tuple[tuple[tuple[NT, RHSRule[NT, T]], int],...] = tuple()

        not_annotated: list[tuple[NT, RHSRule[NT, T]]] = [
            (nt, rhs)
            for nt, rules in self._rules.items()
            for rhs in rules
        ]

        symbol_depths: tuple[tuple[NT, int],...] = tuple()

        nts: list[NT] = []

        # every rule that only derives nonterminals has length 1
        for nt, rhs in not_annotated:
            if not list(rhs.non_terminals()):
                # rule only derives terminals
                for ((l, r), _) in annotated:
                    if l == nt and r == rhs:
                        # rule already annotated
                        new = list(annotated)
                        new.remove(((l, r), _))
                        annotated = tuple(new)
                annotated = annotated + (((nt, rhs), 1),)
                not_annotated.remove((nt, rhs))
                nts.append(nt)

        for nt in nts:
            # if a right hand side has the minmal length 1, the symbol also has this length
            symbol_depths = symbol_depths + ((nt, 1),)

        while not_annotated:
            termination_check = not_annotated.copy()
            for nt, rhs in not_annotated:
                # check if all nonterminals in rhs are already annotated
                if all(s in map(lambda t: t[0], symbol_depths) for s in rhs.non_terminals()):
                    # the length of a rule is the maximum of its nonterminal lenghts + 1
                    for ((l, r), _) in annotated:
                        if l == nt and r == rhs:
                            # rule already annotated
                            new = list(annotated)
                            new.remove(((l, r), _))
                            annotated = tuple(new)
                    new_depth = max(r for t in rhs.non_terminals() for (l, r) in symbol_depths if l == t) + 1
                    annotated = annotated + (((nt, rhs), new_depth),)
                    not_annotated.remove((nt, rhs))
                    # The first time we derive a length for a right hand side, we can assume, that it is the minimum length and therefore set the symbol depth
                    for (l, r) in symbol_depths:
                        if l == nt and new_depth < r:
                            # symbol already annotated
                            new = list(symbol_depths)
                            new.remove((l, r))
                            symbol_depths = tuple(new)
                    symbol_depths = symbol_depths + ((nt, new_depth),)
                    if all((nt, rule) in map(lambda t: t[0], annotated) for rule in self._rules[nt]):
                        # all rules of this nonterminal are already annotated
                        # the length of a terminal symbol is the minimum of the length of its rules
                        for (l, r) in symbol_depths:
                            if l == nt:
                                # rule already annotated
                                new = list(symbol_depths)
                                new.remove((l, r))
                                symbol_depths = tuple(new)
                        symbol_depths = symbol_depths + ((nt, min(n for (s, _), n in annotated if s == nt)),)
            if termination_check == not_annotated:
                # no more rules can be annotated
                break

        # if there are still rules that are not annotated, we have a problem with the grammar
        if not_annotated:
            raise ValueError(
                f"Grammar contains problems. The following rules could not be annotated: {not_annotated} \n These rules have been annotated: {annotated} \n These symbols have been annotated: {symbol_depths}"
            )

        return annotated, symbol_depths

    def minimum_tree_depth(self, start: NT) -> int:
        """
        Compute a lower bound for the minimum depth of a tree generated by the grammar from the given nonterminal.
        """
        _, nt_length = self.annotations()
        for nt, length in nt_length:
            if nt == start:
                return length
        return -1
