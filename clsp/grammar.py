from __future__ import annotations
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence, Mapping
from queue import PriorityQueue
from dataclasses import dataclass, field
from itertools import product
from typing import Generic, TypeVar, Any, Optional
from .tree import Tree

NT = TypeVar("NT") # type of non-terminals
T = TypeVar("T") # type of terminals

@dataclass(frozen=True)
class TerminalArgument(Generic[T]):
    name: str
    value: T
    
@dataclass(frozen=True)
class NonTerminalArgument(Generic[NT]):
    value: NT

@dataclass(frozen=True)
class NamedNonTerminalArgument(NonTerminalArgument[NT]):
    name: str

Argument = TerminalArgument | NonTerminalArgument[NT]

@dataclass(frozen=True)
class RHSRule(Generic[NT, T]):
    arguments: tuple[Argument, ...]
    predicates: tuple[Callable[[dict[str, Any]], bool], ...]
    terminal: T

    @property
    def non_terminals(self) -> frozenset[NT]:
        """Set of non-terminals occurring in the body of the rule."""
        return frozenset(arg.value for arg in self.arguments if isinstance(arg, NonTerminalArgument))

    @property
    def argument_names(self) -> tuple[str | None, ...]:
        """Names of named arguments."""
        return tuple(a.name if isinstance(a, NamedNonTerminalArgument) or isinstance(a, TerminalArgument) else None for a in self.arguments)

    @property
    def literal_substitution(self):
        return {n.name: n.value for n in self.arguments if isinstance(n, TerminalArgument)}


@dataclass
class Grammar(Generic[NT, T]):
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

    def prune(self) -> Grammar[NT, T]:
        """Keep only productive rules."""

        ground_types: set[NT] = set()
        queue: set[NT] = set()
        inverse_grammar: dict[NT, set[tuple[NT, frozenset[NT]]]] = defaultdict(set)

        for n, exprs in self._rules.items():
            for expr in exprs:
                non_terminals = expr.non_terminals
                for m in non_terminals:
                    inverse_grammar[m].add((n, non_terminals))
                if not non_terminals:
                    queue.add(n)

        while queue:
            n = queue.pop()
            if n not in ground_types:
                ground_types.add(n)
                for m, non_terminals in inverse_grammar[n]:
                    if m not in ground_types and all(t in ground_types for t in non_terminals):
                        queue.add(m)

        result = Grammar[NT, T]({
                target: deque(
                    possibility
                    for possibility in self._rules[target]
                    if all(t in ground_types for t in possibility.non_terminals)
                )
                for target in ground_types
            })
        return result
    
    def _enumerate_tree_vectors(
        self,
        non_terminals: Sequence[NT | None],
        existing_terms: Mapping[NT, set[Tree[T]]],
        nt_term: tuple[NT, Tree[T]] | None = None,
    ) -> Iterable[tuple[Tree[T] | None, ...]]:
        """Enumerate possible term vectors for a given list of non-terminals and existing terms. Use nt_term at least once (if given)."""
        if nt_term is None:
            yield from product(*([n] if n is None else existing_terms[n] for n in non_terminals))
        else:
            nt, term = nt_term
            for i, n in enumerate(non_terminals):
                if n == nt:
                    arg_lists: Iterable[Iterable[Tree[T] | None]] = ([None] if m is None else [term] if i == j else existing_terms[m] for j, m in enumerate(non_terminals))
                    yield from product(*arg_lists)

    def _generate_new_trees(
        self,
        rule: RHSRule[NT, T],
        existing_terms: Mapping[NT, set[Tree[T]]],
        max_count: Optional[int] = None,
        nt_old_term: Optional[tuple[NT, Tree[T]]] = None,
    ) -> set[Tree[T]]:
        # Genererate new terms for rule `rule` from existing terms up to `max_count`
        # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

        output_set: set[Tree[T]] = set()
        if max_count == 0:
            return output_set
        
        named_non_terminals = [a.value if isinstance(a, NamedNonTerminalArgument) else None for a in rule.arguments]
        unnamed_non_terminals = [a.value if isinstance(a, NonTerminalArgument) and not isinstance(a, NamedNonTerminalArgument) else None for a in rule.arguments]
        literal_arguments = [Tree(a.value) if isinstance(a, TerminalArgument) else None for a in rule.arguments]

        def interleave(
            parameters: Sequence[Tree[T] | None],
            literal_arguments: Sequence[Tree[T] | None],
            arguments: Sequence[Tree[T] | None],
        ) -> Iterable[Tree[T]]:
            """Interleave parameters, literal arguments and arguments."""
            for parameter, literal_argument, argument in zip(parameters, literal_arguments, arguments):
                if parameter is not None:
                    yield parameter
                elif literal_argument is not None:
                    yield literal_argument
                elif argument is not None:
                    yield argument
                else:
                    raise ValueError("All arguments of interleave are None")

        specific_substitution = lambda parameters: {a.name: p for p, a in zip(parameters, rule.arguments) if isinstance(a, NamedNonTerminalArgument)} | rule.literal_substitution
        if nt_old_term is None:
            for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms):
                substitution = specific_substitution(parameters)
                if all(predicate(substitution) for predicate in rule.predicates):
                    for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                        output_set.add(Tree(
                            rule.terminal,
                            tuple(interleave(parameters, literal_arguments, arguments)),
                            child_names=rule.argument_names,
                        ))
                        if max_count is not None and len(output_set) >= max_count:
                            return output_set
        else:
            for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms, nt_old_term):
                substitution = specific_substitution(parameters)
                if all(predicate(substitution) for predicate in rule.predicates):
                    for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                        output_set.add(Tree(
                            rule.terminal,
                            tuple(interleave(parameters, literal_arguments, arguments)),
                            child_names=rule.argument_names,
                        ))
                        if max_count is not None and len(output_set) >= max_count:
                            return output_set
            all_parameters: deque[tuple[Tree[T] | None, ...]] | None = None
            for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                if all_parameters is None:
                    all_parameters = deque()
                    for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms):
                        substitution = specific_substitution(parameters)
                        if all(predicate(substitution) for predicate in rule.predicates):
                            all_parameters.append(parameters)
                for parameters in all_parameters:
                    output_set.add(Tree(
                        rule.terminal,
                        tuple(interleave(parameters, literal_arguments, arguments)),
                        child_names=rule.argument_names,
                    ))
                    if max_count is not None and len(output_set) >= max_count:
                        return output_set
        return output_set

    def enumerate_trees(
        self,
        start: NT,
        max_count: Optional[int] = None,
        max_bucket_size: Optional[int] = None,
    ) -> Iterable[Tree[T]]:
        """
        Enumerate terms as an iterator efficiently - all terms are enumerated, no guaranteed term order.
        """
        if start not in self.nonterminals():
            return

        queues: dict[NT, PriorityQueue[Tree[T]]] = {n: PriorityQueue() for n in self.nonterminals()}
        existing_terms: dict[NT, set[Tree[T]]] = {n: set() for n in self.nonterminals()}
        inverse_grammar: dict[NT, deque[tuple[NT, RHSRule[NT, T]]]] = {
            n: deque() for n in self.nonterminals()
        }
        all_results: set[Tree[T]] = set()

        for n, exprs in self._rules.items():
            for expr in exprs:
                for m in expr.non_terminals:
                    inverse_grammar[m].append((n, expr))
                for new_term in self._generate_new_trees(expr, existing_terms):
                    queues[n].put(new_term)
                    if n == start and new_term not in all_results:
                        if max_count is not None and len(all_results) >= max_count:
                            return
                        yield new_term
                        all_results.add(new_term)

        current_bucket_size = 1

        while (max_bucket_size is None or current_bucket_size <= max_bucket_size) and any(
            not queue.empty() for queue in queues.values()
        ):
            non_terminals = {n for n in self.nonterminals() if not queues[n].empty()}

            while non_terminals:
                n = non_terminals.pop()
                results = existing_terms[n]
                while len(results) < current_bucket_size and not queues[n].empty():
                    term = queues[n].get()
                    if term in results:
                        continue
                    results.add(term)
                    for m, expr in inverse_grammar[n]:
                        if len(existing_terms[m]) < current_bucket_size:
                            non_terminals.add(m)
                        if m == start:
                            for new_term in self._generate_new_trees(
                                expr, existing_terms, max_count, (n, term)
                            ):
                                if new_term not in all_results:
                                    if max_count is not None and len(all_results) >= max_count:
                                        return
                                    yield new_term
                                    all_results.add(new_term)
                                    queues[start].put(new_term)
                        else:
                            for new_term in self._generate_new_trees(
                                expr, existing_terms, max_bucket_size, (n, term)
                            ):
                                queues[m].put(new_term)
            current_bucket_size += 1
        return
