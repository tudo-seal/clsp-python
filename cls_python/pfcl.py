# Propositional Finite Combinatory Logic

from abc import ABC
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial, reduce
from itertools import chain
from typing import Any, Callable, Sequence, TypeAlias, TypeVar

from .boolean import BooleanTerm, minimal_dnf_as_list
from .combinatorics import maximal_elements, minimal_covers, partition
from .enumeration import ComputationStep, EmptyStep
from .subtypes import Subtypes
from .types import Arrow, Constructor, Intersection, Type

# ([sigma_1, ..., sigma_n], tau) means sigma_1 -> ... -> sigma_n -> tau
MultiArrow: TypeAlias = tuple[list[Type], Type]
# (tau_0, tau_1, ..., tau_n) means tau_0 and (not tau_1) and ... and (not tau_n)
Clause: TypeAlias = tuple[Type, frozenset[Type]]

TreeGrammar: TypeAlias = dict[
    Clause | BooleanTerm[Type], deque[tuple[object, list[Clause]]]
]


def show_clause(clause: Clause) -> str:
    flat_clause: Iterable[Type] = chain([clause[0]], clause[1])
    return " and not ".join(map(str, flat_clause))


def show_grammar(grammar: TreeGrammar) -> Iterable[str]:
    for clause, possibilities in grammar.items():
        lhs = str(clause) if isinstance(clause, BooleanTerm) else show_clause(clause)
        yield (
            lhs
            + " => "
            + "; ".join(
                (str(combinator) + "(" + ", ".join(map(show_clause, args)) + ")")
                for combinator, args in possibilities
            )
        )


def mstr(m: MultiArrow) -> tuple[str, str]:
    return (str(list(map(str, m[0]))), str(m[1]))


@dataclass(frozen=True)
class Rule(ABC):
    target: Type = field(init=True, kw_only=True)
    is_combinator: bool = field(init=True, kw_only=True)


@dataclass(frozen=True)
class Failed(Rule):
    target: Type = field()
    is_combinator: bool = field(default=False, init=False)

    def __str__(self) -> str:
        return f"Failed({str(self.target)})"


@dataclass(frozen=True)
class Combinator(Rule):
    target: Type = field()
    is_combinator: bool = field(default=True, init=False)
    combinator: object = field(init=True)

    def __str__(self) -> str:
        return f"Combinator({str(self.target)}, {str(self.combinator)})"


@dataclass(frozen=True)
class Apply(Rule):
    target: Type = field()
    is_combinator: bool = field(default=False, init=False)
    function_type: Type = field(init=True)
    argument_type: Type = field(init=True)

    def __str__(self) -> str:
        return (
            f"@({str(self.function_type)}, {str(self.argument_type)}) : {self.target}"
        )


@dataclass(frozen=True)
class Tree(object):
    rule: Rule = field(init=True)
    children: tuple["Tree", ...] = field(init=True, default_factory=lambda: ())

    class Evaluator(ComputationStep):
        def __init__(self, outer: "Tree", results: list[Any]):
            self.outer: "Tree" = outer
            self.results = results

        def __iter__(self) -> Iterator[ComputationStep]:
            match self.outer.rule:
                case Combinator(_, c):
                    self.results.append(c)
                case Apply(_, _, _):
                    f_arg: list[Any] = []
                    yield Tree.Evaluator(self.outer.children[0], f_arg)
                    yield Tree.Evaluator(self.outer.children[1], f_arg)
                    self.results.append(partial(f_arg[0])(f_arg[1]))
                case _:
                    raise TypeError(f"Cannot apply rule: {self.outer.rule}")
            yield EmptyStep()

    def evaluate(self) -> Any:
        result: list[Any] = []
        self.Evaluator(self, result).run()
        return result[0]

    def __str__(self) -> str:
        match self.rule:
            case Combinator(_, _):
                return str(self.rule.combinator)
            case Apply(_, _, _):
                return f"{str(self.children[0])}({str(self.children[1])})"
            case _:
                return f"{str(self.rule)} @ ({', '.join(map(str, self.children))})"

            # case Combinator(_, _): return str(self.rule)
            # case Apply(_, _, _): return f"{str(self.children[0])}({str(self.children[1])})"
            # case _: return f"{str(self.rule)} @ ({', '.join(map(str, self.children))})"


class FiniteCombinatoryLogic(object):
    def __init__(self, repository: dict[object, Type], subtypes: Subtypes):
        self.repository: dict[object, list[list[MultiArrow]]] = {
            c: list(FiniteCombinatoryLogic._function_types(ty))
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(ty: Type) -> Iterable[list[MultiArrow]]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type) -> Iterable[tuple[Type, Type]]:
            tys: deque[Type] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        current: list[MultiArrow] = [([], ty)]
        while len(current) != 0:
            yield current
            current = [
                (args + [new_arg], new_tgt)
                for (args, tgt) in current
                for (new_arg, new_tgt) in unary_function_types(tgt)
            ]

    def _omega_rules(self, target: Type) -> set[Rule]:
        return {
            Apply(target, target, target),
            *map(lambda c: Combinator(target, c), self.repository.keys()),
        }

    @staticmethod
    def _combinatory_expression_rules(
        combinator: object, arguments: list[Clause], target: Type
    ) -> Iterable[Rule]:
        """Rules from combinatory expression `combinator(arguments[0], ..., arguments[n])`."""

        remaining_arguments: deque[Clause] = deque(arguments)
        while remaining_arguments:
            argument = FiniteCombinatoryLogic.clause_to_type(remaining_arguments.pop())
            yield Apply(target, Arrow(argument, target), argument)
            target = Arrow(argument, target)
        yield Combinator(target, combinator)

    def _subqueries(
        self, nary_types: list[MultiArrow], paths: list[Type]
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow, Type], bool
        ] = lambda m, t: self.subtypes.check_subtype(m[1], t)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[
            [Iterable[Type], Iterable[Type]], Any  # TODO: Fix types
        ] = lambda args1, args2: map(Intersection, args1, args2)
        intersected_args = (
            list(reduce(intersect_args, (m[0] for m in ms))) for ms in covers
        )
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(self.subtypes.check_subtype, args1, args2)
        )
        return maximal_elements(intersected_args, compare_args)

    @staticmethod
    def list_of_types_to_clause(types: Iterable[Type]) -> Clause:
        """Given a list of types, where the first element represents a positive type, and the
        remaining elements represent negative types, create a Clause"""

        list_representation = list(types)

        return (list_representation[0], frozenset(list_representation[1:]))

    def _combine_arguments(
        self, positive_arguments: list[list[Type]], negative_arguments: list[list[Type]]
    ) -> list[list[Clause]]:
        result: deque[list[deque[Type]]] = deque()
        for pos in positive_arguments:
            result.append(list(map(lambda ty: deque((ty,)), pos)))
        for neg in negative_arguments:
            new_result: deque[list[deque[Type]]] = deque()
            for i in range(len(neg)):
                for args in result:
                    new_args = args.copy()
                    new_args[i] = new_args[i].copy()
                    new_args[i].append(neg[i])
                    new_result.append(new_args)
            result = new_result
        return list(
            list(map(FiniteCombinatoryLogic.list_of_types_to_clause, args))
            for args in result
        )

    @staticmethod
    def clause_to_type(clause: Clause | BooleanTerm[Type]) -> Type:
        return Constructor(name=clause)

    def boolean_to_clauses(self, target: BooleanTerm[Type]) -> list[Clause]:
        dnf = minimal_dnf_as_list(target)

        clauses: list[Clause] = []

        for encoded_clause in dnf:
            encoded_negatives, encoded_positives = partition(
                lambda lit: lit[0], encoded_clause
            )
            positives = [lit[1] for lit in encoded_positives]
            negatives = [lit[1] for lit in encoded_negatives]

            positive_intersection = reduce(Intersection, positives)

            clauses.append((positive_intersection, frozenset(negatives)))

        return clauses

    def inhabit(self, *targets: Clause | BooleanTerm[Type]) -> TreeGrammar:
        clause_targets = []
        boolean_terms = {}
        for target in targets:
            if isinstance(target, BooleanTerm):
                boolean_terms[target] = self.boolean_to_clauses(target)
                clause_targets.extend(boolean_terms[target])
            else:
                clause_targets.append(target)

        # dictionary of type |-> sequence of combinatory expressions
        memo: TreeGrammar = dict()

        remaining_targets: deque[Clause] = deque(clause_targets)

        while remaining_targets:
            target = remaining_targets.pop()
            if memo.get(target) is None:
                # target type was not seen before
                # paths: list[Type] = list(target.organized)
                possibilities: deque[tuple[object, list[Clause]]] = deque()
                memo.update({target: possibilities})

                # If the positive part is omega, skip this iteration, since this would inhabit
                # mostly "Junk"
                if target[0].is_omega:
                    continue

                all_positive_paths: list[Type] = list(target[0].organized)
                all_negative_paths = [list(ty.organized) for ty in target[1]]

                # try each combinator and arity
                for combinator, combinator_type in self.repository.items():
                    for nary_types in combinator_type:
                        positive_arguments: list[list[Type]] = list(
                            self._subqueries(nary_types, all_positive_paths)
                        )
                        negative_arguments: list[list[Type]] = list(
                            chain.from_iterable(
                                self._subqueries(nary_types, paths)
                                for paths in all_negative_paths
                            )
                        )
                        for subquery in self._combine_arguments(
                            positive_arguments, negative_arguments
                        ):
                            possibilities.append((combinator, subquery))
                            remaining_targets.extendleft(subquery)

        # generate rules for the boolean_terms
        for term, clauses in boolean_terms.items():
            rhs_of_clauses: deque[tuple[object, list[Clause]]] = deque(
                (rhs for clause in clauses for rhs in memo[clause])
            )
            memo[term] = rhs_of_clauses

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: TreeGrammar) -> None:
        """Keep only productive grammar rules."""

        def is_ground(
            args: list[Clause], ground_types: set[Clause | BooleanTerm[Type]]
        ) -> bool:
            return all(True for arg in args if arg in ground_types)

        ground_types: set[Clause | BooleanTerm[Type]] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True for (_, args) in memo[ty] if is_ground(args, ground_types)
            ),
            memo.keys(),
        )
        # initialize inhabited (ground) types
        while new_ground_types:
            ground_types.update(new_ground_types)
            new_ground_types, candidates = partition(
                lambda ty: any(
                    True for _, args in memo[ty] if is_ground(args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.items():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(possibility[1], ground_types)
            )
