# Propositional Finite Combinatory Logic

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial, reduce
from itertools import chain
from typing import Any, Callable, Sequence, TypeAlias, TypeVar

from .boolean import BooleanTerm, minimal_dnf_as_list
from .combinatorics import maximal_elements, minimal_covers, partition
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

    def _subqueries(
        self, nary_types: list[MultiArrow], paths: list[Type]
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow, Type], bool
        ] = lambda m, t: self.subtypes.check_subtype(m[1], t)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
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
        clause_targets: deque[Clause] = deque()
        boolean_terms = {}
        for target in targets:
            if isinstance(target, BooleanTerm):
                boolean_terms[target] = self.boolean_to_clauses(target)
                clause_targets.extend(boolean_terms[target])
            else:
                clause_targets.append(target)

        # dictionary of type |-> sequence of combinatory expressions
        memo: TreeGrammar = dict()

        while clause_targets:
            target = clause_targets.pop()
            if memo.get(target) is None:
                # target type was not seen before
                # paths: list[Type] = list(target.organized)
                possibilities: deque[tuple[object, list[Clause]]] = deque()
                memo.update({target: possibilities})
                # If the positive part is omega, then the result is junk
                if target[0].is_omega:
                    continue
                # If the positive part is a subtype of the negative part, then there are no inhabitants
                if any(True for ty in target[1] if self.subtypes.check_subtype(target[0], ty)):
                    continue
                
                all_positive_paths: list[Type] = list(target[0].organized)
                all_negative_paths = [list(ty.organized) for ty in target[1]]

                # try each combinator and arity
                for combinator, combinator_type in self.repository.items():
                    for nary_types in combinator_type:
                        positive_arguments: list[list[Type]] = list(
                            self._subqueries(nary_types, all_positive_paths)
                        )
                        if len(positive_arguments) == 0:
                            continue
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
                            clause_targets.extendleft(subquery)

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
