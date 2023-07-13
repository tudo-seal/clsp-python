# Propositional Finite Combinatory Logic

from __future__ import annotations
from collections import deque
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Optional, reveal_type
from uuid import uuid4

from cls.grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import (
    Arrow,
    Intersection,
    Literal,
    Param,
    LitParamSpec,
    ParamSpec,
    TermParamSpec,
    Type,
)

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")

# ([theta_1, ..., theta_m], [sigma_1, ..., sigma_n], tau) means theta_1 => ... => theta_m => sigma_1 -> ... -> sigma_n -> tau


@dataclass(frozen=True)
class MultiArrow(Generic[T]):
    # lit_params: list[LitParamSpec[T]]
    # term_params: list[TermParamSpec[T]]
    args: list[Type[T]]
    target: Type[T]

    def subst(self, substitution: dict[str, Literal]) -> MultiArrow[T]:
        return MultiArrow(
            [arg.subst(substitution) for arg in self.args],
            self.target.subst(substitution),
        )


InstantiationMeta: TypeAlias = tuple[
    list[TermParamSpec[T]],
    list[Predicate],
    list[Literal | GVar],
    dict[str, Literal],
]

TreeGrammar: TypeAlias = MutableMapping[Type[T], deque[tuple[C, list[Type[T]]]]]


def show_grammar(grammar: TreeGrammar[T, C]) -> Iterable[str]:
    for clause, possibilities in grammar.items():
        lhs = str(clause)
        yield (
            lhs
            + " => "
            + "; ".join(
                (str(combinator) + "(" + ", ".join(map(str, args)) + ")")
                for combinator, args in possibilities
            )
        )


def mstr(m: MultiArrow[T]) -> tuple[str, str]:
    return str(list(map(str, m.args))), str(m.target)


class FiniteCombinatoryLogic(Generic[T, C]):
    def __init__(
        self,
        repository: Mapping[C, Param[T] | Type[T]],
        subtypes: Subtypes[T] = Subtypes({}),
        literals: Optional[dict[Any, list[Any]]] = None,
    ):
        self.literals: dict[Any, list[Any]] = {} if literals is None else literals
        self.repository: Mapping[
            C,
            tuple[Sequence[InstantiationMeta[T]], list[list[MultiArrow[T]]]],
        ] = {
            c: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(
        p_or_ty: Param[T] | Type[T], literals: dict[Any, list[Any]]
    ) -> tuple[Sequence[InstantiationMeta[T]], list[list[MultiArrow[T]]],]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type[T]) -> Iterable[tuple[Type[T], Type[T]]]:
            tys: deque[Type[T]] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        def split_params(
            ty: Param[T] | Type[T],
        ) -> tuple[list[ParamSpec[T]], Type[T]]:
            params: list[ParamSpec[T]] = []
            while isinstance(ty, Param):
                if isinstance(ty.type, Type):
                    params.append(TermParamSpec(ty.name, ty.type, ty.predicate))
                else:
                    params.append(LitParamSpec(ty.name, ty.type, ty.predicate))
                ty = ty.inner
            return (params, ty)

        params, ty = split_params(p_or_ty)
        instantiations = list(FiniteCombinatoryLogic._instantiate(literals, params))
        current: list[MultiArrow[T]] = [MultiArrow([], ty)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]
        return (instantiations, multiarrows)

    def _subqueries(
        self, nary_types: list[MultiArrow[T]], paths: list[Type[T]]
    ) -> Sequence[list[Type[T]]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow[T], Type[T]], bool
        ] = lambda m, t: self.subtypes.check_subtype(m.target, t)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[
            [Iterable[Type[T]], Iterable[Type[T]]], list[Type[T]]
        ] = lambda args1, args2: [Intersection(a, b) for a, b in zip(args1, args2)]

        intersected_args = (
            list(reduce(intersect_args, (m.args for m in ms))) for ms in covers
        )
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(self.subtypes.check_subtype, args1, args2)
        )
        return maximal_elements(intersected_args, compare_args)

    @staticmethod
    def _instantiate(
        literals: dict[Any, list[Any]],
        params: list[LitParamSpec[T] | TermParamSpec[T]],
    ) -> Iterator[InstantiationMeta[T]]:
        substitutions: Sequence[dict[str, Any]] = [{}]
        args: list[str | GVar] = []
        term_params: list[TermParamSpec[T]] = []

        for param in params:
            if isinstance(param, LitParamSpec):
                if param.type not in literals:
                    return []
                else:
                    args.append(param.name)
                    substitutions = list(
                        filter(
                            lambda substs: param.predicate(substs),
                            (
                                s | {param.name: Literal(literal, param.type)}
                                for s in substitutions
                                for literal in literals[param.type]
                            ),
                        )
                    )
            elif isinstance(param, TermParamSpec):
                args.append(GVar(param.name))
                # binders[param.name] = param.type
                term_params.append(param)

        if len(substitutions) == 0:
            substitutions = [{}]

        for substitution in substitutions:
            predicates = []
            for term_param in term_params:
                predicates.append(
                    Predicate(term_param.predicate, predicate_substs=substitution)
                )
            instantiated_combinator_args = [
                substitution[arg] if not isinstance(arg, GVar) else arg for arg in args
            ]
            yield (
                term_params,
                predicates,
                instantiated_combinator_args,
                substitution,
            )

    def inhabit(
        self, *targets: Type[T]
    ) -> ParameterizedTreeGrammar[Type[T], C | Literal]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: ParameterizedTreeGrammar[
            Type[T], C | Literal
        ] = ParameterizedTreeGrammar()
        for lit_ty, literals in self.literals.items():
            for lit in literals:
                memo.add_rule(Literal(lit, lit_ty), RHSRule({}, [], lit, []))

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                possibilities: deque[RHSRule[Type[T], C | Literal]] = deque()
                memo.update({current_target: possibilities})
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type[T]] = list(current_target.organized)

                # try each combinator and arity
                for combinator, (meta, combinator_type) in self.repository.items():
                    for params, predicates, args, substitutions in meta:
                        for p_nary_types in combinator_type:
                            nary_types = [
                                ty.subst(substitutions) for ty in p_nary_types
                            ]
                            arguments: list[list[Type[T]]] = list(
                                self._subqueries(nary_types, paths)
                            )
                            if len(arguments) == 0:
                                continue

                            for subquery in arguments:
                                unique_var_names: list[str] = [
                                    str(uuid4()) for _ in subquery
                                ]
                                possibilities.append(
                                    RHSRule(
                                        {
                                            unique_var_names[i]: subquery[i]
                                            for i in range(len(subquery))
                                        }
                                        | {param.name: param.type for param in params},
                                        predicates,
                                        combinator,
                                        args
                                        + [
                                            GVar(unique_var_names[i])
                                            for i in range(len(subquery))
                                        ],
                                    )
                                )
                                type_targets.extendleft(subquery)

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: ParameterizedTreeGrammar[Type[T], C]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(
            binder: dict[str, Type[T]],
            args: Sequence[Literal | GVar],
            ground_types: set[Type[T]],
        ) -> bool:
            return all(
                True
                for arg in args
                if isinstance(arg, Literal) or binder[arg.name] in ground_types
            )

        ground_types: set[Type[T]] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True
                for rule in memo[ty]
                if is_ground(rule.binder, rule.args, ground_types)
            ),
            memo.nonterminals(),
        )
        # initialize inhabited (ground) types
        while new_ground_types:
            ground_types.update(new_ground_types)
            new_ground_types, candidates = partition(
                lambda ty: any(
                    True
                    for rule in memo[ty]
                    if is_ground(rule.binder, rule.args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.as_tuples():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(possibility.binder, possibility.args, ground_types)
            )
