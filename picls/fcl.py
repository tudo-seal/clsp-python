# Propositional Finite Combinatory Logic

from __future__ import annotations
from collections import deque
from collections.abc import (
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
from functools import reduce
from itertools import compress
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Optional

from .grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import (
    Arrow,
    Intersection,
    Literal,
    Param,
    LitParamSpec,
    SetTo,
    TermParamSpec,
    Type,
)

C = TypeVar("C")


@dataclass(frozen=True)
class MultiArrow:
    args: list[Type]
    target: Type

    def subst(self, substitution: dict[str, Literal]) -> MultiArrow:
        return MultiArrow(
            [arg.subst(substitution) for arg in self.args],
            self.target.subst(substitution),
        )


InstantiationMeta: TypeAlias = tuple[
    list[TermParamSpec],
    list[Predicate],
    list[Literal | GVar],
    dict[str, Literal],
]


class FiniteCombinatoryLogic(Generic[C]):
    def __init__(
        self,
        repository: Mapping[C, Param | Type],
        subtypes: Subtypes = Subtypes({}),
        literals: Optional[Mapping[str, list[Any]]] = None,
    ):
        self.literals: Mapping[str, list[Any]] = {} if literals is None else literals
        self.repository: Mapping[
            C,
            tuple[list[InstantiationMeta], list[list[MultiArrow]]],
        ] = {
            c: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(
        p_or_ty: Param | Type, literals: Mapping[str, list[Any]]
    ) -> tuple[list[InstantiationMeta], list[list[MultiArrow]],]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type) -> Iterable[tuple[Type, Type]]:
            tys: deque[Type] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        def split_params(
            ty: Param | Type,
        ) -> tuple[list[TermParamSpec | LitParamSpec], Type]:
            params: list[TermParamSpec | LitParamSpec] = []
            while isinstance(ty, Param):
                params.append(ty.get_spec())
                ty = ty.inner
            return (params, ty)

        params, ty = split_params(p_or_ty)
        instantiations = list(FiniteCombinatoryLogic._instantiate(literals, params))
        current: list[MultiArrow] = [MultiArrow([], ty)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]
        return (instantiations, multiarrows)

    @staticmethod
    def _instantiate(
        literals: Mapping[str, list[Any]],
        params: Sequence[LitParamSpec | TermParamSpec],
    ) -> Iterable[InstantiationMeta]:
        substitutions: deque[dict[str, Literal]] = deque([{}])
        args: deque[str | GVar] = deque()
        term_params: list[TermParamSpec] = []

        for param in params:
            if isinstance(param, LitParamSpec):
                if param.group not in literals:
                    return []
                else:
                    args.append(param.name)
                    if isinstance(param.predicate, SetTo):
                        filter_list = []
                        for substitution in substitutions:
                            value = param.predicate.compute(substitution)
                            filter_list.append(value in literals[param.group])
                            substitution[param.name] = Literal(value, param.group)

                        substitutions = deque(compress(substitutions, filter_list))
                    else:
                        substitutions = deque(
                            filter(
                                lambda substs: callable(param.predicate)
                                and param.predicate(substs),
                                (
                                    s | {param.name: Literal(literal, param.group)}
                                    for s in substitutions
                                    for literal in literals[param.group]
                                ),
                            )
                        )
            elif isinstance(param, TermParamSpec):
                args.append(GVar(param.name))
                term_params.append(param)

        for substitution in substitutions:
            predicates = []
            for term_param in term_params:
                predicates.append(
                    Predicate(term_param.predicate, predicate_substs=substitution)
                )
            instantiated_combinator_args: list[Literal | GVar] = [
                substitution[arg] if not isinstance(arg, GVar) else arg for arg in args
            ]
            yield (
                term_params,
                predicates,
                instantiated_combinator_args,
                substitution,
            )

    def _subqueries(
        self,
        nary_types: list[MultiArrow],
        paths: list[Type],
        substitutions: dict[str, Literal],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow, Type], bool
        ] = lambda m, t: self.subtypes.check_subtype(m.target, t, substitutions)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[
            [Iterable[Type], Iterable[Type]], list[Type]
        ] = lambda args1, args2: [Intersection(a, b) for a, b in zip(args1, args2)]

        intersected_args = (
            list(reduce(intersect_args, (m.args for m in ms))) for ms in covers
        )
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(
                lambda a, b: self.subtypes.check_subtype(a, b, substitutions),
                args1,
                args2,
            )
        )
        return maximal_elements(intersected_args, compare_args)

    def inhabit(self, *targets: Type) -> ParameterizedTreeGrammar[Type, C]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: ParameterizedTreeGrammar[Type, C] = ParameterizedTreeGrammar()

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                possibilities: deque[RHSRule[Type, C]] = deque()
                if isinstance(current_target, Literal):
                    continue
                memo.update({current_target: possibilities})
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type] = list(current_target.organized)

                # try each combinator and arity
                for combinator, (meta, combinator_type) in self.repository.items():
                    for params, predicates, args, substitutions in meta:
                        for nary_types in combinator_type:
                            arguments: list[list[Type]] = list(
                                self._subqueries(nary_types, paths, substitutions)
                            )
                            if len(arguments) == 0:
                                continue

                            specific_params = {
                                param.name: param.group.subst(substitutions)
                                for param in params
                            }

                            type_targets.extend(specific_params.values())

                            for subquery in (
                                [ty.subst(substitutions) for ty in query]
                                for query in arguments
                            ):
                                possibilities.append(
                                    RHSRule(
                                        specific_params,
                                        predicates,
                                        combinator,
                                        args,
                                        subquery,
                                    )
                                )
                                type_targets.extendleft(subquery)

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: ParameterizedTreeGrammar[Type, C]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(
            binder: dict[str, Type],
            parameters: Sequence[Literal | GVar],
            args: Sequence[Type],
            ground_types: set[Type],
        ) -> bool:
            return all(
                True
                for parameter in parameters
                if isinstance(parameter, Literal)
                or binder[parameter.name] in ground_types
            ) and all(True for arg in args if arg in ground_types)

        ground_types: set[Type] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True
                for rule in memo[ty]
                if is_ground(rule.binder, rule.parameters, rule.args, ground_types)
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
                    if is_ground(rule.binder, rule.parameters, rule.args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.as_tuples():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(
                    possibility.binder,
                    possibility.parameters,
                    possibility.args,
                    ground_types,
                )
            )
