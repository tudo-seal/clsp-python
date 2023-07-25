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
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Optional

from cls.grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule

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


class LiteralNotFoundException(Exception):
    pass


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
            tuple[list[InstantiationMeta[T]], list[list[MultiArrow[T]]]],
        ] = {
            c: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(
        p_or_ty: Param[T] | Type[T], literals: dict[Any, list[Any]]
    ) -> tuple[list[InstantiationMeta[T]], list[list[MultiArrow[T]]],]:
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
        ) -> tuple[list[TermParamSpec[T] | LitParamSpec[T]], Type[T]]:
            params: list[TermParamSpec[T] | LitParamSpec[T]] = []
            while isinstance(ty, Param):
                if isinstance(ty.type, Type):
                    params.append(ty.get_term_spec())
                else:
                    params.append(ty.get_lit_spec())
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
        self,
        nary_types: list[MultiArrow[T]],
        paths: list[Type[T]],
        substitutions: dict[str, Literal],
    ) -> Sequence[list[Type[T]]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow[T], Type[T]], bool
        ] = lambda m, t: self.subtypes.check_subtype(m.target, t, substitutions)
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
            map(
                lambda a, b: self.subtypes.check_subtype(a, b, substitutions),
                args1,
                args2,
            )
        )
        return maximal_elements(intersected_args, compare_args)

    @staticmethod
    def _instantiate(
        literals: dict[Any, list[Any]],
        params: list[LitParamSpec[T] | TermParamSpec[T]],
    ) -> Iterator[InstantiationMeta[T]]:
        substitutions: Sequence[dict[str, Literal]] = deque([{}])
        set_tos: list[tuple[str, Any, SetTo]] = []
        args: deque[str | GVar] = deque()
        term_params: list[TermParamSpec[T]] = []

        for param in params:
            if isinstance(param, LitParamSpec):
                if param.type not in literals:
                    return []
                else:
                    args.append(param.name)
                    if isinstance(param.predicate, SetTo):
                        flag_for_deletion = []
                        for i, substitution in enumerate(substitutions):
                            value = param.predicate.compute(substitution)
                            if value not in literals[param.type]:
                                flag_for_deletion.append(i)
                            substitution[param.name] = Literal(value, param.type)

                        for invalid_substitution in flag_for_deletion:
                            del substitutions[invalid_substitution]
                    else:
                        substitutions = deque(
                            filter(
                                lambda substs: callable(param.predicate)
                                and param.predicate(substs),
                                (
                                    s | {param.name: Literal(literal, param.type)}
                                    for s in substitutions
                                    for literal in literals[param.type]
                                ),
                            )
                        )
            elif isinstance(param, TermParamSpec):
                args.append(GVar(param.name))
                term_params.append(param)

        for substitution in substitutions:
            # try:
            #     for name, ty, set_to in set_tos:
            #         value = set_to.compute(substitution)
            #         if value not in literals[ty]:
            #             raise LiteralNotFoundException()
            #         substitution[name] = Literal(value, ty)
            #         print(substitution)
            # except LiteralNotFoundException:
            #     continue
            #
            # predicates = (
            #     Predicate(term_param.predicate, predicate_substs=substitution)
            #     for term_param in term_params
            # )
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

    def inhabit(self, *targets: Type[T]) -> ParameterizedTreeGrammar[Type[T], C]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: ParameterizedTreeGrammar[Type[T], C] = ParameterizedTreeGrammar()

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                possibilities: deque[RHSRule[Type[T], C]] = deque()
                memo.update({current_target: possibilities})
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type[T]] = list(current_target.organized)

                # try each combinator and arity
                for combinator, (meta, combinator_type) in self.repository.items():
                    for params, predicates, args, substitutions in meta:
                        for nary_types in combinator_type:
                            arguments: list[list[Type[T]]] = list(
                                self._subqueries(nary_types, paths, substitutions)
                            )
                            if len(arguments) == 0:
                                continue

                            specific_params = {
                                param.name: param.type.subst(substitutions)
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
    def _prune(memo: ParameterizedTreeGrammar[Type[T], C]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(
            binder: dict[str, Type[T]],
            parameters: Sequence[Literal | GVar],
            args: Sequence[Type[T]],
            ground_types: set[Type[T]],
        ) -> bool:
            return all(
                True
                for parameter in parameters
                if isinstance(parameter, Literal)
                or binder[parameter.name] in ground_types
            ) and all(True for arg in args if arg in ground_types)

        ground_types: set[Type[T]] = set()
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
