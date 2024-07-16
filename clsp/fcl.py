# Propositional Finite Combinatory Logic

from __future__ import annotations
from collections import deque
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import (
    Any,
    Callable,
    Generic,
    MutableMapping,
    TypeGuard,
    TypeVar,
    Optional,
    cast,
)

from .grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Ambiguous, Subtypes
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

    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"{[str(a) for a in self.args]} -> {str(self.target)}"
        else:
            return str(self.target)


# InstantiationMeta: TypeAlias = tuple[
#     list[TermParamSpec],
#     list[Predicate],
#     list[Literal | GVar],
#     dict[str, Literal],
# ]


@dataclass
class ParamInfo:
    literal_params: list[LitParamSpec]
    term_params: list[TermParamSpec]
    lvar_to_group: dict[str, str]
    params_order: list[LitParamSpec | TermParamSpec]


@dataclass
class Instantiation:
    substituted_term_predicates: list[Predicate]
    substitution: dict[str, Literal]


class FiniteCombinatoryLogic(Generic[C]):
    def __init__(
        self,
        repository: Mapping[C, Param | Type],
        subtypes: Subtypes = Subtypes({}),
        literals: Optional[Mapping[str, list[Any]]] = None,
    ):
        self.literals: Mapping[str, list[Any]] = {} if literals is None else literals
        self.repository: MutableMapping[
            C,
            tuple[ParamInfo, Optional[list[Instantiation]], list[list[MultiArrow]]],
        ] = {
            c: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(
        p_or_ty: Param | Type, literals: Mapping[str, list[Any]]
    ) -> tuple[
        ParamInfo,
        None,
        list[list[MultiArrow]],
    ]:
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
        parameters = ParamInfo([], [], {}, params)
        for param in params:
            if isinstance(param, LitParamSpec):
                parameters.literal_params.append(param)
                parameters.lvar_to_group[param.name] = param.group
            else:
                parameters.term_params.append(param)

        # instantiations = FiniteCombinatoryLogic._instantiate(literals, parameters)

        current: list[MultiArrow] = [MultiArrow([], ty)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]
        return (parameters, None, multiarrows)

    @staticmethod
    def _add_set_to(
        name: str,
        set_to_preds: list[SetTo],
        substitutions: deque[dict[str, Literal]],
        group: str,
        literals: Mapping[str, list[Any]],
        initial_value: Optional[Literal] = None,
    ) -> Iterable[dict[str, Literal]]:
        for s in substitutions:
            values = {pred.compute(s) for pred in set_to_preds}
            if len(values) != 1:
                continue
            value = tuple(values)[0]

            # if initial_value is not None and value != initial_value.value:
            #     continue

            if any(pred.override for pred in set_to_preds) or value in literals[group]:
                yield s | {name: Literal(value, group)}
            # value = (
            #     {
            #         result
            #         for pred in set_to_preds
            #         if (result := pred.compute(s)) in literals[group] or pred.override
            #     }
            #     | {initial_value.value}
            #     if initial_value is not None
            #     else {}
            # )
            # if len(value) != 1:
            #     continue
            #     # if initial_value is not None and value != initial_value.value:
            #     #     continue
            #     # if set_to_pred.override or value in literals[group]:
            # yield s | {name: Literal(tuple(value)[0], group)}

    @staticmethod
    def _instantiate(
        literals: Mapping[str, list[Any]],
        parameters: ParamInfo,
        initial_substitution: Optional[dict[str, Literal]] = None,
    ) -> list[Instantiation]:
        if initial_substitution is None:
            initial_substitution = {}
        substitutions: deque[dict[str, Literal]] = deque([{}])

        for literal_parameter in parameters.literal_params:
            if literal_parameter.group not in literals:
                return []
            else:
                normal_preds, set_to_preds = partition(
                    lambda pred: isinstance(pred, SetTo), literal_parameter.predicate
                )

                if literal_parameter.name in initial_substitution:
                    set_to_preds.append(
                        SetTo(
                            lambda _: initial_substitution[literal_parameter.name].value
                        )
                    )

                if len(set_to_preds) > 0:
                    substitutions = deque(
                        filter(
                            lambda substs: all(
                                callable(npred) and npred(substs)
                                for npred in normal_preds
                            ),
                            FiniteCombinatoryLogic._add_set_to(
                                literal_parameter.name,
                                cast(list[SetTo], set_to_preds),
                                substitutions,
                                literal_parameter.group,
                                literals,
                            ),
                        )
                    )
                else:
                    substitutions = deque(
                        filter(
                            lambda substs: all(
                                callable(npred) and npred(substs)
                                for npred in normal_preds
                            ),
                            (
                                s
                                | {
                                    literal_parameter.name: Literal(
                                        literal, literal_parameter.group
                                    )
                                }
                                for s in substitutions
                                for literal in (literals[literal_parameter.group])
                            ),
                        )
                    )

        instantiations: list[Instantiation] = []
        for substitution in substitutions:
            predicates: list[Predicate] = []
            for term_param in parameters.term_params:
                predicates.extend(
                    Predicate(pred, predicate_substs=substitution)
                    for pred in term_param.predicate
                )
            instantiations.append(
                Instantiation(
                    predicates,
                    substitution,
                )
            )
        return instantiations

    def _subqueries(
        self,
        nary_types: list[MultiArrow],
        paths: list[Type],
        substitutions: dict[str, Literal],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[[MultiArrow, Type], bool] = (
            lambda m, t: self.subtypes.check_subtype(m.target, t, substitutions)
        )
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[[Iterable[Type], Iterable[Type]], list[Type]] = (
            lambda args1, args2: [Intersection(a, b) for a, b in zip(args1, args2)]
        )

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

    def _forced_substitutions(
        self,
        paths: list[Type],
        combinator_type: list[list[MultiArrow]],
        groups: dict[str, str],
    ) -> dict[str, Literal] | None | Ambiguous:
        """
        Computes a substitution that needs to be part of every substitution S such that
        S(combinator_type) <: S(paths).

        If no substitution can make this valid, None is returned.

        If there is no unique smallest substitution, Ambiguous() is returned.
        """

        def is_substitution(
            infered_result: Mapping[str, Literal] | None | Ambiguous,
        ) -> TypeGuard[dict[str, Literal] | Ambiguous]:
            return infered_result is not None

        all_substitutions: list[dict[str, Literal]] = []
        for path in paths:
            # Check all targets of the multiarrows
            candidates = [ty for nary_types in combinator_type for ty in nary_types]

            substitutions: list[dict[str, Literal] | Ambiguous] = list(
                filter(
                    is_substitution,
                    (
                        self.subtypes.infer_substitution(ty.target, path, groups)
                        for ty in candidates
                    ),
                )
            )
            # if no substitution is applicable, this path cannot be covered
            if len(substitutions) == 0:
                return None

            # TODO this can be done better
            # if a path can be covered by multiple targets: Don't bother
            if len(substitutions) > 1:
                return Ambiguous()

            substitution = substitutions[0]

            # If the substitution is Ambiguous -> Don't bother
            if isinstance(substitution, Ambiguous):
                return Ambiguous()

            all_substitutions.append(substitution)

        # Check substitutions for consistency.
        # If two substitutions substitute the same variable by diffent values => Impossible
        # If two substitutions substitute diffent variables => Union
        return_subsitution: dict[str, Literal] = {}
        for substitution in all_substitutions:
            for k, v in substitution.items():
                if k in return_subsitution:
                    if v != return_subsitution[k]:
                        return None
                else:
                    return_subsitution[k] = v

        return return_subsitution

    def inhabit(self, *targets: Type) -> ParameterizedTreeGrammar[Type, C]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: ParameterizedTreeGrammar[Type, C] = ParameterizedTreeGrammar()

        seen: set[Type] = set()

        while type_targets:
            current_target = type_targets.pop()

            # target type was not seen before
            if current_target not in seen:
                seen.add(current_target)
                # TODO is this correct? consider current_target = Intersection(Literal("3", "int"), Literal("3", "int"))
                if isinstance(current_target, Literal):
                    continue
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type] = list(current_target.organized)

                # try each combinator
                for combinator, (
                    parameters,
                    instantiations,
                    combinator_type,
                ) in self.repository.items():
                    # Compute if there are forced substitutions
                    substitution = self._forced_substitutions(
                        paths, combinator_type, parameters.lvar_to_group
                    )

                    # If no substitution can lead to a match, ignore this combinator
                    if substitution is None:
                        continue

                    # If there is at most one possible substitution for each variable
                    # only consider these substitutions
                    if not isinstance(substitution, Ambiguous):
                        # Keep the forced substitutions and enumerate the rest
                        instantiations = self._instantiate(
                            self.literals, parameters, substitution
                        )
                    else:
                        # otherwise enumerate the whole substitution space.
                        # but do this only the first time... this is time consuming
                        # Update the repository with the enumerated substitutions.
                        if instantiations is None:
                            instantiations = self._instantiate(
                                self.literals, parameters
                            )
                            self.repository[combinator] = (
                                parameters,
                                instantiations,
                                combinator_type,
                            )

                    # regardless of how the substitutions were constructed, carry on
                    # with inhabitation. Consider all possible substitutions,
                    for instantiation in instantiations:
                        # and every arity of the combinator type
                        for nary_types in combinator_type:
                            parameter_arguments: list[Literal | GVar] = [
                                (
                                    instantiation.substitution[arg.name]
                                    if not isinstance(arg, TermParamSpec)
                                    else GVar(arg.name)
                                )
                                for arg in parameters.params_order
                            ]

                            arguments: list[list[Type]] = list(
                                self._subqueries(
                                    nary_types, paths, instantiation.substitution
                                )
                            )

                            if len(arguments) == 0:
                                continue

                            specific_params = {
                                param.name: param.group.subst(
                                    instantiation.substitution
                                )
                                for param in parameters.term_params
                            }

                            type_targets.extend(specific_params.values())

                            for subquery in (
                                [ty.subst(instantiation.substitution) for ty in query]
                                for query in arguments
                            ):
                                memo.add_rule(
                                    current_target,
                                    RHSRule(
                                        specific_params,
                                        instantiation.substituted_term_predicates,
                                        combinator,
                                        parameter_arguments,
                                        subquery,
                                    ),
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
                isinstance(parameter, Literal) or binder[parameter.name] in ground_types
                for parameter in parameters
            ) and all(isinstance(arg, Literal) or arg in ground_types for arg in args)

        ground_types: set[Type] = set()
        candidates, new_ground_types = partition(
            lambda ty: any(
                is_ground(rule.binder, rule.parameters, rule.args, ground_types)
                for rule in memo[ty]
            ),
            memo.nonterminals(),
        )
        while new_ground_types:
            ground_types.update(new_ground_types)
            candidates, new_ground_types = partition(
                lambda ty: any(
                    is_ground(rule.binder, rule.parameters, rule.args, ground_types)
                    for rule in memo[ty]
                ),
                candidates,
            )

        non_ground_types = set(memo.nonterminals()).difference(ground_types)
        for target in non_ground_types:
            del memo._rules[target]

        for target in ground_types:
            memo[target] = deque(
                possibility
                for possibility in memo[target]
                if all(
                    ty in ground_types
                    for ty in chain(possibility.binder.values(), possibility.args)
                )
            )
