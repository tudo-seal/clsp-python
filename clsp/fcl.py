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
from itertools import chain, filterfalse
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Optional

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


@dataclass
class Instantiation:
    substituted_term_predicates: list[Predicate]
    vars: list[Literal | GVar]  # TODO
    substitution: dict[str, Literal]


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
            tuple[ParamInfo, list[Instantiation], list[list[MultiArrow]]],
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
        list[Instantiation],
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
        parameters = ParamInfo([], [], {})
        for param in params:
            if isinstance(param, LitParamSpec):
                parameters.literal_params.append(param)
                parameters.lvar_to_group[param.name] = param.group
            else:
                parameters.term_params.append(param)

        instantiations = FiniteCombinatoryLogic._instantiate(literals, parameters)

        current: list[MultiArrow] = [MultiArrow([], ty)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]
        return (parameters, instantiations, multiarrows)

    @staticmethod
    def _add_set_to(
        name: str,
        set_to_pred: SetTo,
        substitutions: deque[dict[str, Literal]],
        group: str,
        literals: Mapping[str, list[Any]],
    ) -> Iterable[dict[str, Literal]]:
        for s in substitutions:
            value = set_to_pred.compute(s)
            if set_to_pred.override or value in literals[group]:
                yield s | {name: Literal(value, group)}

    @staticmethod
    def _instantiate(
        literals: Mapping[str, list[Any]],
        parameters: ParamInfo,
        initial_substitution: Optional[dict[str, Literal]] = None,
    ) -> list[Instantiation]:
        if initial_substitution is None:
            initial_substitution = {}
        substitutions: deque[dict[str, Literal]] = deque([initial_substitution])
        args: deque[str | GVar] = deque()

        for literal_parameter in parameters.literal_params:
            if literal_parameter.group not in literals:
                return []
            else:
                args.append(literal_parameter.name)
                normal_preds = list(
                    filterfalse(
                        lambda pred: isinstance(pred, SetTo),
                        literal_parameter.predicate,
                    )
                )

                setto = False
                # Only the last setto takes effect
                for pred in literal_parameter.predicate:
                    if isinstance(pred, SetTo):
                        setto = True
                        substitutions = deque(
                            filter(
                                lambda substs: all(
                                    callable(npred) and npred(substs)
                                    for npred in normal_preds
                                ),
                                FiniteCombinatoryLogic._add_set_to(
                                    literal_parameter.name,
                                    pred,
                                    substitutions,
                                    literal_parameter.group,
                                    literals,
                                ),
                            )
                        )

                # If we do not have at least one "setto", we need to enumerate all possible
                # literals
                if not setto:
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
                                for literal in literals[literal_parameter.group]
                            ),
                        )
                    )

        for term_parameter in parameters.term_params:
            args.append(GVar(term_parameter.name))

        instantiations: list[Instantiation] = []
        for substitution in substitutions:
            predicates: list[Predicate] = []
            for term_param in parameters.term_params:
                predicates.extend(
                    Predicate(pred, predicate_substs=substitution)
                    for pred in term_param.predicate
                )
            instantiated_combinator_args: list[Literal | GVar] = [
                substitution[arg] if not isinstance(arg, GVar) else arg for arg in args
            ]
            instantiations.append(
                Instantiation(
                    predicates,
                    instantiated_combinator_args,
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

    def _has_subsitution(
        self,
        paths: list[Type],
        combinator_type: list[list[MultiArrow]],
        groups: dict[str, str],
    ) -> dict[str, Literal] | None | Ambiguous:
        """
        Simply checks, if there is a substitution covering all paths.

        Such coarse, much overapproximation
        """
        all_substitutions = []
        for path in paths:
            # Check all targets of the multiarrows
            candidates = [ty for nary_types in combinator_type for ty in nary_types]
            substitutions = list(
                filter(
                    lambda x: x is not None,
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

            # If the substitution is Ambiguous -> Don't bother
            if isinstance(substitutions[0], Ambiguous):
                return Ambiguous()

            all_substitutions.extend(substitutions)

        # Check substitutions for consistency.
        # If two substitutions substitute the same variable by diffent values => Impossible
        # If two substitutions substitute diffente variables => Union
        return_subsitution = {}
        for substitution in all_substitutions:
            for k, v in substitution.items():
                if k in return_subsitution:
                    if v != return_subsitution[k]:
                        return None
                else:
                    return_subsitution[k] = v

        # for all remaining literal variables, we still need to create all possible substitutions

        # print(all_substitutions)
        # print(
        #     f"{[str(p) for p in paths]} can be covered by {[str(m) for t in combinator_type for m in t]}"
        # )

        return return_subsitution

    def get_rules(
        self, combinator, term_params, instantiation, nary_types, paths
    ) -> tuple[deque[Type], list[RHSRule]]:
        new_type_targets = deque()
        rules = []
        arguments: list[list[Type]] = list(
            self._subqueries(nary_types, paths, instantiation.substitution)
        )

        if len(arguments) == 0:
            return (new_type_targets, [])

        specific_params = {
            param.name: param.group.subst(instantiation.substitution)
            for param in term_params
        }

        new_type_targets.extend(specific_params.values())

        for subquery in (
            [ty.subst(instantiation.substitution) for ty in query]
            for query in arguments
        ):
            rules.append(
                RHSRule(
                    specific_params,
                    instantiation.substituted_term_predicates,
                    combinator,
                    instantiation.vars,
                    subquery,
                ),
            )

            new_type_targets.extendleft(subquery)

        return (new_type_targets, rules)

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

                # try each combinator and arity
                for combinator, (
                    parameters,
                    instantiations,
                    combinator_type,
                ) in self.repository.items():
                    # Simply reject impossible candidates
                    # 28.3

                    substitution = self._has_subsitution(
                        paths, combinator_type, parameters.lvar_to_group
                    )

                    if substitution is None:
                        continue

                    if not isinstance(substitution, Ambiguous):
                        # only take substitutions, that are compatible with the inferred

                        instantiations = filter(
                            lambda instantiation: all(
                                instantiation.substitution[var] == subst
                                for var, subst in substitution.items()
                            ),
                            instantiations,
                        )

                        # Fill up other variables:
                        # substitution

                    for instantiation in instantiations:
                        for nary_types in combinator_type:
                            new_targets, rules = self.get_rules(
                                combinator,
                                parameters.term_params,
                                instantiation,
                                nary_types,
                                paths,
                            )
                            type_targets.extend(new_targets)
                            for rule in rules:
                                memo.add_rule(current_target, rule)

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
