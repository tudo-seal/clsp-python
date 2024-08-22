# Propositional Finite Combinatory Logic

from __future__ import annotations
from collections import deque, defaultdict
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Generic,
    MutableMapping,
    TypeVar,
    Optional,
    cast,
)

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

    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"{[str(a) for a in self.args]} -> {str(self.target)}"
        else:
            return str(self.target)


@dataclass
class ParamInfo:
    literal_params: list[LitParamSpec]
    term_params: list[TermParamSpec]
    lvar_to_group: dict[str, str]
    params_selector: list[bool]
    variable_names: list[str]
    cache: bool
    infer: bool

    @property
    def params_order(self) -> Iterable[LitParamSpec | TermParamSpec]:
        lit_iter = self.literal_params.__iter__()
        term_iter = self.literal_params.__iter__()
        for selector in self.params_selector:
            if selector:
                yield lit_iter.__next__()
            else:
                yield term_iter.__next__()

    def intantiate(self, substitution: dict[str, Literal]) -> Iterable[Literal | GVar]:
        lit_iter = self.literal_params.__iter__()
        term_iter = self.term_params.__iter__()
        for i, selector in enumerate(self.params_selector):
            param: TermParamSpec | LitParamSpec
            if selector:
                param = lit_iter.__next__()
                yield substitution[param.name]
            else:
                param = term_iter.__next__()
                yield GVar(param.name)


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
            tuple[ParamInfo, Optional[Iterable[Instantiation]], list[list[MultiArrow]]],
        ] = {
            c: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _function_types(p_or_ty: Param | Type, literals: Mapping[str, list[Any]]) -> tuple[
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

        parameters = ParamInfo([], [], {}, [], [], cache=False, infer=True)
        for param in params:
            parameters.variable_names.append(param.name)
            if isinstance(param, LitParamSpec):
                parameters.literal_params.append(param)
                parameters.lvar_to_group[param.name] = param.group
                parameters.params_selector.append(True)
            else:
                parameters.params_selector.append(False)
                parameters.term_params.append(param)

        parameters.cache = all(lparam.cache for lparam in parameters.literal_params)
        parameters.infer = all(lparam.infer for lparam in parameters.literal_params)

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
    ) -> Iterable[dict[str, Literal]]:
        for s in substitutions:
            values = {pred.compute(s) for pred in set_to_preds}
            if len(values) != 1:
                continue
            value = tuple(values)[0]

            if any(pred.override for pred in set_to_preds) or value in literals[group]:
                yield s | {name: Literal(value, group)}

    @staticmethod
    def _instantiate(
        literals: Mapping[str, list[Any]],
        parameters: ParamInfo,
        initial_substitution: Optional[dict[str, Literal]] = None,
        prior_instantiations: Optional[list[Instantiation]] = None,
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
                        SetTo(lambda _: initial_substitution[literal_parameter.name].value)
                    )

                if len(set_to_preds) > 0:
                    substitutions = deque(
                        filter(
                            lambda substs: all(
                                callable(npred) and npred(substs) for npred in normal_preds
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
                                callable(npred) and npred(substs) for npred in normal_preds
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
                    Predicate(pred, predicate_substs=substitution) for pred in term_param.predicate
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

        intersected_args = (list(reduce(intersect_args, (m.args for m in ms))) for ms in covers)
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
    ) -> dict[str, Literal] | None:
        """
        Computes a substitution that needs to be part of every substitution S such that
        S(combinator_type) <= paths.

        If no substitution can make this valid, None is returned.

        If there is no unique smallest substitution, Ambiguous() is returned.
        """

        return_subsitution: dict[str, Literal] = {}

        all_substitutions = []
        for path in paths:
            # Check all targets of the multiarrows
            candidates = [ty for nary_types in combinator_type for ty in nary_types]

            multiarrows_and_substitutions = [
                substitution
                for ty in candidates
                if (substitution := self.subtypes.infer_substitution(ty.target, path, groups))
                is not None
            ]

            # if no substitution is applicable, this path cannot be covered
            if len(multiarrows_and_substitutions) == 0:
                return None

            # TODO this can be done better
            # if a path can be covered by multiple targets: Don't bother
            if len(multiarrows_and_substitutions) > 1:
                return {}

            substitution = multiarrows_and_substitutions[0]

            all_substitutions.append(substitution)
            # for k, v in substitution.items():
            #     if k in return_subsitution:
            #         if v != return_subsitution[k]:
            #             del return_subsitution[k]
            #         else:
            #             return_subsitution[k] = v

        # Check substitutions for consistency.
        # If two substitutions substitute the same variable by diffent values => Intersection
        # If two substitutions substitute diffent variables => Union
        for substitution in all_substitutions:
            for k, v in substitution.items():
                if k in return_subsitution:
                    if v != return_subsitution[k]:
                        del return_subsitution[k]
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
                    selected_instantiations: Optional[Iterable[Instantiation]] = instantiations

                    if parameters.infer and parameters.literal_params:
                        # Compute if there are forced substitutions
                        substitution = self._forced_substitutions(
                            paths, combinator_type, parameters.lvar_to_group
                        )

                        # If no substitution can lead to a match, ignore this combinator
                        if substitution is None:
                            continue
                    else:
                        substitution = {}

                    # If there is at most one possible substitution for each variable
                    # only consider these substitutions
                    if substitution != {} and not parameters.cache:
                        # Keep the forced substitutions and enumerate the rest
                        selected_instantiations = self._instantiate(
                            self.literals, parameters, substitution
                        )
                    else:
                        # otherwise enumerate the whole substitution space.
                        # but do this only the first time... this is time consuming
                        # Update the repository with the enumerated substitutions.
                        if selected_instantiations is None:
                            selected_instantiations = self._instantiate(self.literals, parameters)
                            self.repository[combinator] = (
                                parameters,
                                selected_instantiations,
                                combinator_type,
                            )
                        if substitution != {}:
                            selected_instantiations = (
                                i
                                for i in selected_instantiations
                                if all(i.substitution[k] == v for k, v in substitution.items())
                            )

                    # regardless of how the substitutions were constructed, carry on
                    # with inhabitation. Consider all possible substitutions,
                    for instantiation in selected_instantiations:
                        specific_params = None
                        parameter_arguments = None

                        # and every arity of the combinator type
                        for nary_types in combinator_type:
                            arguments: list[list[Type]] = list(
                                self._subqueries(nary_types, paths, instantiation.substitution)
                            )

                            if not arguments:
                                continue

                            if not specific_params:  # Do this only once for each instantiation
                                specific_params = {
                                    param.name: param.group.subst(instantiation.substitution)
                                    for param in parameters.term_params
                                }

                                type_targets.extend(specific_params.values())
                            if not parameter_arguments:
                                parameter_arguments = list(
                                    parameters.intantiate(instantiation.substitution)
                                )

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
                                        parameters.variable_names,
                                        subquery,
                                    ),
                                )

                                type_targets.extendleft(subquery)

        # prune not inhabited types
        return FiniteCombinatoryLogic._prune(memo)

    @staticmethod
    def _prune(memo: ParameterizedTreeGrammar[Type, C]) -> ParameterizedTreeGrammar[Type, C]:
        """Keep only productive grammar rules."""

        ground_types: set[Type] = set()
        queue: set[Type] = set()
        inverse_grammar: dict[Type, set[tuple[Type, frozenset[Type]]]] = defaultdict(set)

        for n, exprs in memo.as_tuples():
            for expr in exprs:
                non_terminals = expr.non_terminals()
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

        return ParameterizedTreeGrammar(
            {
                target: deque(
                    possibility
                    for possibility in memo[target]
                    if all(t in ground_types for t in possibility.non_terminals())
                )
                for target in ground_types
            }
        )
