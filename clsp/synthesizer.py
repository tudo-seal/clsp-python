"""Synthesizer implementing Finite Combinatory Logic with Predicates.
   It constructs a logic program via `constructSolutionSpace` from the following ingredients:
   - collection of component specifications
   - parameter space
   - optional specification taxonomy
   - target specification"""

from __future__ import annotations
from collections import deque
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
    Protocol,
    TypeVar,
)

from .solution_space import SolutionSpace, RHSRule, TerminalArgument, NonTerminalArgument

from .combinatorics import (
    maximal_elements,
    minimal_covers,
)
from .subtypes import Subtypes, Taxonomy
from .types import (
    Arrow,
    Intersection,
    Literal,
    LiteralParameter,
    TermParameter,
    Abstraction,
    Type,
    Var,
)

# type of components
C = TypeVar("C")

# type of component specifications
Specification = Abstraction | Type

class Contains(Protocol):
    def __contains__(self, value: object) -> bool: ...

# type of parameter space
ParameterSpace = Mapping[str, Iterable[Any] | Contains]

@dataclass(frozen=True)
class MultiArrow:
    # type of shape arg1 -> arg2 -> ... -> argN -> target
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
    # information on parameters of a combinator
    literal_params: list[LiteralParameter]
    term_params: list[TermParameter]
    lvar_to_group: dict[str, str]
    param_names: list[Var | str]

class Synthesizer(Generic[C]):
    def __init__(
        self,
        componentSpecifications: Mapping[C, Specification],
        parameterSpace: ParameterSpace | None = None,
        taxonomy: Taxonomy = {},
    ):
        self.literals: ParameterSpace = {} if parameterSpace is None else parameterSpace
        self.repository: MutableMapping[
            C,
            tuple[ParamInfo, list[dict[str, Literal]] | None, list[list[MultiArrow]]],
        ] = {c: Synthesizer._function_types(ty) for c, ty in componentSpecifications.items()}
        self.subtypes = Subtypes(taxonomy)

    @staticmethod
    def _function_types(
        parameterizedType: Specification,
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

        parameters = ParamInfo([], [], {}, [])
        while isinstance(parameterizedType, Abstraction):
            param = parameterizedType.parameter
            #TODO account for parameters.variable_names.append(param.name)
            if isinstance(param, LiteralParameter):
                parameters.literal_params.append(param)
                parameters.lvar_to_group[param.name] = param.group
                parameters.param_names.append(Var(param.name))
            elif isinstance(param, TermParameter):
                parameters.term_params.append(param)
                parameters.param_names.append(param.name)
            parameterizedType = parameterizedType.body

        current: list[MultiArrow] = [MultiArrow([], parameterizedType)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]
        return (parameters, None, multiarrows)

    def _enumerate_substitutions(
        self,
        parameters: ParamInfo,
        initial_substitution: dict[str, Literal] = {},
    ) -> list[dict[str, Literal]]:
        """Enumerate all substitutions for the given parameters.
           Take initial_substitution with inferred literals into account."""

        substitutions: deque[dict[str, Literal]] = deque([{}])

        for literal_parameter in parameters.literal_params:
            if literal_parameter.group not in self.literals:
                return []
            if not substitutions:
                return []
            else:
                new_substitutions: deque[dict[str, Literal]] = deque()
                for substitution in substitutions:
                    values: Sequence[Literal]
                    if literal_parameter.name in initial_substitution:
                        value = initial_substitution[literal_parameter.name]
                        if literal_parameter.values is not None and value not in literal_parameter.values(substitution):
                            # the inferred value is not in the set of values
                            continue
                        if not value.value in self.literals[literal_parameter.group]:
                            # the inferred value is not in the group
                            continue
                        values = [value]
                    elif literal_parameter.values is not None:
                        values = [value
                                  for value in literal_parameter.values(substitution)
                                  if value.value in self.literals[literal_parameter.group]]
                    else:
                        concrete_values = self.literals[literal_parameter.group]
                        if not isinstance(concrete_values, Iterable):
                            raise RuntimeError(
                                f"The value of variable {literal_parameter.name} could not be inferred."
                            )
                        values = [Literal(value, literal_parameter.group) for value in concrete_values]

                    for value in values:
                        new_substitution = substitution | {literal_parameter.name: value}
                        if literal_parameter.predicate(new_substitution):
                            new_substitutions.append(new_substitution)
                substitutions = new_substitutions

        return list(substitutions)

    def _subqueries(
        self,
        nary_types: list[MultiArrow],
        paths: list[Type],
        substitution: dict[str, Literal],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[[MultiArrow, Type], bool] = (
            lambda m, t: self.subtypes.check_subtype(m.target, t, substitution)
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
                lambda a, b: self.subtypes.check_subtype(a, b, substitution),
                args1,
                args2,
            )
        )
        return maximal_elements(intersected_args, compare_args)

    def _necessary_substitution(
        self,
        paths: list[Type],
        combinator_type: list[list[MultiArrow]],
        groups: dict[str, str],
    ) -> dict[str, Literal] | None:
        """
        Computes a substitution that needs to be part of every substitution S such that
        S(combinator_type) <= paths.

        If no substitution can make this valid, None is returned.
        """

        return_subsitution: dict[str, Literal] = {}

        for path in paths:
            # Check all targets of the multiarrows
            candidates: list[Type] = [ty.target for nary_types in combinator_type for ty in nary_types]

            substitutions: list[dict[str, Literal]] = [
                substitution
                for candidate in candidates
                if (substitution := self.subtypes.infer_substitution(candidate, path, groups))
                is not None
            ]

            # if no substitution is applicable, this path cannot be covered
            if len(substitutions) == 0:
                return None

            if len(substitutions) == 1:
                # if a path is uniquely covered, then update a unified substitution
                for k, v in substitutions[0].items():
                    if k in return_subsitution:
                        if v != return_subsitution[k]:
                            # there are inconsistent necessary substitutions
                            return None
                    else:
                        return_subsitution[k] = v

        return return_subsitution

    def constructSolutionSpace(self, *targets: Type) -> SolutionSpace[Type, C]:
        """Constructs a logic program in the current environment for the given target types."""
        type_targets = deque(targets)

        # constructed logic program
        memo: SolutionSpace[Type, C] = SolutionSpace()

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
                    # Compute necessary substitutions
                    substitution = self._necessary_substitution(paths, combinator_type, parameters.lvar_to_group)

                    # If there cannot be a suitable substitution, ignore this combinator
                    if substitution is None:
                        continue

                    # If there is a unique substitution, use it directly
                    if substitution:
                        # Keep necessary substitutions and enumerate the rest
                        selected_instantiations = self._enumerate_substitutions(parameters, substitution)
                    else:
                        # otherwise enumerate all substitutions (only the first time).
                        # update the repository with the enumerated substitutions.
                        if instantiations is None:
                            selected_instantiations = self._enumerate_substitutions(parameters)
                            self.repository[combinator] = (
                                parameters,
                                selected_instantiations,
                                combinator_type,
                            )
                        else:
                            selected_instantiations = instantiations                

                    # consider all possible instantiations
                    for instantiation in selected_instantiations:
                        specific_params = None
                        parameter_arguments = None

                        # and every arity of the combinator type
                        for nary_types in combinator_type:
                            arguments: list[list[Type]] = list(
                                self._subqueries(nary_types, paths, instantiation)
                            )

                            if not arguments:
                                continue

                            if not specific_params:  # do this only once for each instantiation
                                specific_params = {
                                    param.name: param.group.subst(instantiation)
                                    for param in parameters.term_params
                                }

                                type_targets.extend(specific_params.values())
                            if not parameter_arguments: # do this only once for each instantiation
                                    parameter_arguments = tuple(TerminalArgument(n.name, instantiation[n.name].value)
                                                           if isinstance(n, Var) else NonTerminalArgument[Type](n, specific_params[n])
                                                           for n in parameters.param_names)

                            term_predicates = tuple(term_param.predicate for term_param in parameters.term_params)
                            for subquery in (
                                [ty.subst(instantiation) for ty in query]
                                for query in arguments
                            ):
                                memo.add_rule(
                                    current_target,
                                    RHSRule(
                                        parameter_arguments + tuple(NonTerminalArgument(None, ty) for ty in subquery),
                                        term_predicates,
                                        combinator,
                                    ),
                                )

                                type_targets.extendleft(subquery)

        # prune not inhabited types
        return memo.prune()
