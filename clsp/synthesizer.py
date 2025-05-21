"""Synthesizer implementing Finite Combinatory Logic with Predicates.
   It constructs a logic program via `constructSolutionSpace` from the following ingredients:
   - collection of component specifications
   - parameter space
   - optional specification taxonomy
   - target specification"""

from collections import deque
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
    Hashable,
    Container,
)
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)
from .solution_space import SolutionSpace, TerminalArgument, NonTerminalArgument
from .combinatorics import maximal_elements, minimal_covers
from .subtypes import Subtypes, Taxonomy
from .types import (
    Arrow,
    Intersection,
    Parameter,
    LiteralParameter,
    TermParameter,
    Abstraction,
    Implication,
    Predicate,
    Type,
)

# type of components
C = TypeVar("C", bound=Hashable)

# type of component specifications
Specification = Abstraction | Implication | Type

# type of parameter space
ParameterSpace = Mapping[str, Iterable | Container]

@dataclass(frozen=True)
class MultiArrow:
    # type of shape arg1 -> arg2 -> ... -> argN -> target
    args: tuple[Type, ...]
    target: Type

    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"{[str(a) for a in self.args]} -> {str(self.target)}"
        else:
            return str(self.target)

@dataclass()
class CombinatorInfo:
    # container for auximiary information about a combinator
    prefix: list[LiteralParameter | TermParameter | Predicate]
    groups: dict[str, str]
    term_predicates: tuple[Callable[[dict[str, Any]], bool], ...]
    instantiations: deque[dict[str, Any]] | None
    type: list[list[MultiArrow]]

class Synthesizer(Generic[C]):
    def __init__(
        self,
        componentSpecifications: Mapping[C, Specification],
        parameterSpace: ParameterSpace | None = None,
        taxonomy: Taxonomy = {},
    ):
        self.literals: ParameterSpace = {} if parameterSpace is None else {
            k: frozenset(vs) if isinstance(vs, Iterable) and all(isinstance(v, Hashable) for v in vs) else vs for k, vs in parameterSpace.items()
            }
        self.repository: tuple[tuple[C, CombinatorInfo], ...] = tuple((c, Synthesizer._function_types(ty)) for c, ty in componentSpecifications.items())
        self.subtypes = Subtypes(taxonomy)

    @staticmethod
    def _function_types(
        parameterizedType: Specification,
    ) -> CombinatorInfo:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type) -> Iterable[tuple[Type, Type]]:
            tys: deque[Type] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        prefix: list[LiteralParameter | TermParameter | Predicate] = []
        groups: dict[str, str] = {}
        while not isinstance(parameterizedType, Type):
            if isinstance(parameterizedType, Abstraction):
                param = parameterizedType.parameter
                if isinstance(param, LiteralParameter):
                    prefix.append(param)
                    groups[param.name] = param.group
                elif isinstance(param, TermParameter):
                    prefix.append(param)
                parameterizedType = parameterizedType.body
            elif isinstance(parameterizedType, Implication):
                prefix.append(parameterizedType.predicate)
                parameterizedType = parameterizedType.body

        current: list[MultiArrow] = [MultiArrow(tuple(), parameterizedType)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow(c.args + (new_arg ,), new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]

        term_predicates: tuple[Callable[[dict[str, Any]], bool], ...] = tuple(p.constraint for p in prefix if isinstance(p, Predicate) and not p.only_literals)
        return CombinatorInfo(prefix, groups, term_predicates, None, multiarrows)

    def _enumerate_substitutions(
        self,
        prefix: list[LiteralParameter | TermParameter | Predicate],
        initial_substitution: dict[str, Any] = {},
    ) -> deque[dict[str, Any]]:
        """Enumerate all substitutions for the given parameters.
           Take initial_substitution with inferred literals into account."""

        substitutions: deque[dict[str, Any]] = deque([{}])

        for parameter in prefix:
            new_substitutions: deque[dict[str, Any]] = deque()
            if isinstance(parameter, LiteralParameter):
                if parameter.group not in self.literals or not substitutions:
                    return deque()
                else:
                    for substitution in substitutions:
                        if parameter.name in initial_substitution:
                            value = initial_substitution[parameter.name]
                            if parameter.values is not None and value not in parameter.values(substitution):
                                # the inferred value is not in the set of values
                                continue
                            if not value in self.literals[parameter.group]:
                                # the inferred value is not in the group
                                continue
                            new_substitutions.append({**substitution, parameter.name: value})
                        elif parameter.values is not None:
                            for value in parameter.values(substitution):
                                if value in self.literals[parameter.group]:
                                    new_substitutions.append({**substitution, parameter.name: value})
                        else:
                            concrete_values = self.literals[parameter.group]
                            if not isinstance(concrete_values, Iterable):
                                raise RuntimeError(
                                    f"The value of variable {parameter.name} could not be inferred."
                                )
                            for value in concrete_values:
                                new_substitutions.append({**substitution, parameter.name: value})

                    substitutions = new_substitutions

            if isinstance(parameter, Predicate) and parameter.only_literals:
                substitutions = deque(substitution for substitution in substitutions if parameter.constraint(substitution))

        return substitutions

    def _subqueries(
        self,
        nary_types: list[MultiArrow],
        paths: Iterable[Type],
        groups: dict[str, str],
        substitution: dict[str, Any],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[[MultiArrow, Type], bool] = (
            lambda m, t: self.subtypes.check_subtype(m.target, t, groups, substitution)
        )
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[[Iterable[Type], Iterable[Type]], tuple[Type, ...]] = (
            lambda args1, args2: tuple(Intersection(a, b) for a, b in zip(args1, args2))
        )

        intersected_args = (list(reduce(intersect_args, (m.args for m in ms))) for ms in covers)
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(
                lambda a, b: self.subtypes.check_subtype(a, b, groups, substitution),
                args1,
                args2,
            )
        )
        return maximal_elements(intersected_args, compare_args)

    def _necessary_substitution(
        self,
        paths: Iterable[Type],
        combinator_type: list[list[MultiArrow]],
        groups: dict[str, str],
    ) -> dict[str, Any] | None:
        """
        Computes a substitution that needs to be part of every substitution S such that
        S(combinator_type) <= paths.

        If no substitution can make this valid, None is returned.
        """

        result: dict[str, Any] = {}

        for path in paths:
            unique_substitution: dict[str, Any] | None = None
            is_unique = True

            for nary_types in combinator_type:
                for ty in nary_types:
                    substitution = self.subtypes.infer_substitution(ty.target, path, groups)
                    if substitution is None:
                        continue
                    if unique_substitution is None:
                        unique_substitution = substitution
                    else:
                        is_unique = False
                        break
                if not is_unique:
                    break

            if unique_substitution is None:
                return None  # no substitution for this path
            if not is_unique:
                continue  # substitution not unique substitution — skip

            # merge consistent substitution
            for k, v in unique_substitution.items():
                if k in result:
                    if result[k] != v:
                        return None  # conflict in necessary substitution
                else:
                    result[k] = v

        return result

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

                # try each combinator
                for combinator, combinatorInfo in self.repository:
                    # Compute necessary substitutions
                    substitution = self._necessary_substitution(current_target.organized, combinatorInfo.type, combinatorInfo.groups)

                    # If there cannot be a suitable substitution, ignore this combinator
                    if substitution is None:
                        continue

                    # If there is a unique substitution, use it directly
                    if substitution:
                        # Keep necessary substitutions and enumerate the rest
                        selected_instantiations = self._enumerate_substitutions(combinatorInfo.prefix, substitution)
                    else:
                        # Enumerate all substitutions (only the first time).
                        if combinatorInfo.instantiations is None:
                            combinatorInfo.instantiations = self._enumerate_substitutions(combinatorInfo.prefix)
                        selected_instantiations = combinatorInfo.instantiations                

                    # consider all possible instantiations
                    for instantiation in selected_instantiations:
                        parameter_arguments = None

                        # and every arity of the combinator type
                        for nary_types in combinatorInfo.type:
                            arguments: Sequence[list[Type]] = self._subqueries(nary_types, current_target.organized, combinatorInfo.groups, instantiation)

                            if len(arguments) == 0:
                                continue

                            if not parameter_arguments: # do this only once for each instantiation
                                    parameter_arguments = tuple(TerminalArgument(param.name, instantiation[param.name])
                                                                if isinstance(param, LiteralParameter)
                                                                else NonTerminalArgument[Type](param.name, param.group.subst(combinatorInfo.groups, instantiation))
                                                                for param in combinatorInfo.prefix
                                                                if isinstance(param, Parameter))
                                    type_targets.extend(argument.value for argument in parameter_arguments if isinstance(argument, NonTerminalArgument))

                            for subquery in (
                                tuple(NonTerminalArgument(None, ty.subst(combinatorInfo.groups, instantiation)) for ty in query)
                                for query in arguments
                            ):
                                memo.add_rule(current_target, combinator, parameter_arguments + subquery, combinatorInfo.term_predicates)
                                type_targets.extendleft(q.value for q in subquery)

        # prune not inhabited types
        return memo.prune()
