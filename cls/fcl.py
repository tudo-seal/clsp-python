# Propositional Finite Combinatory Logic

from collections import deque, namedtuple
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, TypeAlias, TypeVar

from cls.grammar import Binder, GroundTerm, ParameterizedTreeGrammar, RHSRule

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import Arrow, Intersection, Omega, Param, ParamSpec, Type

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")

# ([theta_1, ..., theta_m], [sigma_1, ..., sigma_n], tau) means theta_1 => ... => theta_m => sigma_1 -> ... -> sigma_n -> tau


@dataclass(frozen=True)
class MultiArrow(Generic[T]):
    params: list[ParamSpec[T]]
    args: list[Type[T]]
    target: Type[T]


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
    return (str(list(map(str, m.args))), str(m.target))


class FiniteCombinatoryLogic(Generic[T, C]):
    def __init__(
        self,
        repository: Mapping[C, Param[T] | Type[T]],
        subtypes: Subtypes[T] = Subtypes({}),
        literals: dict[Any, list[Any]] = {},
    ):
        self.repository: Mapping[C, list[list[MultiArrow[T]]]] = {
            c: list(FiniteCombinatoryLogic._function_types(ty))
            for c, ty in repository.items()
        }
        self.subtypes = subtypes
        self.literals = literals

    @staticmethod
    def _function_types(p_or_ty: Param[T] | Type[T]) -> Iterable[list[MultiArrow[T]]]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type[T]) -> Iterable[tuple[Type[T], Type[T]]]:
            tys: deque[Type[T]] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        def split_params(ty: Param[T] | Type[T]) -> tuple[list[ParamSpec[T]], Type[T]]:
            params: list[ParamSpec[T]] = []
            while isinstance(ty, Param):
                params.append(ParamSpec(ty.name, ty.type, ty.predicate))
                ty = ty.inner
            return (params, ty)

        params, ty = split_params(p_or_ty)
        current: list[MultiArrow[T]] = [MultiArrow(params, [], ty)]

        while len(current) != 0:
            yield current
            current = [
                MultiArrow(c.params, c.args + [new_arg], new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]

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

    def _instantiate(
        self, combinator: C, ty: MultiArrow[T], type_targets: deque[Type[T]]
    ) -> tuple[C, MultiArrow[T], dict[str, Any]]:
        if len(ty.params) > 0:
            pass
        else:
            return (combinator, ty, {})
        for param in ty.params:
            # Check if literal parameter or term parameter
            if isinstance(param.type, Type):
                if param.type in self.literals:
                    for literal in self.literals[param.type]:
                        pass

        return (combinator, ty, {})

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
                for combinator, combinator_type in self.repository.items():
                    for nary_types in combinator_type:
                        arguments: list[list[Type[T]]] = list(
                            self._subqueries(nary_types, paths)
                        )
                        if len(arguments) == 0:
                            continue

                        for subquery in arguments:
                            possibilities.append(
                                RHSRule([], [], GroundTerm(combinator, subquery))
                            )
                            type_targets.extendleft(subquery)

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: ParameterizedTreeGrammar[Type[T], C]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(
            args: Sequence[str | Binder[Type[T]] | Type[T]], ground_types: set[Type[T]]
        ) -> bool:
            return all(True for arg in args if arg in ground_types)

        ground_types: set[Type[T]] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True
                for rule in memo[ty]
                if is_ground(rule.ground_term.args, ground_types)
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
                    if is_ground(rule.ground_term.args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.as_tuples():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(possibility.ground_term.args, ground_types)
            )
