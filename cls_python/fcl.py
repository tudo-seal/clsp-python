from abc import ABC
from collections import deque
from collections.abc import Iterator, Iterable
from dataclasses import field, dataclass
from functools import partial, reduce, cached_property
from typing import Any, Callable, Optional, Tuple, TypeAlias

from .enumeration import ComputationStep
from .enumeration import EmptyStep
from .enumeration import Enumeration
from .subtypes import Subtypes
from .types import Type, Arrow, Intersection, Omega
from .combinatorics import minimal_covers, maximal_elements, partition

MultiArrow: TypeAlias = Tuple[list[Type], Type]

@dataclass(frozen=True)
class Rule(ABC):
    target: Type = field(init=True, kw_only=True)
    is_combinator: bool = field(init=True, kw_only=True)


@dataclass(frozen=True)
class Failed(Rule):
    target: Type = field()
    is_combinator: bool = field(default=False, init=False)

    def __str__(self):
        return f"Failed({str(self.target)})"


@dataclass(frozen=True)
class Combinator(Rule):
    target: Type = field()
    is_combinator: bool = field(default=True, init=False)
    combinator: object = field(init=True)

    def __str__(self):
        return f"Combinator({str(self.target)}, {str(self.combinator)})"


@dataclass(frozen=True)
class Apply(Rule):
    target: Type = field()
    is_combinator: bool = field(default=False, init=False)
    function_type: Type = field(init=True)
    argument_type: Type = field(init=True)

    def __str__(self):
        return f"@({str(self.function_type)}, {str(self.argument_type)}) : {self.target}"


@dataclass(frozen=True)
class Tree(object):
    rule: Rule = field(init=True)
    children: Tuple['Tree', ...] = field(init=True, default_factory=lambda: ())

    class Evaluator(ComputationStep):
        def __init__(self, outer: 'Tree', results: list[Any]):
            self.outer: 'Tree' = outer
            self.results = results

        def __iter__(self) -> Iterator[ComputationStep]:
            match self.outer.rule:
                case Combinator(_, c):
                    self.results.append(c)
                case Apply(_, _, _):
                    f_arg : list[Any] = []
                    yield Tree.Evaluator(self.outer.children[0], f_arg)
                    yield Tree.Evaluator(self.outer.children[1], f_arg)
                    self.results.append(partial(f_arg[0])(f_arg[1]))
                case _:
                    raise TypeError(f"Cannot apply rule: {self.outer.rule}")
            yield EmptyStep()

    def evaluate(self) -> Any:
        result: list[Any] = []
        self.Evaluator(self, result).run()
        return result[0]

    def __str__(self):
        match self.rule:
            case Combinator(_, _): return str(self.rule)
            case Apply(_, _, _): return f"{str(self.children[0])}({str(self.children[1])})"
            case _: return f"{str(self.rule)} @ ({', '.join(map(str, self.children))})"


@dataclass(frozen=True)
class InhabitationResult(object):
    targets: list[Type] = field(init=True)
    rules: set[Rule] = field(init=True)

    @cached_property
    def grouped_rules(self) -> dict[Type, set[Rule]]:
        result: dict[Type, set[Rule]] = dict()
        for rule in self.rules:
            group: Optional[set[Rule]] = result.get(rule.target)
            if group:
                group.add(rule)
            else:
                result[rule.target] = {rule}
        return result

    def check_empty(self, target: Type) -> bool:
        for rule in self.grouped_rules.get(target, {Failed(target)}):
            if isinstance(rule, Failed):
                return True
        return False

    @cached_property
    def non_empty(self) -> bool:
        for target in self.targets:
            if self.check_empty(target):
                return False
        return bool(self.targets)

    def __bool__(self) -> bool:
        return self.non_empty

    @cached_property
    def infinite(self) -> bool:
        if not self:
            return False

        reachable: dict[Type, set[Type]] = {}
        for (target, rules) in self.grouped_rules.items():
            entry: set[Type] = set()
            for rule in rules:
                match rule:
                    case Apply(target, lhs, rhs):
                        next_reached: set[Type] = {lhs, rhs}
                        entry.update(next_reached)
                    case _:
                        pass
            reachable[target] = entry

        changed: bool = True
        to_check: set[Type] = set(self.targets)
        while changed:
            changed = False
            next_to_check = set()
            for target in to_check:
                can_reach = reachable[target]
                if target in can_reach:
                    return True
                newly_reached = set().union(*(reachable[reached] for reached in can_reach))
                for new_target in newly_reached:
                    if target == new_target:
                        return True
                    elif new_target not in to_check:
                        changed = True
                        next_to_check.add(new_target)
                        can_reach.add(new_target)
                    elif new_target not in can_reach:
                        changed = True
                        can_reach.add(new_target)
            to_check.update(next_to_check)
        return False

    def size(self) -> int:
        if self.infinite:
            return -1
        maximum = self.raw.unsafe_max_size()
        size = 0
        values = self.raw.all_values()
        for _ in range(0, maximum+1):
            trees = next(values)
            size += len(trees)
        return size

    def __getitem__(self, target: Type) -> Enumeration[Tree]:
        if target in self.enumeration_map:
            return self.enumeration_map[target]
        else:
            return Enumeration.empty()

    @staticmethod
    def combinator_result(r: Combinator) -> Enumeration[Tree]:
        return Enumeration.singleton(Tree(r, ()))

    @staticmethod
    def apply_result(result: dict[Type, Enumeration[Tree]], r: Apply) -> Enumeration[Tree]:
        def mkapp(left_and_right):
            return Tree(r, (left_and_right[0], left_and_right[1]))

        def apf():
            return (result[r.function_type] * result[r.argument_type]) \
                    .map(mkapp).pay()
        applied = Enumeration.lazy(apf)
        return applied

    @cached_property
    def enumeration_map(self) -> dict[Type, Enumeration[Tree]]:
        result: dict[Type, Enumeration[Tree]] = dict()
        for (target, rules) in self.grouped_rules.items():
            _enum: Enumeration[Tree] = Enumeration.empty()
            for rule in rules:
                match rule:
                    case Combinator(_, _) as r:
                        _enum = _enum + InhabitationResult.combinator_result(r)
                    case Apply(_, _, _) as r:
                        _enum = _enum + InhabitationResult.apply_result(result, r)
                    case _:
                        pass
            result[target] = _enum
        return result

    @cached_property
    def raw(self) -> Enumeration[Tree | list[Tree]]:
        if not self:
            return Enumeration.empty()
        if len(self.targets) == 1:
            return self.enumeration_map[self.targets[0]]
        else:
            result: Enumeration[list[Tree]] = Enumeration.singleton([])
            for target in self.targets:
                result = (result * self.enumeration_map[target]).map(lambda x: [*x[0], x[1]])
            return result

    @cached_property
    def evaluated(self) -> Enumeration[Any | list[Any]]:
        if len(self.targets) == 1:
            return self.raw.map(lambda t: t.evaluate())
        else:
            return self.raw.map(lambda l: list(map(lambda t: t.evaluate(), l)))
        
class FiniteCombinatoryLogic(object):


    def __init__(self, repository: dict[object, Type], subtypes: Subtypes):
        self.repository: dict[object, list[list[MultiArrow]]] = \
            {c : list(FiniteCombinatoryLogic._function_types(ty)) for c, ty in repository.items()}
        self.subtypes = subtypes

    @staticmethod
    def _function_types(ty: Type) -> Iterable[list[MultiArrow]] :
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type) -> Iterable[tuple[Type, Type]] :
            tys: deque[Type] = deque((ty, ))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        current: list[MultiArrow] = [([], ty)]
        while len(current) != 0:
            yield current
            current = [(args + [new_arg], new_tgt) for (args, tgt) in current
                for (new_arg, new_tgt) in unary_function_types(tgt)]


    def _omega_rules(self, target: Type) -> set[Rule]:
        return {Apply(target, target, target),
                *map(lambda c: Combinator(target, c), self.repository.keys())}

    @staticmethod
    def _combinatory_expression_rules(combinator: object, arguments: list[Type], target: Type) -> Iterable[Rule]:
        """Rules from combinatory expression `combinator(arguments[0], ..., arguments[n])`."""

        remaining_arguments: deque[Type] = deque(arguments)
        while remaining_arguments:
            argument = remaining_arguments.pop()
            yield Apply(target, Arrow(argument, target), argument)
            target = Arrow(argument, target)
        yield Combinator(target, combinator)

    def inhabit(self, *targets: Type) -> InhabitationResult:
        # dictionary of type |-> sequence of combinatory expressions
        memo: dict[Type, deque[tuple[object, list[Type]]]] = dict()
        remaining_targets: deque[Type] = deque(targets)
        # intermediate rules (stopgap from prior fcl implementation)
        result: set[Rule] = set()
        while remaining_targets:
            target = remaining_targets.pop()
            if memo.get(target) is None:
                # target type was not seen before
                paths: list[Type] = list(target._organized())
                possibilities: deque[tuple[object, list[Type]]] = deque()
                memo.update({target : possibilities})
                if target.is_omega:
                    result |= self._omega_rules(target)
                else:
                    # try each combinator and arity
                    for combinator, combinator_type in self.repository.items():
                        for nary_types in combinator_type:
                            # does the target of a multi-arrow contain a given type?
                            target_contains: Callable[[MultiArrow, Type], bool] = \
                                lambda m, t: self.subtypes.check_subtype(m[1], t)
                            # cover target using targets of multi-arrows in nary_types
                            covers = minimal_covers(nary_types, paths, target_contains)
                            # intersect corresponding arguments of multi-arrows in each cover
                            intersect_args: Callable[[list[Type], list[Type]], map[Type]] = \
                                lambda args1, args2: map(Intersection, args1, args2)
                            intersected_args = (list(reduce(intersect_args, (m[0] for m in ms))) for ms in covers)
                            # consider only maximal argument vectors
                            compare_args = lambda args1, args2: all(map(self.subtypes.check_subtype, args1, args2))
                            for subquery in maximal_elements(intersected_args, compare_args):
                                possibilities.append((combinator, subquery))
                                remaining_targets.extendleft(subquery)
        
        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)
        # convert memo into resulting set of rules
        for target, possibilities in memo.items():
            if len(possibilities) == 0 and not target.is_omega:
                result.add(Failed(target))
            for combinator, arguments in possibilities:
                result.update(self._combinatory_expression_rules(combinator, arguments, target))

        #return InhabitationResult(targets=list(targets), rules=FiniteCombinatoryLogic._prune(result))
        return InhabitationResult(targets=list(targets), rules=result)
    
    @staticmethod
    def _prune(memo: dict[Type, deque[tuple[object, list[Type]]]]) -> None :
        """Keep only productive grammar rules."""

        ground_types: set[Type] = set()
        is_ground = lambda args: all(True for arg in args if arg in ground_types)
        new_ground_types, candidates = partition(lambda ty: any(True for (_, args) in memo[ty] if is_ground(args)), memo.keys())
        # initialize inhabited (ground) types
        while new_ground_types:
            ground_types.update(new_ground_types)
            new_ground_types, candidates = partition(lambda ty: any(True for _, args in memo[ty] if is_ground(args)), candidates)

        for target, possibilities in memo.items():
            memo[target] = deque(possibility for possibility in possibilities if is_ground(possibility[1]))
