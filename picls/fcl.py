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
from multiprocessing.connection import Connection
from multiprocessing.managers import BaseManager
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Optional
from uuid import uuid4, UUID

from multiprocessing import JoinableQueue, Lock, Manager, Pipe, Queue, Process
from queue import Empty

from .grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import (
    Arrow,
    Constructor,
    Intersection,
    Literal,
    Omega,
    Param,
    LitParamSpec,
    Product,
    SetTo,
    TermParamSpec,
    Type,
)

C = TypeVar("C")

# ([theta_1, ..., theta_m], [sigma_1, ..., sigma_n], tau) means
#   theta_1 => ... => theta_m => sigma_1 -> ... -> sigma_n -> tau


# class InhabitationManager(BaseManager):
#     pass
#
#
# InhabitationManager.register("set", set)
#
# puts = 0
# dones = 0


# def putprint():
#     global puts
#     puts += 1
#     print(f"put: {dones}/{puts}")
#
#
# def doneprint():
#     global dones
#     dones += 1
#     print(f"done: {dones}/{puts}")


# class Process:
#     def __init__(self, target: Callable[..., None], args: tuple[Any, ...]) -> None:
#         self.f = target
#         self.args = args
#
#     def start(self) -> None:
#         self.f(*self.args)
#
#     def join(self) -> None:
#         pass


def encode_type(
    typ: Type, constructors: list[str], litvalues: list[Any], litgroups: list[str]
) -> list[int]:
    match typ:
        case Omega():
            return [0]
        case Arrow(src, tgt):
            return (
                [1]
                + encode_type(src, constructors, litvalues, litgroups)
                + encode_type(tgt, constructors, litvalues, litgroups)
            )
        case Intersection(left, right):
            return (
                [2]
                + encode_type(left, constructors, litvalues, litgroups)
                + encode_type(right, constructors, litvalues, litgroups)
            )
        case Product(left, right):
            return (
                [3]
                + encode_type(left, constructors, litvalues, litgroups)
                + encode_type(right, constructors, litvalues, litgroups)
            )
        case Constructor(name, args):
            return [4, constructors.index(name)] + encode_type(
                args, constructors, litvalues, litgroups
            )
        case Literal(value, group):
            return [5, litvalues.index(value), litgroups.index(group)]
    return []


def decode_type(
    encoded_types: list[int],
    constructors: list[str],
    litvalues: list[Any],
    litgroups: list[str],
) -> Type:
    head = encoded_types.pop()
    match head:
        case 0:
            return Omega()
        case 1:
            src = decode_type(encoded_types, constructors, litvalues, litgroups)
            tgt = decode_type(encoded_types, constructors, litvalues, litgroups)
            return Arrow(src, tgt)
        case 2:
            left = decode_type(encoded_types, constructors, litvalues, litgroups)
            right = decode_type(encoded_types, constructors, litvalues, litgroups)
            return Intersection(left, right)
        case 3:
            left = decode_type(encoded_types, constructors, litvalues, litgroups)
            right = decode_type(encoded_types, constructors, litvalues, litgroups)
            return Product(left, right)
        case 4:
            name = constructors[encoded_types.pop()]
            args = decode_type(encoded_types, constructors, litvalues, litgroups)
            return Constructor(name, args)
        case 5:
            value = litvalues[encoded_types.pop()]
            group = litgroups[encoded_types.pop()]
            return Literal(value, group)
    return Omega()


T = TypeVar("T")


# class XQueue(Generic[T]):
#     def __init__(self) -> None:
#         self._queue: deque[T] = deque()
#         self.puts = 0
#         self.dones = 0
#         self.gets = 0
#
#     def put(self, item: T) -> None:
#         self.puts += 1
#         self._queue.append(item)
#
#     def get_nowait(self) -> T:
#         if len(self._queue) == 0:
#             raise Empty
#         self.gets += 1
#         return self._queue.popleft()
#
#     def empty(self) -> bool:
#         return len(self._queue) == 0
#
#     def task_done(self):
#         self.dones += 1
#
#     def join(self) -> None:
#         if self.puts == self.dones:
#             return
#         print("NONONO")
#
#     def qsize(self) -> int:
#         return len(self._queue)


@dataclass(frozen=True)
class MultiArrow:
    # lit_params: list[LitParamSpec]
    # term_params: list[TermParamSpec]
    args: list[Type]
    target: Type

    def subst(self, substitution: dict[str, Literal]) -> MultiArrow:
        return MultiArrow(
            [arg.subst(substitution) for arg in self.args],
            self.target.subst(substitution),
        )


# T may be UUID or Predicate
@dataclass
class InstantiationMeta(Generic[T]):
    term_params: list[tuple[str, Type]]
    predicates: list[T]
    arguments: list[Literal | GVar]
    substitutions: dict[str, Literal]


TreeGrammar: TypeAlias = MutableMapping[Type, deque[tuple[C, list[Type]]]]


class LiteralNotFoundException(Exception):
    pass


class FiniteCombinatoryLogic(Generic[C]):
    def __init__(
        self,
        repository: Mapping[C, Param | Type],
        subtypes: Optional[Mapping[str, set[str]]] | Subtypes = None,
        literals: Optional[Mapping[str, list[Any]]] = None,
    ):
        self.literals: Mapping[str, list[Any]] = {} if literals is None else literals
        self.c_to_uuid: Mapping[C, UUID] = {c: uuid4() for c in repository.keys()}
        self.uuid_to_c: Mapping[UUID, C] = {
            self.c_to_uuid[c]: c for c in repository.keys()
        }
        self._repository: Mapping[
            UUID,
            tuple[list[InstantiationMeta[Predicate]], list[list[MultiArrow]]],
        ] = {
            self.c_to_uuid[c]: FiniteCombinatoryLogic._function_types(ty, self.literals)
            for c, ty in repository.items()
        }
        result: tuple[
            Mapping[
                UUID,
                tuple[list[InstantiationMeta[UUID]], list[list[MultiArrow]]],
            ],
            Mapping[UUID, Predicate],
        ] = self.transform_pred_uuid()
        self.uuid_to_pred = result[1]
        self.repository = result[0]
        self.subtypes: Mapping[str, set[str]] = (
            {}
            if subtypes is None
            else Subtypes.env_closure(subtypes)
            if not isinstance(subtypes, Subtypes)
            else subtypes.environment
        )

    def _get_all_constructor_names(self) -> list[str]:
        def exctract_constructors(typ: Type) -> list[str]:
            return []

        collect = []
        for _, marrows in self._repository.values():
            for marrow in marrows:
                for arrow in marrow:
                    collect.extend(exctract_constructors(arrow.target))
                    for typ in arrow.args:
                        collect.extend(exctract_constructors(typ))
        return list(set(collect))

    def transform_pred_uuid(
        self,
    ) -> tuple[
        Mapping[
            UUID,
            tuple[list[InstantiationMeta[UUID]], list[list[MultiArrow]]],
        ],
        Mapping[UUID, Predicate],
    ]:
        output1: MutableMapping[
            UUID,
            tuple[list[InstantiationMeta[UUID]], list[list[MultiArrow]]],
        ] = {}
        output2: MutableMapping[UUID, Predicate] = {}
        for c, (meta_list, multiarrows) in self._repository.items():
            new_metas = []
            for meta in meta_list:
                pred_uuid_list = []
                for predicate in meta.predicates:
                    pred_uuid = uuid4()
                    output2[pred_uuid] = predicate
                    pred_uuid_list.append(pred_uuid)
                new_metas.append(
                    InstantiationMeta(
                        meta.term_params,
                        pred_uuid_list,
                        meta.arguments,
                        meta.substitutions,
                    )
                )

            output1.update({c: (new_metas, multiarrows)})

        return output1, output2

    @staticmethod
    def _function_types(
        p_or_ty: Param | Type, literals: Mapping[str, list[Any]]
    ) -> tuple[list[InstantiationMeta[Predicate]], list[list[MultiArrow]],]:
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
    ) -> Iterable[InstantiationMeta[Predicate]]:
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
            yield InstantiationMeta(
                [(p.name, p.group) for p in term_params],
                predicates,
                instantiated_combinator_args,
                substitution,
            )

    @staticmethod
    def _subqueries(
        nary_types: Sequence[MultiArrow],
        paths: Sequence[Type],
        substitutions: Mapping[str, Literal],
        subtypes: Mapping[str, set[str]],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow, Type], bool
        ] = lambda m, t: Subtypes.check_subtype(m.target, t, substitutions, subtypes)
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
                lambda a, b: Subtypes.check_subtype(a, b, substitutions, subtypes),
                args1,
                args2,
            )
        )
        return maximal_elements(intersected_args, compare_args)

    @staticmethod
    def _inhabit_single(
        # current_target: Type,
        target_queue: JoinableQueue[Type],
        # rules_queue: JoinableQueue[tuple[Type, deque[RHSRule[Type, UUID, UUID]]]],
        repository: Mapping[
            UUID, tuple[list[InstantiationMeta[UUID]], list[list[MultiArrow]]]
        ],
        subtypes: Mapping[str, set[str]],
        seen: dict[int, None],
        rules: dict[
            Type,
            deque[
                tuple[
                    dict[str, Type],
                    list[UUID],
                    UUID,
                    list[Literal | GVar],
                    list[Type],
                ]
            ],
        ],
        transactionlock,
    ) -> None:
        # get one
        localseen = set()
        while True:
            # get as much, as you can get
            with transactionlock:
                one_target = target_queue.get()
                seen[hash(one_target)] = None
                localseen.add(one_target)
            batch = [one_target]
            try:
                while True:
                    with transactionlock:
                        next_target = target_queue.get_nowait()
                        localseen.add(next_target)
                        seen[hash(next_target)] = None
                    batch.append(next_target)
            except Empty:
                pass
            for current_target in batch:
                # rules[current_target] = None

                possibilities: deque[
                    tuple[
                        dict[str, Type],
                        list[UUID],
                        UUID,
                        list[Literal | GVar],
                        list[Type],
                    ]
                ] = deque()
                # rules[current_target] = possibilities

                paths: list[Type] = list(current_target.organized)

                # try each combinator and arity
                for combinator, (metas, combinator_type) in repository.items():
                    for meta in metas:
                        for nary_types in combinator_type:
                            arguments: list[list[Type]] = list(
                                FiniteCombinatoryLogic._subqueries(
                                    nary_types, paths, meta.substitutions, subtypes
                                )
                            )
                            if len(arguments) == 0:
                                continue

                            specific_params = {
                                param[0]: param[1].subst(meta.substitutions)
                                for param in meta.term_params
                            }

                            for typ in specific_params.values():
                                # If the target is omega, then the result is junk
                                if typ.is_omega:
                                    # target_queue.task_done()
                                    continue

                                if typ in localseen:
                                    continue

                                if isinstance(typ, Literal):
                                    # target_queue.task_done()
                                    continue

                                if hash(typ) in seen:
                                    # target_queue.task_done()
                                    continue
                                target_queue.put(typ)

                            for subquery in (
                                [ty.subst(meta.substitutions) for ty in query]
                                for query in arguments
                            ):
                                possibilities.append(
                                    (
                                        specific_params,
                                        meta.predicates,
                                        combinator,
                                        meta.arguments,
                                        subquery,
                                    )
                                )
                                for typ in subquery:
                                    # If the target is omega, then the result is junk
                                    if typ.is_omega:
                                        # target_queue.task_done()
                                        continue

                                    if typ in localseen:
                                        continue

                                    if isinstance(typ, Literal):
                                        # target_queue.task_done()
                                        continue

                                    if hash(typ) in seen:
                                        # target_queue.task_done()
                                        continue
                                    target_queue.put(typ)
                # rules[current_target] = possibilities
                rules[current_target] = possibilities
                target_queue.task_done()

    @staticmethod
    def rule_gatherer(
        recv: Connection, queue: JoinableQueue[Type], send: Connection
    ) -> None:
        data = bytearray()
        recv.recv_bytes_into(data)
        queue.join()

    def multiinhabit(
        self, *targets: Type, process_count: int = 4
    ) -> ParameterizedTreeGrammar[Type, C, Predicate]:
        # init queue
        target_queue: JoinableQueue[Optional[Type]] = JoinableQueue()
        for target in targets:
            target_queue.put(target)
        transactionlock = Lock()

        memo: ParameterizedTreeGrammar[Type, UUID, UUID] = ParameterizedTreeGrammar()

        manager = Manager()

        # We only use the keys of the dict. We would like to use a set, but sets are not
        # implementented as a manages synchronized resource. Dicts are, and using the keys of the
        # dict as a set is still faster than a list
        # seen: MutableMapping[Type, None] = manager.dict()
        # memo_rules = manager.dict()
        seen: MutableMapping[int, None] = manager.dict()
        # memo_rules = manager.dict()
        mmemo_rules = []

        processes = []
        for process_nr in range(process_count):
            memo_rules = manager.dict()
            mmemo_rules.append(memo_rules)
            processes.append(
                Process(
                    target=FiniteCombinatoryLogic._inhabit_single,
                    args=(
                        target_queue,
                        self.repository,
                        self.subtypes,
                        seen,
                        memo_rules,
                        transactionlock,
                    ),
                )
            )
            processes[-1].start()

        # watcher = Process(
        #     target=FiniteCombinatoryLogic.qsizewatcher, args=(target_queue,)
        # )
        # watcher.start()

        # Wait until queue is empty
        target_queue.join()
        # memo_rules[1].close()

        # for typ, rules in memo_rules.items():
        # try:
        #     while True:
        #         typ, rules = memo_rules[0].recv()
        #         possibilities = deque()
        #         for tuple in rules:
        #             possibilities.append(RHSRule(*tuple))
        #         memo.update({typ: possibilities})
        # except EOFError:
        #     pass
        for memo_rules in mmemo_rules:
            for typ, rules in memo_rules.items():
                possibilities = deque()
                for tuple in rules:
                    possibilities.append(RHSRule(*tuple))
                memo.update({typ: possibilities})

        for process in processes:
            process.terminate()
        # watcher.terminate()

        return memo.map_over_uuids(
            lambda uuid: self.uuid_to_c[uuid], lambda uuid: self.uuid_to_pred[uuid]
        )

    def inhabit(self, *targets: Type) -> ParameterizedTreeGrammar[Type, C, Predicate]:
        return self.multiinhabit(*targets, process_count=2)
        # return self.multiinhabit(*targets)
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: ParameterizedTreeGrammar[Type, UUID, UUID] = ParameterizedTreeGrammar()

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                possibilities: deque[RHSRule[Type, UUID, UUID]] = deque()
                if isinstance(current_target, Literal):
                    continue
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type] = list(current_target.organized)

                # try each combinator and arity
                for combinator, (metas, combinator_type) in self.repository.items():
                    for meta in metas:
                        for nary_types in combinator_type:
                            arguments: list[list[Type]] = list(
                                self._subqueries(
                                    nary_types, paths, meta.substitutions, self.subtypes
                                )
                            )
                            if len(arguments) == 0:
                                continue

                            specific_params = {
                                param[0]: param[1].subst(meta.substitutions)
                                for param in meta.term_params
                            }

                            type_targets.extend(specific_params.values())

                            for subquery in (
                                [ty.subst(meta.substitutions) for ty in query]
                                for query in arguments
                            ):
                                possibilities.append(
                                    RHSRule(
                                        specific_params,
                                        meta.predicates,
                                        combinator,
                                        meta.arguments,
                                        subquery,
                                    )
                                )
                                type_targets.extendleft(subquery)
                memo.update({current_target: possibilities})

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo.map_over_uuids(
            lambda uuid: self.uuid_to_c[uuid], lambda uuid: self.uuid_to_pred[uuid]
        )

    @staticmethod
    def _prune(memo: ParameterizedTreeGrammar[Type, Any, Any]) -> None:
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
