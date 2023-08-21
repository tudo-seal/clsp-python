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
from uuid import uuid4, UUID

from multiprocessing import (
    Lock,
    Manager,
    Process,
    Semaphore,
    Value,
)


from multiprocessing.managers import SharedMemoryManager
from pickle import UnpicklingError, dumps, loads

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


def encode_type(typ: Type) -> bytes:
    match typ:
        case Arrow(src, tgt):
            return b"\x01" + encode_type(src) + encode_type(tgt)
        case Intersection(left, right):
            return b"\x02" + encode_type(left) + encode_type(right)
        case Product(left, right):
            return b"\x03" + encode_type(left) + encode_type(right)
        case Constructor(name, args):
            return b"\x04" + name.encode() + b"\x00" + encode_type(args)
        case Literal(value, group):
            byte_value = dumps(value)
            return (
                b"\x05"
                + len(byte_value).to_bytes(4, "big")
                + byte_value
                + group.encode()
                + b"\x00\x01"  # see https://github.com/python/cpython/issues/106939
            )
        case Omega():
            return b"\x06"
    return b"\x00"


def decode_type(encoded_type: bytes):
    def _decode_type(encoded_type: bytes, index: int = 0) -> tuple[int, Type]:
        match encoded_type[index]:
            case 1:
                end_src, src = _decode_type(encoded_type, index + 1)
                end_tgt, tgt = _decode_type(encoded_type, end_src + 1)
                return (end_tgt, Arrow(src, tgt))
            case 2:
                end_left, left = _decode_type(encoded_type, index + 1)
                end_right, right = _decode_type(encoded_type, end_left + 1)
                return (end_right, Intersection(left, right))
            case 3:
                end_left, left = _decode_type(encoded_type, index + 1)
                end_right, right = _decode_type(encoded_type, end_left + 1)
                return (end_right, Product(left, right))
            case 4:
                index += 1
                end = index
                while encoded_type[end] != 0:
                    end += 1
                name = encoded_type[index:end].decode()
                end_args, args = _decode_type(encoded_type, end + 1)
                return (end_args, Constructor(name, args))
            case 5:
                index += 1
                end_length = index + 4
                value_length = int.from_bytes(encoded_type[index:end_length], "big")
                end_value = end_length + value_length
                literal_value = loads(encoded_type[end_length:end_value])
                end_group = end_value
                while encoded_type[end_group] != 0:
                    end_group += 1
                group_name = encoded_type[end_value:end_group].decode()
                return (end_group + 1, Literal(literal_value, group_name))

            case 6:
                return (index, Omega())
        return (0, Omega())

    return _decode_type(encoded_type)[1]


T = TypeVar("T")


@dataclass(frozen=True)
class MultiArrow:
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
        repository: Mapping[
            UUID, tuple[list[InstantiationMeta[UUID]], list[list[MultiArrow]]]
        ],
        subtypes: Mapping[str, set[str]],
        seen: dict[bytes, None],
        seenlock,
        put_count,
        done_count,
        get_index,
        put_index,
        target_list,
        filled_sem,
        rules_dump,
        done_sem,
        queue_shm_size,
    ) -> None:
        # get one
        localseen = set()
        rules_written = 0
        while True:
            # get as much, as you can get
            filled_sem.acquire()
            batch: list[int] = []
            with get_index.get_lock():
                batch.append(get_index.value)
                get_index.value += 1
                if get_index.value >= queue_shm_size:
                    get_index.value = 0
                while filled_sem.acquire(False):
                    batch.append(get_index.value)
                    get_index.value += 1
                    if get_index.value >= queue_shm_size:
                        get_index.value = 0

            for current_target in (decode_type(target_list[index]) for index in batch):
                possibilities: list[
                    tuple[
                        dict[str, Type],
                        list[UUID],
                        UUID,
                        list[Literal | GVar],
                        list[Type],
                    ]
                ] = list()

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
                                    continue

                                if typ in localseen:
                                    continue
                                localseen.add(typ)

                                if isinstance(typ, Literal):
                                    continue

                                etype = encode_type(typ)
                                with seenlock:
                                    if etype in seen:
                                        continue
                                    seen[etype] = None
                                with put_index.get_lock():
                                    put = put_index.value
                                    put_index.value += 1
                                    if put_index.value == get_index.value:
                                        raise RuntimeError("Queue to small")
                                    if put_index.value >= queue_shm_size:
                                        put_index.value = 0
                                with put_count.get_lock():
                                    put_count.value += 1
                                target_list[put] = etype
                                filled_sem.release()

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
                                        continue

                                    if typ in localseen:
                                        continue
                                    localseen.add(typ)

                                    if isinstance(typ, Literal):
                                        continue

                                    etype = encode_type(typ)
                                    with seenlock:
                                        if etype in seen:
                                            continue
                                        seen[etype] = None

                                    with put_index.get_lock():
                                        put = put_index.value
                                        put_index.value += 1
                                        if put_index.value == get_index.value:
                                            raise RuntimeError("Queue to small")
                                        if put_index.value >= queue_shm_size:
                                            put_index.value = 0

                                    with put_count.get_lock():
                                        put_count.value += 1

                                    target_list[put] = encode_type(typ)
                                    filled_sem.release()
                rules_dump[rules_written] = dumps((current_target, possibilities))
                rules_written += 1
                with done_count.get_lock():
                    done_count.value += 1
                    if done_count.value == put_count.value:
                        done_sem.release()
                        return
                # target_queue.task_done()

    def multiinhabit(
        self, *targets: Type, process_count: int = 4
    ) -> ParameterizedTreeGrammar[Type, C, Predicate]:
        with SharedMemoryManager() as smm:
            # init queue
            queue_shm_size = 1024
            target_list = smm.ShareableList(
                [b"\x01" * 256] * queue_shm_size
            )  # one hundred megabyte of shm
            for i, target in enumerate(targets):
                target_list[i] = encode_type(target)

            put_count = Value("I", len(targets))
            done_count = Value("I", 0)
            get_index = Value("I", 0)  # Points to last entry popped
            put_index = Value("I", len(targets))  # Points to first free entry
            filled_sem = Semaphore(len(targets))
            done_sem = Semaphore(0)
            seenlock = Lock()

            memo: ParameterizedTreeGrammar[
                Type, C, Predicate
            ] = ParameterizedTreeGrammar()

            manager = Manager()

            # We only use the keys of the dict. We would like to use a set, but sets are not
            # implementented as a manages synchronized resource. Dicts are, and using the keys of the
            # dict as a set is still faster than a list
            seen: MutableMapping[int, None] = manager.dict()
            mmemo_rules = []

            processes = []
            for _ in range(process_count):
                memo_rules = smm.ShareableList([b"\x01" * 1024] * 10240)
                mmemo_rules.append(memo_rules)
                processes.append(
                    Process(
                        target=FiniteCombinatoryLogic._inhabit_single,
                        args=(
                            # target_queue,
                            self.repository,
                            self.subtypes,
                            seen,
                            seenlock,
                            put_count,
                            done_count,
                            get_index,
                            put_index,
                            target_list,
                            filled_sem,
                            memo_rules,
                            done_sem,
                            queue_shm_size,
                        ),
                    )
                )
                processes[-1].start()

            # Wait until queue is empty
            done_sem.acquire()

            for memo_rules in mmemo_rules:
                try:
                    for entry in memo_rules:
                        typ, rules = loads(entry)
                        possibilities = deque()
                        for (
                            binder,
                            predicates_uuid,
                            terminal_uuid,
                            parameters,
                            args,
                        ) in rules:
                            possibilities.append(
                                RHSRule(
                                    binder,
                                    [
                                        self.uuid_to_pred[uuid]
                                        for uuid in predicates_uuid
                                    ],
                                    self.uuid_to_c[terminal_uuid],
                                    parameters,
                                    args,
                                )
                            )
                        memo.update({typ: possibilities})
                except UnpicklingError:
                    pass
                except KeyError:
                    pass

            for process in processes:
                process.terminate()

            return memo

    def inhabit(self, *targets: Type) -> ParameterizedTreeGrammar[Type, C, Predicate]:
        return self.multiinhabit(*targets, process_count=1)
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
