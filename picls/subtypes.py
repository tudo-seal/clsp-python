from collections import deque
from collections.abc import Mapping
from typing import Optional

from .types import Arrow, Constructor, Intersection, Literal, Product, TVar, Type


class Subtypes:
    def __init__(self, environment: Optional[Mapping[str, set[str]]] = None):
        self.environment = {}
        if environment is not None:
            self.environment = self._transitive_closure(
                self._reflexive_closure(environment)
            )

    @staticmethod
    def env_closure(environment: Mapping[str, set[str]]) -> Mapping[str, set[str]]:
        return Subtypes._transitive_closure(Subtypes._reflexive_closure(environment))

    @staticmethod
    def _check_subtype_rec(
        subtypes: deque[Type],
        supertype: Type,
        substitutions: Mapping[str, Literal],
        environment: Mapping[str, set[str]],
    ) -> bool:
        if supertype.is_omega:
            return True
        match supertype:
            case Literal(name2, group2):
                while subtypes:
                    match subtypes.pop():
                        case Literal(name1, group1):
                            if name2 == name1 and group1 == group2:
                                return True
                        case TVar(name1):
                            if substitutions[name1] == supertype:
                                return True
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return False
            case Constructor(name2, arg2):
                casted_constr: deque[Type] = deque()
                while subtypes:
                    match subtypes.pop():
                        case Constructor(name1, arg1):
                            if name2 == name1 or name2 in environment.get(name1, {}):
                                casted_constr.append(arg1)
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return len(casted_constr) != 0 and Subtypes._check_subtype_rec(
                    casted_constr, arg2, substitutions, environment
                )
            case Arrow(src2, tgt2):
                casted_arr: deque[Type] = deque()
                while subtypes:
                    match subtypes.pop():
                        case Arrow(src1, tgt1):
                            if Subtypes._check_subtype_rec(
                                deque((src2,)), src1, substitutions, environment
                            ):
                                casted_arr.append(tgt1)
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return len(casted_arr) != 0 and Subtypes._check_subtype_rec(
                    casted_arr, tgt2, substitutions, environment
                )
            case Product(l2, r2):
                casted_l: deque[Type] = deque()
                casted_r: deque[Type] = deque()
                while subtypes:
                    match subtypes.pop():
                        case Product(l1, r1):
                            casted_l.append(l1)
                            casted_r.append(r1)
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return (
                    len(casted_l) != 0
                    and len(casted_r) != 0
                    and Subtypes._check_subtype_rec(
                        casted_l, l2, substitutions, environment
                    )
                    and Subtypes._check_subtype_rec(
                        casted_r, r2, substitutions, environment
                    )
                )
            case Intersection(l, r):
                return Subtypes._check_subtype_rec(
                    subtypes, l, substitutions, environment
                ) and Subtypes._check_subtype_rec(
                    subtypes, r, substitutions, environment
                )
            case TVar(name):
                while subtypes:
                    match subtypes.pop():
                        case Literal(value, group):
                            x = substitutions[name]
                            if x.value == value and x.group == group:
                                return True
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return False
            case _:
                raise TypeError(f"Unsupported type in check_subtype: {supertype}")

    @staticmethod
    def check_subtype(
        subtype: Type,
        supertype: Type,
        substitutions: Mapping[str, Literal],
        environment: Mapping[str, set[str]],
    ) -> bool:
        """Decides whether subtype <= supertype."""

        return Subtypes._check_subtype_rec(
            deque((subtype,)), supertype, substitutions, environment
        )

    @staticmethod
    def _reflexive_closure(env: Mapping[str, set[str]]) -> dict[str, set[str]]:
        all_types: set[str] = set(env.keys())
        for v in env.values():
            all_types.update(v)
        result: dict[str, set[str]] = {
            subtype: {subtype}.union(env.get(subtype, set())) for subtype in all_types
        }
        return result

    @staticmethod
    def _transitive_closure(env: Mapping[str, set[str]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {
            subtype: supertypes.copy() for (subtype, supertypes) in env.items()
        }
        has_changed = True

        while has_changed:
            has_changed = False
            for known_supertypes in result.values():
                for supertype in known_supertypes.copy():
                    to_add: set[str] = {
                        new_supertype
                        for new_supertype in result[supertype]
                        if new_supertype not in known_supertypes
                    }
                    if to_add:
                        has_changed = True
                    known_supertypes.update(to_add)

        return result

    # def minimize(self, tys: set[Type]) -> set[Type]:
    #     result: set[Type] = set()
    #     for ty in tys:
    #         if all(map(lambda ot: not self.check_subtype(ot, ty), result)):
    #             result = {ty, *(ot for ot in result if not self.check_subtype(ty, ot))}
    #     return result
