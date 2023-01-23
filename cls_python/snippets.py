
from cls_python.fcl import MultiArrow
from cls_python.combinatorics import minimal_covers, minimal_elements
from cls_python.types import Omega, Type, Arrow, Intersection
from cls_python.fcl import FiniteCombinatoryLogic

from functools import reduce
from collections.abc import Sequence
from collections import deque

def _multiarrows_from_type(type: Type) -> list[list[MultiArrow]] :
    def _split_ty(type: Type) -> tuple[Type, Sequence[tuple[Type, Type]]] :
        """Splits a type into a nullary part and a list of (src, tgt) pairs."""

        nullary: deque[Type] = deque()
        unary: deque[tuple[Type, Type]] = deque()
        types: deque[Type] = deque((type))
        while types:
            current = types.pop()
            if current.is_omega:
                continue
            match current:
                case Arrow(src, tgt):
                    unary.appendleft((src, tgt))
                case Intersection(sigma, tau):
                    types.extend(sigma, tau)
                case _:
                    nullary.appendleft(current)
        return (reduce(Intersection, nullary, Omega()), unary)

    result = []
    while not type.is_omega:
        type, arrs = _split_ty(type)
        
    current = type
    _split_ty
    

def _compare_multiarrow_arguments(fcl: FiniteCombinatoryLogic, lesser: MultiArrow, greater: MultiArrow) -> bool:
    """`True` if each argument in `lesser` multi-arrow is smaller or equal to the corresponding argument in `greater` multi-arrow."""
    return (len(lesser[0]) == len(greater[0])
            and all(fcl.subtypes.check_subtype(lesser_arg, greater_arg)
                    for (lesser_arg, greater_arg) in zip(lesser[0], greater[0])))
                    
def _compute_subqueries(fcl: FiniteCombinatoryLogic, combinator_type: list[MultiArrow], target: Type) -> Sequence[MultiArrow] :
    """(TODO improve description) Compute possibilities to inhabit target type using  combinator_type.
    
    target cannot be equivalent to omega!
    """

    sets: list[MultiArrow] = combinator_type
    elements: list[Type] = list(target.organized)
    contains = lambda m, t: fcl.subtypes.check_subtype(m[1], t)
    # possibilities to cover target using targets of multi-arrows in combinator_type
    covers: list[list[MultiArrow]] = minimal_covers(sets, elements, contains)
    # merge multi-arrows in each cover
    accumulated_covers = map(lambda ms: reduce(fcl._merge_multi_arrow, ms), covers)
    compare_ma_args = lambda m1, m2: _compare_multiarrow_arguments(fcl, m2, m1)
    # remove redundant multi-arrows
    return minimal_elements(accumulated_covers, compare_ma_args)
