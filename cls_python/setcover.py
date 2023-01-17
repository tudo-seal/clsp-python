from collections import deque
from collections.abc import Sequence, Callable
from typing import TypeVar

S = TypeVar('S') # Type of Sets
E = TypeVar('E') # Type of Elements

def minimal_elements(elements: Sequence[E], compare: Callable[[E, E], bool]) -> Sequence[E] :
    """Enumerate minimal elements with respect to compare.
    
    `compare(e1, e2) == True` iff `e1` smaller or equal to `e2`.
    """

    result: deque[E] = deque()
    candidates: deque[E] = deque(elements)
    while candidates:
        e1 = candidates.pop()
        if (all(not compare(e2, e1) for e2 in result + candidates)):
            result.appendleft(e1)
    return result


def minimal_covers(sets: list[S], to_cover: list[E], contains: Callable[[S, E], bool]) -> list[list[S]]:
    """List minimal covers of elements in to_cover using given sets.
    
    Properties of each `cover: list[S]`
    - for every `e: E` in `to_cover` there is at least one `s: S` in `cover` such that `contains(s, e) == True`
    - no `s: S` can be removed from `cover`
    """
    # sets necessarily included in any cover
    necessary_sets: set[int] = set()
    # for each element e: list of sets containing e
    relevant_sets: deque[list[int]] = deque()
    for i in range(len(to_cover)):
        covering_sets = [j for j in range(len(sets)) if contains(sets[j], to_cover[i])]
        if len(covering_sets) == 0: # at least one element cannot be covered
            return []
        elif len(covering_sets) == 1: # exactly one set is relevant
            necessary_sets.add(covering_sets[0])
        relevant_sets.append(covering_sets)

    # overapproximate covers (not necessarily minimal)
    covers: deque[set[int]] = deque()
    covers.appendleft(necessary_sets)
    for r in relevant_sets:
        new_covers: deque[set[int]] = deque()
        for c in covers:
            if c.isdisjoint(r):
                for j in r:
                    new_c = c.copy()
                    new_c.add(j)
                    new_covers.append(new_c)
            else: # element is already covered by c
                new_covers.append(c)
        covers = new_covers

    # collect minimal covers (there is no smaller or equivalent cover)
    result = minimal_elements(covers, lambda e1, e2: e1.issubset(e2))
    return [[sets[j] for j in c] for c in result]
