from collections import deque
from collections.abc import Sequence, Callable
from typing import TypeVar

S = TypeVar('S') # Type of Sets
E = TypeVar('E') # Type of Elements

def minimal_elements(elements: Sequence[E], compare: Callable[[E, E], bool]) -> Sequence[E] :
    """Enumerate minimal elements with respect to compare.
    
    `compare(e1, e2) == True` iff `e1` smaller or equal to `e2`.
    """

    candidates: deque[E] = deque(elements)
    result: deque[E] = deque()
    while candidates:
        new_candidates: deque[E] = deque()
        e1 = candidates.pop()
        while candidates:
            e2 = candidates.pop()
            if compare(e1, e2):
                continue # e2 is redundant
            elif compare(e2, e1):
                e1 = e2 # e1 is redundant
                candidates.extendleft(new_candidates)
                new_candidates.clear()
            else:
                new_candidates.appendleft(e2)
        candidates = new_candidates
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
    relevant_sets: deque[set[int]] = deque()
    for i in range(len(to_cover)):
        covering_sets = {j for j in range(len(sets)) if contains(sets[j], to_cover[i])}
        if len(covering_sets) == 0: # at least one element cannot be covered
            return []
        elif len(covering_sets) == 1: # exactly one set is relevant
            necessary_sets.add(covering_sets.pop())
        else: # more than one set is relevant
            relevant_sets.append(covering_sets)

    # collect minimal covers (there is no smaller or equivalent cover)
    covers: deque[set[int]] = deque()
    covers.appendleft(necessary_sets)
    for r in relevant_sets:
        partitioning = (deque(), deque())
        for c in covers:
            partitioning[c.isdisjoint(r)].append(c)

        covers = partitioning[0].copy()
        for c1 in partitioning[1]:
            js: set[int] = r.copy()
            for c2 in partitioning[0]:
                missing = c2.difference(c1)
                if len(missing) == 1:
                    # c2 is a subset of c1 + {missing element}
                    js.discard(missing.pop())
            for j in js:
                new_c = c1.copy()
                new_c.add(j)
                covers.append(new_c)
    return [[sets[j] for j in c] for c in covers]