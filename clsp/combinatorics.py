from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Generic, TypeGuard, TypeVar, cast

S = TypeVar("S")  # Type of Sets
E = TypeVar("E")  # Type of Elements
E1 = TypeVar("E1")
E2 = TypeVar("E2")


class typed_partitioner(Generic[E1, E2]):
    def __init__(self, predicate: Callable[[E1 | E2], TypeGuard[E2]]) -> None:
        self.predicate = predicate

    def partition(self, elements: Iterable[E1 | E2]) -> tuple[deque[E1], deque[E2]]:
        partitioning: tuple[deque[E1], deque[E2]] = (deque(), deque())
        for element in elements:
            if self.predicate(element):
                partitioning[1].append(element)
            else:
                partitioning[0].append(cast(E1, element))
        return partitioning


def partition(predicate: Callable[[E], bool], elements: Iterable[E]) -> tuple[deque[E], deque[E]]:
    """Partition elements of an Iterable according to a predicate. Narrowing types.

    Returns: (elements not satisfying predicate, elements satisfying predicate)."""

    partitioning: tuple[deque[E], deque[E]] = (deque(), deque())
    for element in elements:
        if predicate(element):
            partitioning[1].append(element)
        else:
            partitioning[0].append(element)
    return partitioning


def maximal_elements(elements: Iterable[E], compare: Callable[[E, E], bool]) -> Sequence[E]:
    """Enumerate maximal elements with respect to compare.

    `compare(e1, e2) == True` iff `e1` smaller or equal to `e2`.
    """

    candidates: deque[E] = deque(elements)
    result: deque[E] = deque()
    while candidates:
        new_candidates: deque[E] = deque()
        e1 = candidates.pop()
        while candidates:
            e2 = candidates.pop()
            if compare(e2, e1):
                continue  # e2 is redundant
            elif compare(e1, e2):
                e1 = e2  # e1 is redundant
                candidates.extendleft(new_candidates)
                new_candidates.clear()
            else:
                new_candidates.appendleft(e2)
        candidates = new_candidates
        result.appendleft(e1)
    return result


def minimal_covers(
    sets: Sequence[S], to_cover: Sequence[E], contains: Callable[[S, E], bool]
) -> list[list[S]]:
    """List minimal covers of elements in to_cover using given sets.

    Properties of each `cover: list[S]`
    - for every `e: E` in `to_cover` there is at least one `s: S` in `cover` such that
      `contains(s, e) == True`
    - no `s: S` can be removed from `cover`
    """
    # sets necessarily included in any cover
    necessary_sets: set[int] = set()
    # for each element e: sets containing e
    relevant_sets: deque[set[int]] = deque()
    for i in range(len(to_cover)):
        covering_sets = {j for j in range(len(sets)) if contains(sets[j], to_cover[i])}
        if len(covering_sets) == 0:  # at least one element cannot be covered
            return []
        elif len(covering_sets) == 1:  # exactly one set is relevant
            necessary_sets.add(covering_sets.pop())
        else:  # more than one set is relevant
            relevant_sets.append(covering_sets)

    # collect minimal covers (there is no smaller or equivalent cover)
    covers: deque[set[int]] = deque()
    covers.appendleft(necessary_sets)
    for r in relevant_sets:
        partitioning = partition(r.isdisjoint, covers)
        covers = partitioning[0].copy()
        for c1 in partitioning[1]:
            js: set[int] = r.copy()
            for c2 in partitioning[0]:
                missing = c2.difference(c1)
                if len(missing) == 1:
                    # c2 is a subset of c1 + {one missing element}
                    js.discard(missing.pop())
            for j in js:
                new_c = c1.copy()
                new_c.add(j)
                covers.append(new_c)
    return [[sets[j] for j in c] for c in covers]
