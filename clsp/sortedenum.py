from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar
from itertools import groupby
import itertools
import heapq
from timeit import default_timer

T = TypeVar("T")


def nexts(input: tuple[int, ...], length: int) -> Iterable[tuple[int, ...]]:
    """
    For a given tuple (a_1, a_2,...,a_n), compute the set:

      { (a_1 + 1, a_2,..., a_n)
      , (a_1, a_2 + 1,..., a_n)
      , ...
      , (a_1, a_2,..., a_n + 1) }
    """
    output = []
    for d in range(length):
        index_list = list(input)
        index_list[d] += 1
        output.append(tuple(index_list))
    return output


def sorted_product(
    *lists: Sequence[T], key: Callable[[T], int], combine: Callable[[Iterable[T]], T]
) -> Iterable[T]:
    """
    Lazily transforms the product of a set of sorted lists into a sorted output list.

    At its core, this computes the following statement lazily

    ```
    sorted(map(combine, product(*lists)), key=key)
    ````

    We do two assumptions on `lists`, `key` and `combine`:

      (1) All lists in `lists` need to be sorted according to `key`
      (2) for all (a_1, a_2, ... a_n) in product(*lists),

            key(a_1) + key(a_2) + ... + key(a_n) = key(combine(a_1, a_2, ... , a_n))


    The main idea is using a combination of BFS, heapsort and bucketsort, to efficiently traverse
    indices of the input lists.

    First we group the entries of the input lists by their value (according to the key). Since (2)
    holds, each entry in such a group can be treated the same.

    An index into the grouped lists, is a tuple of length `len(lists)`.  We can interpret the
    indices as a graph with, where the vertices are the indices, and there exists an edge A -> B,
    if B is increased by 1 at exactly one. We now apply a BFS to enumerate all indices.

    We start with (0,..,0) as the index of the guaranteed smallest combination of values. The next
    smallest value corresponds to the index of one of (0,...,0) successors. We mark these as
    visited, add the indices to a bucket marked with their respective values and store these values
    in a min-heap of possible smallest values.

    In each iteration, we extract the smallest value in this min-heap, add for each index in the
    corresponding bucket, we yield all combinations of the corresponding grouped lists and add
    the (unvisited) successors of those indices their corresponding bucket, mark them as visited
    and add their values to the heap.

    Since we grouped the input lists by value and the input lists are in ascending order, we know
    that each succeeding tuple of indices needs to yield a value grater than the current value.
    Since we always extract the lowest value, and enumerate all possible indices with the BFS
    approach, we get all possible combinations in ascending order.

    Args:
      *lists (Sequence[T]): Sorted input lists
      key (Callable[[T], int]): Key-Function to sort
      combine (Callable[[T, ...], T]): Function, that specifies how to combine the values from the
                                       input list to get a output value

    Yields:
      T: Combined values from the product of *lists in sorted order.

    Examples:
      This takes the lists [1,2,3], [4,5,6] and [7,8] and adds all possible combinations.
        * 1 + 4 + 7 = 12
        * 2 + 4 + 7 = 13
        * 1 + 5 + 7 = 13
        * 1 + 4 + 8 = 13
        * 2 + 5 + 7 = 14
        ...

      >>> print(list(sorted_product([1,2,3],[4,5,6],[7,8], key=lambda x:x, combine=sum)))
      [12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 17]
    """

    number_of_lists = len(lists)

    # First group the elements of the input list by size. The size is stored alongside the elements
    # for performance reasons.
    grouped_lists: list[list[tuple[int, tuple[T, ...]]]] = [
        [(val, tuple(group)) for val, group in groupby(lst, key=key)] for lst in lists
    ]

    # The buckets will contain the tuple of indices into `grouped_lists`.
    # The key of the buckets are the combined sizes.
    buckets: dict[int, list[tuple[int, ...]]] = {}

    # During the search, the neighbors of already returned values are stored in a min-heap
    possible_smallest_values = []

    # Only visit each index once
    visited = set()

    # The first index is (0,0,...,0). If any list does not have at least one entry, we return []
    initial_index = (0,) * number_of_lists
    try:
        smallest_value = sum(lst[0][0] for lst in grouped_lists)
    except IndexError:
        return []

    # initialize the buckets, visited list and min-heap
    buckets[smallest_value] = [initial_index]
    visited.add(initial_index)
    possible_smallest_values.append(smallest_value)

    # This ends, when there is no more data
    while True:
        # Get smallest value from the potential values
        try:
            smallest_value = heapq.heappop(possible_smallest_values)
        except IndexError:
            return

        # lookup all discovered indices corresponding to this value
        for index in buckets[smallest_value]:
            visited.remove(index)

            # return the all the combined values in `grouped_lists` corresponding to the respective
            # index
            yield from map(
                combine,
                (itertools.product(*map(lambda group, idx: group[idx][1], grouped_lists, index))),
            )

            # Find all neighbors of the smallest indices
            for next_index in nexts(index, number_of_lists):
                # only visit each index once
                if next_index not in visited:
                    visited.add(next_index)
                    try:
                        # Add value of the newly discovered index to their corresponding bucket
                        value = sum(
                            map(
                                lambda group, idx: group[idx][0],
                                grouped_lists,
                                next_index,
                            )
                        )
                        try:
                            buckets[value].append(next_index)
                        except KeyError:
                            heapq.heappush(possible_smallest_values, value)
                            buckets[value] = [next_index]

                    # If a subindex exceeds the length of their respective input list, skip that
                    # index
                    except IndexError:
                        continue

        # clean up
        del buckets[smallest_value]


def pure_bench() -> None:
    l1 = list(range(20))
    l2 = list(range(0, 4000, 10))
    l3 = [1] * 20 + [2] * 30
    l4 = [1] * 30

    lists = [l1, l2, l3, l4]  # , l3]
    _ = list(sorted_product(*lists, key=lambda x: x, combine=sum))


def main() -> None:
    l1 = list(range(10))
    l2 = list(range(0, 4000, 10))
    l3 = [1] * 20 + [2] * 30
    l4 = [1] * 30

    lists = [l1, l2, l3, l4]  # , l3]

    time1 = default_timer()
    _ = sorted(map(sum, itertools.product(*lists)), key=lambda x: x)
    tdelta1 = default_timer() - time1
    print(f"sorted(product(*lists)): {tdelta1}")
    time2 = default_timer()
    x1 = list(sorted_product(*lists, key=lambda x: x, combine=sum))
    tdelta2 = default_timer() - time2
    print(f"sorted_product_buckets(*lists): {tdelta2}")
    print(f"ratio: {tdelta2/tdelta1}")
    y1 = x1
    if y1 == sorted(y1):
        print(f"sorted, ({len(y1)})")

    # if len(x1) == len(s1):
    #     for elem in s1:
    #         if elem in x1:
    #             continue
    #         print(f"{elem} is not there :(")
    #         return


if __name__ == "__main__":
    main()
