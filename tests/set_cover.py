from bcls.combinatorics import minimal_covers, maximal_elements
from itertools import combinations
from collections import deque
from random import randrange

# TODO: testing framework
contains = lambda s, e: e in s
sets: list[list[int]] = [[1, 4], [7, 3], [7, 9], [0, 1, 2], [1, 3], [2, 3], [1, 6]]
elements: list[int] = [0, 1, 2, 3, 7]
print(minimal_covers(sets, elements, contains))


# naive minimal set cover implementation
def naive_minimal_covers(sets, to_cover, contains):
    covers = deque()
    for i in range(len(sets) + 1):
        for cover in combinations(range(len(sets)), i):
            if all(any(contains(sets[s], e) for s in cover) for e in to_cover):
                covers.append(cover)
    covers = list(map(list, covers))
    covers = maximal_elements(covers, lambda c1, c2: all(s in c1 for s in c2))
    return [[sets[i] for i in c] for c in covers]


def equivalent_lists(l1, l2):
    return len(l1) == len(l2) and all(e in l2 for e in l1) and all(e in l1 for e in l2)


def equivalent_covers(covers1, covers2):
    return (
        len(covers1) == len(covers2)
        and all(any(equivalent_lists(c1, c2) for c2 in covers2) for c1 in covers1)
        and all(any(equivalent_lists(c1, c2) for c1 in covers1) for c2 in covers2)
    )


print(naive_minimal_covers(sets, elements, contains))
print(
    equivalent_covers(
        minimal_covers(sets, elements, contains),
        naive_minimal_covers(sets, elements, contains),
    )
)


# [7, 8, 9, 5, 3, 4, 1, 4] by [[], [5], [], [5, 5, 0, 0, 1, 4, 5, 4, 1], [5], [7, 7, 9, 3, 7, 0, 7, 7, 0, 3, 8, 4], [4, 6, 8, 7, 1, 9, 2, 8, 6], [7, 8, 9, 4, 1, 9, 3, 5], [8, 5]]
elements = [7, 8, 9, 5, 3, 4, 1, 4]
sets = [
    [],
    [5],
    [],
    [5, 5, 0, 0, 1, 4, 5, 4, 1],
    [5],
    [7, 7, 9, 3, 7, 0, 7, 7, 0, 3, 8, 4],
    [4, 6, 8, 7, 1, 9, 2, 8, 6],
    [7, 8, 9, 4, 1, 9, 3, 5],
    [8, 5],
]
print(minimal_covers(sets, elements, contains))
print(naive_minimal_covers(sets, elements, contains))
print(
    equivalent_covers(
        minimal_covers(sets, elements, contains),
        naive_minimal_covers(sets, elements, contains),
    )
)

# equivalence exhaustive check
max_elements = 10
max_sets = 10


def random_set() -> list[int]:
    num_elements = randrange(2 * max_elements)
    return [randrange(max_elements) for i in range(num_elements)]


current_index = 0
for _ in range(100000):
    if current_index % 10000 == 0:
        print(current_index)
    current_index += 1
    sets = [random_set() for _ in range(randrange(max_sets))]
    elements = random_set()
    covers1 = minimal_covers(sets, elements, contains)
    covers2 = naive_minimal_covers(sets, elements, contains)
    if not equivalent_covers(covers1, covers2):
        print(f"cover {elements} by {sets}")
        print(covers1)
        print(covers2)
        print("COVERS NOT EQUIVALENT")
        break
