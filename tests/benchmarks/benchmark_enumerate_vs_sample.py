import timeit

from clsp.dsl import DSL
from clsp.enumeration import Tree
from clsp.fcl import Contains

from clsp.types import Constructor, Literal,  LVar

from clsp.search import Enumerate, RandomSample

from typing import Any

# to allow infinite literal groups, we need to define a subclass of Contains for that group
class Nat(Contains):
    def __contains__(self, value: object) -> bool:
        return isinstance(value, int) and value >= 0

# Our Delta will contain Booleans as the two elementary set and natural numbers as an infinite set (good for indexing).
base_delta: dict[str, list[Any]] = {"nat": Nat(),
                                    "bool": [True, False]}

delta = base_delta | {
    "dimension": [n for n in range(0, 3)]
}

gamma = {
    "edge": DSL()
    .In(Constructor("graph",
                    Constructor("input", Literal(1, "dimension")) &
                    Constructor("output", Literal(1, "dimension")) &
                    Constructor("size", Literal(1, "nat")))), # Constructor("size", Literal(0, "nat")))),
    "vertex": DSL()
    .Use("m", "dimension")
    .Use("n", "dimension")
    .In(Constructor("graph",
                    Constructor("input", LVar("m")) &
                    Constructor("output", LVar("n")) &
                    Constructor("size", Literal(1, "nat")))),
    "beside": DSL()
    .Use("m", "dimension")
    .Use("n", "dimension")
    .Use("i", "dimension")
    .Use("o", "dimension")
    .Use("p", "dimension")
    .As(lambda m, i: i - m)
    .Use("q", "dimension")
    .As(lambda n, o: o - n)
    .Use("s3", "nat")
    .Use("s1", "nat")
    .As(lambda s3: {x for x in range(0, s3+1)}, multi_value=True)
    .Use("s2", "nat")
    .As(lambda s3, s1: s3 - s1)
    .Use("x", Constructor("graph",
                    Constructor("input", LVar("m")) &
                    Constructor("output", LVar("n")) &
                    Constructor("size",  LVar("s1"))))
    .Use("y", Constructor("graph",
                    Constructor("input", LVar("p")) &
                    Constructor("output", LVar("q")) &
                    Constructor("size",  LVar("s2"))))
    .In(Constructor("graph",
                    Constructor("input", LVar("i")) &
                    Constructor("output", LVar("o")) &
                    Constructor("size",  LVar("s3")))),
    "before": DSL()
    .Use("m", "dimension")
    .Use("n", "dimension")
    .Use("p", "dimension")
    .Use("s3", "nat")
    .Use("s1", "nat")
    .As(lambda s3: {x for x in range(0, s3+1)}, multi_value=True)
    .Use("s2", "nat")
    .As(lambda s3, s1: s3 - s1)
    .Use("x", Constructor("graph",
                    Constructor("input", LVar("m")) &
                    Constructor("output", LVar("n")) &
                    Constructor("size",  LVar("s1"))))
    .Use("y", Constructor("graph",
                    Constructor("input", LVar("n")) &
                    Constructor("output", LVar("p")) &
                    Constructor("size",  LVar("s2"))))
    .In(Constructor("graph",
                    Constructor("input", LVar("m")) &
                    Constructor("output", LVar("p")) &
                    Constructor("size",  LVar("s3")))),
    "swap": DSL()
    .Use("io", "dimension")
    .Use("m", "dimension")
    .With(lambda io, m: 0 < m < io) # swapping zero connections is neutral
    .Use("n", "dimension")
    .As(lambda io, m: io - m)
    .In(Constructor("graph",
                    Constructor("input", LVar("io")) &
                    Constructor("output", LVar("io")) &
                    Constructor("size", Literal(1, "nat")))), # Constructor("size", Literal(0, "nat")))),
    "copy": DSL()
    .Use("m", "dimension")
    .With(lambda m: m > 0)
    .Use("i", "dimension")
    .Use("o", "dimension")
    .Use("p", "dimension")
    .As(lambda m, i: i // m)
    .Use("q", "dimension")
    .As(lambda m, o: o // m)
    .Use("s2", "nat")
    .Use("s1", "nat")
    .As(lambda m, s2: s2 // m)
    .Use("x", Constructor("graph",
                    Constructor("input", LVar("p")) &
                    Constructor("output", LVar("q")) &
                    Constructor("size", LVar("s1"))))
    .In(Constructor("graph",
                    Constructor("input", LVar("i")) &
                    Constructor("output", LVar("o")) &
                    Constructor("size", LVar("s2")))),
}

def easy_subtree_kernel(t1: Tree[Any, str], t2: Tree[Any, str]) -> float:
    """
    Computes the kernel of two trees.
    :param t1: The first tree.
    :param t2: The second tree.
    :return: The distance between the two trees.
    """
    subtrees1: list[Tree[Any, str]] = list(map(lambda x: x[0], t1.subtrees([])))
    subtrees2: list[Tree[Any, str]] = list(map(lambda x: x[0], t2.subtrees([])))
    mean_len = (len(subtrees1) + len(subtrees2)) / 2
    shared: int = 0
    occur1: dict[Tree[Any, str], int] = {}
    for t in subtrees1:
        k = occur1.get(t)
        if k is None:
            occur1[t] = 1
        else:
            occur1[t] = k + 1
    occur2: dict[Tree[Any, str], int] = {}
    for t in subtrees2:
        k = occur2.get(t)
        if k is None:
            occur2[t] = 1
        else:
            occur2[t] = k + 1
    for t, n in occur1.items():
        k = occur2.get(t)
        if k is not None:
            shared = shared + min(n, k)
    return 1 - (shared / mean_len)

target = Constructor("graph",
                     Constructor("input", Literal(0, "dimension")) &
                     Constructor("output", Literal(0, "dimension")) &
                     Constructor("size", Literal(5, "nat"))
                     )

n = 100
print(f"target: {target}")
print("start enumerate")
start_enum = timeit.default_timer()
enum = Enumerate(gamma, delta, target)
print(f"Inhabitation took {timeit.default_timer() - start_enum} seconds")
grammar = [(nt, rhs) for nt, deque in enum.grammar.as_tuples() for rhs in deque]
print(f"The grammar has {len(grammar)} rules")
start_enum = timeit.default_timer()
result_enum = list(enum.sample(n))
print(f"Enumeration of {n} terms took {timeit.default_timer() - start_enum} seconds")

enum_sizes = [t.size for t in result_enum]
print(f"""
The average size of the enumerated terms is {sum(enum_sizes) / len(enum_sizes)}.
The maximum size of an enumerated term is {max(enum_sizes)}.
The minimum size of an enumerated term is {min(enum_sizes)}.
""")

enum_distances = list([easy_subtree_kernel(t1, t2) for t1 in result_enum for t2 in result_enum if t1 != t2])
print(f"""
Regarding the easy_subtree_kernel:
The average distance between enumerated terms is {sum(enum_distances) / len(enum_distances)}.
The maximum distance between enumerated terms is {max(enum_distances)}.
The minimum distance between enumerated terms is {min(enum_distances)}.
""")

print("start random")
start_random = timeit.default_timer()
random = RandomSample(gamma, delta, target)
print(f"Inhabitation + grammar annotations took {timeit.default_timer() - start_random} seconds")
start_random = timeit.default_timer()
result_random = list(random.sample(n))
print(f"Random sampling of {n} terms  took {timeit.default_timer() - start_random} seconds")

random_sizes = [t.size for t in result_random]
print(f"""
The average size of the random sampled terms is {sum(random_sizes) / len(random_sizes)}.
The maximum size of an random sampled term is {max(random_sizes)}.
The minimum size of an random sampled term is {min(random_sizes)}.
""")

random_distances = list([easy_subtree_kernel(t1, t2) for t1 in result_random for t2 in result_random if t1 != t2])
print(f"""
Regarding the easy_subtree_kernel:
The average distance between random sampled terms is {sum(random_distances) / len(random_distances)}.
The maximum distance between random sampled terms is {max(random_distances)}.
The minimum distance between random sampled terms is {min(random_distances)}.
""")

