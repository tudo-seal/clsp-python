from cls_python import CLSDecoder, CLSEncoder, FiniteCombinatoryLogic, Subtypes

from cls_python.types import Type, Omega, Constructor, Arrow, Intersection
from cls_python.subtypes import Subtypes
from cls_python.fcl import InhabitationResult, FiniteCombinatoryLogic, MultiArrow

from itertools import chain, combinations
from functools import reduce

type1: Type = Omega()
type2: Type = Omega()
type3: Type = Constructor("c")
type4: Type = Constructor("d")
type5: Type = Intersection(type3, type4)
repository: dict[object, Type] = dict[object, Type]({
  "C": type3,
  "D": type4})
environment: dict[object, set] = dict[object, set]()
subtypes: Subtypes = Subtypes(environment)

# cover machine tests

x: Type = Constructor("x")
y: Type = Constructor("y")
z: Type = Constructor("z")

sigma1: MultiArrow = ([Constructor("a")], Intersection(x, y))
sigma2: MultiArrow = ([Constructor("b")], Intersection(y, z))
sigma3: MultiArrow = ([Constructor("c")], Intersection(x, z))

to_cover: set[Type] = {x, y, z}
actual_type: Type = Intersection(x, Intersection(y, z))

combinator_type: list[list[MultiArrow]] = [[sigma1, sigma2, sigma3, sigma3]]

def print_multiarrow(ma : MultiArrow):
    for arg in ma[0]:
        print(arg, end="")
        print("->", end="")
    print(ma[1])

from cls_python.snippets import _compute_subqueries

# generic performance test

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# size of the problem input
max_size: int = 6

subsets = list(powerset(range(max_size)))

types1 = [Constructor(str(i)) for i in range(max_size)]
types2 = [Constructor(str(i) + "'") for i in range(max_size)]

types3 = list(map(lambda ts: reduce(Intersection, ts, Omega()), powerset(types1)))

to_cover: set[Type] = set(types1 + types2)
combinator_type: list[MultiArrow] = [([t], t) for t in types3 + types2]

import timeit

actual_type: Type = reduce(Intersection, to_cover)

start = timeit.default_timer()

covers2 = _compute_subqueries(fcl, combinator_type, actual_type)

stop = timeit.default_timer()

print('Time: ', stop - start)  

for c in covers2:
    print_multiarrow(c)



