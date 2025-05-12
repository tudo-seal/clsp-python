import timeit

from clsp.dsl import DSL
from clsp.enumeration import Tree
from clsp.fcl import Contains

from clsp.types import Constructor, Literal,  LVar

from clsp.search import Enumerate, RandomSample

from typing import Any


"""
A repository for directed acyclic graphs, 
following "An initial algebra approach to directed acyclic graphs" from Jeremy Gibbons.
The following algebraic laws are defined on DAGs (if we skip arguments, 
the variables are only the term arguments and the literals that are independent of the term arguments):

beside(x, beside(y,z)) = beside(beside(x,y),z)  (associativity of beside)


before(x, before(y,z)) = before(before(x,y),z)  (associativity of before)


beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p'))) 
=                                                                               (abiding law)
before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))     


swap(n, n, 0) and swap(n, 0, n) are neutral elements and therefore make no difference.

                                                                                            (swap simplification laws)

before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
=
swap(m + n + p, m, n+p)


before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q))) 
=                                                                                 (swap law)
beside(y(m,q),x(n,p))


before(swap(m+n, m, n), swap(n+m, n, m)) = copy(m+n, edge())                  (simplified swap law)


before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
=                                                                                            (derived simplification law)
swap(m + n + p, m+n, p)


These laws will be interpreted as directed equalities, such that they correspond to the following
term rewriting system:

beside(beside(x,y),z) 
-> 
beside(x, beside(y,z))

before(before(x,y),z) 
-> 
before(x, before(y,z))

beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p'))) 
-> 
before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p'))) 

before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p))) 
-> 
swap(m + n + p, m, n+p)

before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q))) 
->
beside(y(m,q),x(n,p))

before(swap(m+n, m, n), swap(n+m, n, m)) 
-> 
copy(m+n, edge()) 

before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
->
swap(m + n + p, m+n, p)

additionally any sequence of beside applied to the same term will be rewritten to copy this term to the length of this sequence:

beside(x, beside(x,y)) 
-> beside(copy(2, x), y)

beside(copy(n, x), beside(copy(m, x), y))
->
beside(copy(n+m, x), y)


From our FSCD-paper "Restricting tree grammars with term rewriting systems" we know,
that it should be sufficient to define a term-predicate that forbids all left-hand sides of the rules,
to describe the set of combinatory terms, that are normal forms of the term rewriting system.
"""

def recognize_beside_associativity(term: Tree[Any, str]) -> bool:
    """
    Recognizes the left-hand side of the beside associativity rewriting rule.
    :param term: The term to check.
    :return: True if the term is a left-hand side of a rule, False otherwise.
    """
    # beside(beside(x,y),z)
    children: list[Tree[Any, str]] = list(term.children)
    non_literal_children: list[Tree[Any, str]] = [c for c in children if not c.is_literal]
    if not term.root == "beside":
        return all([recognize_beside_associativity(t) for t in non_literal_children])
    assert(
        len(non_literal_children) == 2
   )
    left_term = non_literal_children[0]
    if left_term.root == "beside":
        return True
    return all([recognize_beside_associativity(t) for t in non_literal_children])

def recognize_before_associativity(term: Tree[Any, str]) -> bool:
    """
    Recognizes the left-hand side of the before associativity rewriting rule.
    :param term: The term to check.
    :return: True if the term is a left-hand side of a rule, False otherwise.
    """
    # before(before(x,y),z)
    children: list[Tree[Any, str]] = list(term.children)
    non_literal_children: list[Tree[Any, str]] = [c for c in children if not c.is_literal]
    if not term.root == "before":
        return all([recognize_before_associativity(t) for t in non_literal_children])
    assert(
        len(non_literal_children) == 2
        )
    left_term = non_literal_children[0]
    if left_term.root == "before":
        return True
    return all([recognize_before_associativity(t) for t in non_literal_children])

def recognize_abiding(term: Tree[Any, str]) -> bool:
    """
    Recognizes the left-hand side of the abiding rewriting rule.
    :param term: The term to check.
    :return: True if the term is a left-hand side of a rule, False otherwise.
    """
    # beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
    children: list[Tree[Any, str]] = list(term.children)
    non_literal_children: list[Tree[Any, str]] = [c for c in children if not c.is_literal]
    # TODO
    return False


def recognize_left_hand_sides(term: Tree[Any, str]) -> bool:
    """
    Recognizes the left-hand sides of the rules.
    :param term: The term to check.
    :return: True if the term is a left-hand side of a rule, False otherwise.
    """
    # TODO
    return True

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
    :return: The number n of shared subtrees of the two trees divided through the mean amount of subtrees.
    """
    subtrees1: list[Tree[Any, str]] = list(map(lambda x: x[0], t1.subtrees([])))
    subtrees2: list[Tree[Any, str]] = list(map(lambda x: x[0], t2.subtrees([])))
    mean_len = len(subtrees1) + len(subtrees2) / 2
    shared: int = 0
    for t in subtrees1:
        if t in subtrees2:
            shared += 1
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

enum_distances = list(filter(lambda n: n != 1, [easy_subtree_kernel(t1, t2) for t1 in result_enum for t2 in result_enum]))
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

random_distances = list(filter(lambda n: n != 1, [easy_subtree_kernel(t1, t2) for t1 in result_random for t2 in result_random]))
print(f"""
Regarding the easy_subtree_kernel:
The average distance between random sampled terms is {sum(random_distances) / len(random_distances)}.
The maximum distance between random sampled terms is {max(random_distances)}.
The minimum distance between random sampled terms is {min(random_distances)}.
""")

"""
Ausgabe bei der edges und swap noch size 0 hatten:

target: graph(input([0, dimension]) & output([0, dimension]) & size([3, nat]))
start enumerate
Inhabitation took 0.11470774607732892 seconds
The grammar has 318 rules
Enumeration of 100 terms took 0.09271134901791811 seconds

The average size of the enumerated terms is 30.07.
The maximum size of an enumerated term is 42.
The minimum size of an enumerated term is 11.


Regarding the easy_subtree_kernel:
The average distance between enumerated terms is 0.8012879187976645.
The maximum distance between enumerated terms is 0.9777777777777777.
The minimum distance between enumerated terms is 0.2857142857142857.

start random
Inhabitation + grammar annotations took 0.9225658050272614 seconds
Random sampling of 100 terms  took 4.5936380659695715 seconds

The average size of the random sampled terms is 688.49.
The maximum size of an random sampled term is 18953.
The minimum size of an random sampled term is 11.


Regarding the easy_subtree_kernel:
The average distance between random sampled terms is 0.8104361872835941.
The maximum distance between random sampled terms is 0.9998418889006008.
The minimum distance between random sampled terms is 0.09579560828390055.
"""

"""
Ausgaben, in denen edges und swap size 1 haben:

target: graph(input([0, dimension]) & output([0, dimension]) & size([3, nat]))
start enumerate
Inhabitation took 0.1275391650851816 seconds
The grammar has 245 rules
Enumeration of 100 terms took 0.07822113204747438 seconds

The average size of the enumerated terms is 33.14.
The maximum size of an enumerated term is 50.
The minimum size of an enumerated term is 11.


Regarding the easy_subtree_kernel:
The average distance between enumerated terms is 0.7722083826247406.
The maximum distance between enumerated terms is 0.9583333333333334.
The minimum distance between enumerated terms is 0.2558139534883721.

start random
Inhabitation + grammar annotations took 0.6216761311516166 seconds
Random sampling of 100 terms  took 0.1908578600268811 seconds

The average size of the random sampled terms is 50.04.
The maximum size of an random sampled term is 154.
The minimum size of an random sampled term is 11.


Regarding the easy_subtree_kernel:
The average distance between random sampled terms is 0.7809099935217045.
The maximum distance between random sampled terms is 0.984375.
The minimum distance between random sampled terms is 0.19277108433734935.


####################

target: graph(input([0, dimension]) & output([0, dimension]) & size([5, nat]))
start enumerate
Inhabitation took 0.23601386696100235 seconds
The grammar has 722 rules
Enumeration of 100 terms took 0.17270648619160056 seconds

The average size of the enumerated terms is 35.36.
The maximum size of an enumerated term is 49.
The minimum size of an enumerated term is 19.


Regarding the easy_subtree_kernel:
The average distance between enumerated terms is 0.8472608453580226.
The maximum distance between enumerated terms is 0.9848484848484849.
The minimum distance between enumerated terms is 0.30645161290322576.

start random
Inhabitation + grammar annotations took 5.934273890918121 seconds
Random sampling of 100 terms  took 0.8865617769770324 seconds

The average size of the random sampled terms is 83.05.
The maximum size of an random sampled term is 169.
The minimum size of an random sampled term is 19.


Regarding the easy_subtree_kernel:
The average distance between random sampled terms is 0.8168042805708999.
The maximum distance between random sampled terms is 0.9878048780487805.
The minimum distance between random sampled terms is 0.24390243902439024.
"""
