# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration." Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.

import itertools, functools
from collections import deque
from typing import Any, Iterable, Optional, TypeAlias, TypeVar, Callable

I = TypeVar("I")  # non-terminals
Tree: TypeAlias = tuple[object, tuple["Tree", ...]]


def tree_size(tree: Tree) -> int:
    """The number of nodes in a tree."""

    result = 0
    trees: deque[Tree] = deque((tree,))
    while trees:
        result += 1
        trees.extendleft(trees.pop()[1])
    return result


def bounded_union(old_elements: set, new_elements: Iterable, max_count: int) -> set:
    """Return the union of old_elements and new_elements up to max_count total elements as a new set."""

    result: set = old_elements.copy()
    for element in new_elements:
        if len(result) >= max_count:
            return result
        elif element not in result:
            result.add(element)
    return result


def enumerate_terms(
    start: I,
    grammar: dict[I, Iterable[tuple[object, list[I]]]],
    max_count: Optional[int] = 100,
) -> Iterable[Tree]:
    """Given a start symbol and a tree grammar, enumerate at most max_count ground terms derivable from the start symbol ordered by (depth, term size)."""

    # accumulator for previously seen terms
    result: set[Tree] = set()
    terms: dict[I, set[Tree]] = {n: set() for n in grammar.keys()}
    terms_size: int = -1
    while terms_size < sum(len(ts) for ts in terms.values()):
        terms_size = sum(len(ts) for ts in terms.values())
        new_terms = lambda exprs: {
            (c, tuple(args))
            for (c, ms) in exprs
            for args in itertools.product(*(terms[m] for m in ms))
        }
        if max_count is None:
            # new terms are built from previous terms according to grammar
            terms = {n: new_terms(exprs) for (n, exprs) in grammar.items()}
        else:
            terms = {
                n: terms[n]
                if len(terms[n]) >= max_count
                else bounded_union(
                    terms[n], sorted(new_terms(exprs), key=tree_size), max_count
                )
                for (n, exprs) in grammar.items()
            }
        for term in sorted(terms[start], key=tree_size):
            # yield term if not seen previously
            if term not in result:
                result.add(term)
                yield term

def interpret_term(term: Tree) -> Any:
    """Recursively evaluate given term."""

    terms: deque[Tree] = deque((term, ))
    combinators: deque[tuple[Callable, int]] = deque()
    # decompose terms
    while terms:
        t = terms.pop()
        combinators.append((t[0], len(t[1])))
        terms.extend(reversed(t[1]))
    results: deque[Any] = deque()
    # apply/call decomposed terms
    while combinators:
        (c, n) = combinators.pop()
        for _ in range(n):
            c = functools.partial(c, results.pop())

        results.append(c() if isinstance(c, Callable) else c)
        
    return results.pop()

def test():
    d = {"X": [("a", []), ("b", ["X", "Y"])], "Y": [("c", []), ("d", ["Y", "X"])]}
    # d = { "X" : [("x", ["X1"])], "X1" : [("x", ["X2"])], "X2" : [("x", ["X3"])], "X3" : [("x", ["X4"])], "X4" : [("x", ["X5"])], "X5" : [("x", ["Z"])],
    #      "X6" : [("x", ["X7"])], "X7" : [("x", ["X8"])], "X8" : [("x", ["X9"])], "X9" : [("x", ["Z"])],
    #      "Z" : [("a", []), ("b", ["Z", "Y"])], "Y" : [("c", []), ("d", ["Y", "Z"])] }
    # d = { "X" : [("a", []), ("b", ["Y", "Y", "Y"])], "Y" : [("c", []), ("d", ["Z"])], "Z" : [("e", [])] }

    import timeit

    start = timeit.default_timer()

    for i, r in enumerate(
        itertools.islice(enumerate_terms("X", d, max_count=100), 1000000)
    ):
        print(i, (r))

    print("Time: ", timeit.default_timer() - start)

def test2():
    class A(object):
        def __call__(self) -> str:
            return "A"

    class B(object):
        def __call__(self, a: str, b: str) -> str:
            return f"({a}) ->B-> ({b})"

    class C(object):
        def __call__(self) -> str:
            return "C"

    class D(object):
        def __call__(self, a: str, b: str) -> str:
            return f"({a}) ->D-> ({b})"

    d = { "X" : [(A(), []), (B(), ["X", "Y"])], "Y" : [(C(), []), (D(), ["Y", "X"])] }

    import timeit
    start = timeit.default_timer()

    for i,r in enumerate(itertools.islice(enumerate_terms("X", d, max_count = 100),1000000)):
        print(i, interpret_term(r))

    print('Time: ', timeit.default_timer() - start)

if __name__ == "__main__":
    test2()
