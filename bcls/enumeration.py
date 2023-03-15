# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration." Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.

from functools import partial
import itertools
from inspect import signature, _ParameterKind
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, Optional, TypeAlias, TypeVar

I = TypeVar("I")  # non-terminals
T = TypeVar("T", bound=Hashable)

Tree: TypeAlias = tuple[T, tuple["Tree[T]", ...]]


def tree_size(tree: Tree[T]) -> int:
    """The number of nodes in a tree."""

    result = 0
    trees: deque[Tree[T]] = deque((tree,))
    while trees:
        result += 1
        trees.extendleft(trees.pop()[1])
    return result


def bounded_union(
    old_elements: set[I], new_elements: Iterable[I], max_count: int
) -> set[I]:
    """Return the union of old_elements and new_elements up to max_count total elements as a new set."""

    result: set[I] = old_elements.copy()
    for element in new_elements:
        if len(result) >= max_count:
            return result
        elif element not in result:
            result.add(element)
    return result


def enumerate_terms(
    start: I,
    grammar: Mapping[I, Iterable[tuple[T, list[I]]]],
    max_count: Optional[int] = 100,
) -> Iterable[Tree[T]]:
    """Given a start symbol and a tree grammar, enumerate at most max_count ground terms derivable from the start symbol ordered by (depth, term size)."""

    # accumulator for previously seen terms
    result: set[Tree[T]] = set()
    terms: dict[I, set[Tree[T]]] = {n: set() for n in grammar.keys()}
    terms_size: int = -1
    while terms_size < sum(len(ts) for ts in terms.values()):
        terms_size = sum(len(ts) for ts in terms.values())

        new_terms: Callable[
            [Iterable[tuple[T, list[I]]]], set[Tree[T]]
        ] = lambda exprs: {
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


def interpret_term(term: Tree[T]) -> Any:
    """Recursively evaluate given term."""

    terms: deque[Tree[T]] = deque((term,))
    combinators: deque[tuple[T, int]] = deque()
    # decompose terms
    while terms:
        t = terms.pop()
        combinators.append((t[0], len(t[1])))
        terms.extend(reversed(t[1]))
    results: deque[Any] = deque()

    # apply/call decomposed terms
    while combinators:
        (c, n) = combinators.pop()
        current_combinator: partial[Any] | T | Callable[..., Any] = c
        arguments = deque((results.pop() for _ in range(n)))

        while arguments:
            if not callable(current_combinator):
                raise RuntimeError(
                    f"Combinator {c} is applied to {n} argument(s), but can only be applied to {n - len(arguments)}"
                )
            arity_of_c = len(signature(current_combinator).parameters)

            if any(
                (parameter.kind == _ParameterKind.VAR_POSITIONAL)
                for parameter in signature(current_combinator).parameters.values()
            ):
                arity_of_c = len(arguments)

            partial_arguments = deque(arguments.popleft() for _ in range(arity_of_c))
            current_combinator = current_combinator(*partial_arguments)

        results.append(current_combinator)
    return results.pop()


def test() -> None:
    d: Mapping[str, list[tuple[str, list[str]]]] = {
        "X": [("a", []), ("b", ["X", "Y"])],
        "Y": [("c", []), ("d", ["Y", "X"])],
    }
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


def test2() -> None:
    class A:
        def __call__(self) -> str:
            return "A"

    class B:
        def __call__(self, a: str, b: str) -> str:
            return f"({a}) ->B-> ({b})"

    class C:
        def __call__(self) -> str:
            return "C"

    class D:
        def __call__(self, a: str, b: str) -> str:
            return f"({a}) ->D-> ({b})"

    d: dict[str, list[tuple[A | B | C | D | str, list[str]]]] = {
        "X": [(A(), []), (B(), ["X", "Y"]), ("Z", [])],
        "Y": [(C(), []), (D(), ["Y", "X"])],
    }

    import timeit

    start = timeit.default_timer()

    for i, r in enumerate(
        itertools.islice(enumerate_terms("X", d, max_count=100), 1000000)
    ):
        print(i, interpret_term(r))

    print("Time: ", timeit.default_timer() - start)


if __name__ == "__main__":
    test2()
