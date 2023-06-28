# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.

from functools import partial
import itertools
from inspect import Parameter, signature, _ParameterKind, _empty
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, Optional, TypeAlias, TypeVar

S = TypeVar("S")  # non-terminals
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
    old_elements: set[S], new_elements: Iterable[S], max_count: int
) -> set[S]:
    """Return the union of old_elements and new_elements up to max_count elements as a new set."""

    result: set[S] = old_elements.copy()
    for element in new_elements:
        if len(result) >= max_count:
            return result
        elif element not in result:
            result.add(element)
    return result


def enumerate_terms(
    start: S,
    grammar: Mapping[S, Iterable[tuple[T, list[S]]]],
    max_count: Optional[int] = 100,
) -> Iterable[Tree[T]]:
    """Given a start symbol and a tree grammar, enumerate at most max_count ground terms derivable
    from the start symbol ordered by (depth, term size).
    """

    # accumulator for previously seen terms
    result: set[Tree[T]] = set()
    terms: dict[S, set[Tree[T]]] = {n: set() for n in grammar.keys()}
    terms_size: int = -1
    while terms_size < sum(len(ts) for ts in terms.values()):
        terms_size = sum(len(ts) for ts in terms.values())

        new_terms: Callable[
            [Iterable[tuple[T, list[S]]]], set[Tree[T]]
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


def group_by_tree_size(terms: Iterable[Tree[T]]) -> dict[int, set[Tree[T]]]:
    """Groups terms by tree_size as a dictionary mapping size to sets of terms."""

    result: dict[int, set[Tree[T]]] = dict()
    for term in terms:
        size = tree_size(term)
        ts = result.get(size, set())
        ts.add(term)
        result[size] = ts
    return result


def grouped_bounded_union(
    grouped_old_terms: dict[int, set[Tree[T]]],
    grouped_new_terms: dict[int, set[Tree[T]]],
    max_count: int,
    term_size: int,
) -> set[Tree[T]]:
    return set(
        itertools.chain.from_iterable(
            bounded_union(
                grouped_old_terms.get(i, set()),
                grouped_new_terms.get(i, set()),
                max_count,
            )
            for i in range(term_size + 1)
        )
    )


def enumerate_terms_of_size(
    start: S,
    grammar: Mapping[S, Iterable[tuple[T, list[S]]]],
    term_size: int,
    max_count: int,
) -> Iterable[Tree[T]]:
    """Given a start symbol, a tree grammar, and term size, enumerate at most max_count ground terms
    of specified term size derivable from the start symbol."""

    # accumulator for previously seen terms
    result: set[Tree[T]] = set()
    terms: dict[S, set[Tree[T]]] = {n: set() for n in grammar.keys()}
    terms_size: int = -1
    while terms_size < sum(len(ts) for ts in terms.values()):
        terms_size = sum(len(ts) for ts in terms.values())

        new_terms: Callable[
            [Iterable[tuple[T, list[S]]]], set[Tree[T]]
        ] = lambda exprs: {
            (c, tuple(args))
            for (c, ms) in exprs
            for args in itertools.product(*(terms[m] for m in ms))
        }

        terms = {
            n: terms[n]
            if len(terms[n]) >= max_count * (terms_size + 1)
            else grouped_bounded_union(
                group_by_tree_size(terms[n]),
                group_by_tree_size(new_terms(exprs)),
                max_count,
                term_size,
            )
            for (n, exprs) in grammar.items()
        }

        for term in terms[start]:
            # yield term if not seen previously
            if tree_size(term) == term_size and term not in result:
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
        parameters_of_c: Iterable[Parameter] = []
        current_combinator: partial[Any] | T | Callable[..., Any] = c

        if callable(current_combinator):
            try:
                parameters_of_c = list(
                    signature(current_combinator).parameters.values()
                )
            except ValueError:
                raise RuntimeError(
                    f"Combinator {c} does not expose a signature. "
                    "If it's a built-in, you can simply wrap it in another function."
                )

            if n == 0 and len(parameters_of_c) == 0:
                current_combinator = current_combinator()

        arguments = deque((results.pop() for _ in range(n)))

        while arguments:
            if not callable(current_combinator):
                raise RuntimeError(
                    f"Combinator {c} is applied to {n} argument(s), "
                    f"but can only be applied to {n - len(arguments)}"
                )

            use_partial = False

            simple_arity = len(
                list(filter(lambda x: x.default == _empty, parameters_of_c))
            )
            default_arity = len(
                list(filter(lambda x: x.default != _empty, parameters_of_c))
            )

            # if any parameter is marked as var_args, we need to use all available arguments
            pop_all = any(
                map(lambda x: x.kind == _ParameterKind.VAR_POSITIONAL, parameters_of_c)
            )

            # If a var_args parameter is found, we need to subtract it from the normal parameters.
            # Note: python does only allow one parameter in the form of *arg
            if pop_all:
                simple_arity -= 1

            # If a combinator needs more arguments than available, we need to use partial
            # application
            if simple_arity > len(arguments):
                use_partial = True

            fixed_parameters: deque[Any] = deque(
                arguments.popleft() for _ in range(min(simple_arity, len(arguments)))
            )

            var_parameters: deque[Any] = deque()
            if pop_all:
                var_parameters.extend(arguments)
                arguments = deque()

            default_parameters: deque[Any] = deque()
            for _ in range(default_arity):
                try:
                    default_parameters.append(arguments.popleft())
                except IndexError:
                    pass

            if use_partial:
                current_combinator = partial(
                    current_combinator,
                    *fixed_parameters,
                    *var_parameters,
                    *default_parameters,
                )
            else:
                current_combinator = current_combinator(
                    *fixed_parameters, *var_parameters, *default_parameters
                )

        results.append(current_combinator)
    return results.pop()


def test() -> None:
    d: Mapping[str, list[tuple[str, list[str]]]] = {
        "X": [("a", []), ("b", ["X", "Y"])],
        "Y": [("c", []), ("d", ["Y", "X"])],
    }
    # d = {
    #    "X": [("x", ["X1"])],
    #    "X1": [("x", ["X2"])],
    #    "X2": [("x", ["X3"])],
    #    "X3": [("x", ["X4"])],
    #    "X4": [("x", ["X5"])],
    #    "X5": [("x", ["Z"])],
    #    "X6": [("x", ["X7"])],
    #    "X7": [("x", ["X8"])],
    #    "X8": [("x", ["X9"])],
    #    "X9": [("x", ["Z"])],
    #    "Z": [("a", []), ("b", ["Z", "Y"])],
    #    "Y": [("c", []), ("d", ["Y", "Z"])],
    # }
    # d = {
    #    "X": [("a", []), ("b", ["Y", "Y", "Y"])],
    #    "Y": [("c", []), ("d", ["Z"])],
    #    "Z": [("e", [])],
    # }

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
