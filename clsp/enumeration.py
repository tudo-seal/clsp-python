# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.

from functools import partial
import itertools
from inspect import Parameter, signature, _ParameterKind, _empty
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, Generic, Optional, TypeAlias, TypeVar, cast
from heapq import merge
from queue import PriorityQueue
from dataclasses import dataclass, field

from .grammar import (
    GVar,
    ParameterizedTreeGrammar,
    Predicate,
    RHSRule,
)
from .types import Literal
from .sortedenum import sorted_product

S = TypeVar("S")  # non-terminals
T = TypeVar("T", bound=Hashable)


# Tree: TypeAlias = tuple[T, tuple["Tree[T]", ...]]
@dataclass(slots=True)
class Tree(Generic[T]):
    root: T
    children: tuple["Tree[T]", ...] = field(default=())
    hashing_function: Optional[Callable[["Tree[T]"], int]] = field(
        default=None, compare=False, hash=False, repr=False
    )
    size: int = field(init=False, compare=True, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        self.size = 1 + sum(child.size for child in self.children)
        if self.hashing_function is not None:
            self._hash = self.hashing_function(self)
        else:
            self._hash = hash((self.root, self.children))

    def __getitem__(self, i):
        match i:
            case 0:
                return self.root
            case 1:
                return self.children
            case _:
                raise IndexError()

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __lt__(self, other: "Tree[T]") -> bool:
        return self.size < other.size


def tree_size(tree: Tree[T]) -> int:
    """The number of nodes in a tree."""

    # result = 0
    # trees: deque[Tree[T]] = deque((tree,))
    # while trees:
    #     result += 1
    #     trees.extendleft(trees.pop().nodes)
    return tree.size


def takewhile_inclusive(pred: Callable[[T], bool], it: Iterable[T]) -> Iterable[T]:
    """Like takewhile, but also returns the first element not satisfying `pred`"""
    for elem in it:
        yield elem
        if not pred(elem):
            return


def really_new_terms_max_count(
    rule: RHSRule[S, T],
    nt: S,
    old_term: Tree[T],
    existing_terms: dict[S, set[Tree[T]]],
    max_count: Optional[int] = None,
    hashing_function: Optional[Callable[["Tree[T]"], int]] = None,
) -> set[Tree[T]]:
    # Genererate new terms for rule `rule` from existing terms up to `max_count`
    # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

    output_set: set[Tree[T]] = set()
    number_of_parameters = len(rule.parameters)

    # Get all possible positions of a term, that is build by nt in the output term
    all_arguments = [
        rule.binder[p.name] if isinstance(p, GVar) else p for p in rule.parameters
    ] + rule.args
    positions_of_nt = [i for i, e in enumerate(all_arguments) if e == nt]

    cached_complete_parameter_parts: Optional[list[tuple[Tree[T], ...]]] = None
    cached_complete_argument_parts: Optional[list[tuple[Tree[T], ...]]] = None

    for pos in positions_of_nt:
        pos_in_parameters = pos < number_of_parameters
        all_parameter_parts: Iterable[tuple[Tree[T], ...]] = []

        if pos_in_parameters or cached_complete_parameter_parts is None:
            for parameter_part in itertools.product(
                *(
                    (
                        (existing_terms[rule.binder[param.name]] if i != pos else [old_term])
                        if isinstance(param, GVar)
                        else [Tree(param.value, hashing_function=hashing_function)]
                    )
                    for i, param in enumerate(rule.parameters)
                )
            ):
                params_dict = {
                    param.name: term
                    for term, param in zip(parameter_part, rule.parameters)
                    if isinstance(param, GVar)
                }
                if all((predicate.eval(params_dict) for predicate in rule.predicates)):
                    all_parameter_parts.append(parameter_part)
            if not pos_in_parameters:
                cached_complete_parameter_parts = all_parameter_parts
        else:
            all_parameter_parts = cached_complete_parameter_parts

        if not pos_in_parameters or cached_complete_argument_parts is None:
            all_argument_parts = list(
                itertools.product(
                    *(
                        (existing_terms[arg] if i + number_of_parameters != pos else [old_term])
                        for i, arg in enumerate(rule.args)
                    )
                )
            )
            if pos_in_parameters:
                cached_complete_argument_parts = all_argument_parts
        else:
            all_argument_parts = cached_complete_argument_parts

        new_terms = (
            Tree(rule.terminal, param_part + arg_part, hashing_function=hashing_function)
            for param_part, arg_part in itertools.product(all_parameter_parts, all_argument_parts)
        )

        # Add new terms to the output set and check max_count
        output_set.update(new_terms)
        if max_count is not None and len(output_set) >= max_count:
            return set(itertools.islice(output_set, max_count))

    return output_set


tree_with_hash = tuple[T, tuple[int, tuple["tree_with_hash[T]", ...]]]


def enumerate_terms(
    start: S,
    grammar: ParameterizedTreeGrammar[S, T],
    max_count: Optional[int] = None,
    max_bucket_size: Optional[int] = None,
    hashing_function: Optional[Callable[[Tree[T]], int]] = None,
) -> Iterable[Tree[T]]:
    return itertools.islice(
        enumerate_terms_fast(start, grammar, max_count, max_bucket_size, hashing_function),
        max_count,
    )


@dataclass(order=True)
class PrioritizedItem(Generic[S, T]):
    # priority: int
    term: Tree[T]
    non_terminal: S = field(compare=False)


def enumerate_terms_fast(
    start: S,
    grammar: ParameterizedTreeGrammar[S, T],
    max_count: Optional[int] = None,
    max_bucket_size: Optional[int] = None,
    hashing_function: Optional[Callable[[Tree[T]], int]] = None,
) -> Iterable[Tree[T]]:
    """
    Enumerate terms as an iterator efficiently - all terms are enumerated, no guaranteed term order.
    """
    if start not in grammar.nonterminals():
        return

    queue = PriorityQueue()
    existing_terms = {n: set() for n in grammar.nonterminals()}
    inverse_grammar = {n: deque() for n in grammar.nonterminals()}
    additional_results = set()

    for n, exprs in grammar.as_tuples():
        for expr in exprs:
            non_terminals = {
                expr.binder[param.name] for param in expr.parameters if isinstance(param, GVar)
            }
            non_terminals.update(expr.args)
            for m in non_terminals:
                inverse_grammar[m].append((n, expr))
            for new_term in new_terms_max_count(
                expr, existing_terms, hashing_function=hashing_function
            ):
                queue.put(PrioritizedItem(new_term, n))
                if n == start and new_term not in additional_results:
                    if max_count is not None and len(additional_results) >= max_count:
                        return
                    yield new_term
                    additional_results.add(new_term)

    current_bucket_size = 1
    existing_terms[start].update(additional_results)

    while not queue.empty():
        items = queue
        queue = PriorityQueue()
        additional_results = set()
        while not items.empty():
            item = items.get()
            n, term = item.non_terminal, item.term
            results = existing_terms[n]
            if n != start and term in results:
                continue
            if n == start or len(results) < current_bucket_size:
                results.add(term)
                for m, expr in inverse_grammar[n]:
                    if m == start:
                        for new_term in really_new_terms_max_count(
                            expr,
                            n,
                            term,
                            existing_terms,
                            max_count,
                            hashing_function=hashing_function,
                        ):
                            # for new_term in new_terms_max_count(expr, existing_terms, max_count):
                            if (
                                new_term not in existing_terms[start]
                                and new_term not in additional_results
                            ):
                                if (
                                    max_count is not None
                                    and len(existing_terms[start]) + len(additional_results)
                                    >= max_count
                                ):
                                    return
                                yield new_term
                                # print(len(existing_terms[start]) + len(additional_results))
                                # print(new_term)
                                additional_results.add(new_term)
                                queue.put(PrioritizedItem(new_term, m))
                    else:
                        for new_term in really_new_terms_max_count(
                            expr,
                            n,
                            term,
                            existing_terms,
                            max_bucket_size,
                            hashing_function=hashing_function,
                        ):
                            # for new_term in new_terms_max_count(expr, existing_terms, max_bucket_size):
                            if len(existing_terms[m]) >= current_bucket_size:
                                queue.put(PrioritizedItem(new_term, m))
                            else:
                                items.put(PrioritizedItem(new_term, m))
            else:
                queue.put(item)
        existing_terms[start].update(additional_results)
        current_bucket_size += 1
        if max_bucket_size is not None and current_bucket_size > max_bucket_size:
            break
    return


def validate_term(rule: RHSRule[S, T], term: Tree[T]) -> bool:
    arguments = term.children
    substitution = {
        param.name: subterm
        for subterm, param in zip(arguments, rule.parameters)
        if isinstance(param, GVar)
    }
    return all(predicate.eval(substitution) for predicate in rule.predicates)


def enumerate_terms_iter(
    start: S, grammar: ParameterizedTreeGrammar[S, T], max_count: Optional[int] = None
) -> Iterable[Tree[T]]:
    """
    Enumerate terms as an iterator in an ascending way.
    """
    if start not in grammar.nonterminals():
        return

    if max_count is not None:
        max_count += 1

    old_terms: dict[S, list[Tree[T]]] = {n: [] for n in grammar.nonterminals()}
    already_checked: dict[S, set[int]] = {n: set() for n in grammar.nonterminals()}

    terms_size: int = -1

    generation = 0

    there_are_more_new_terms = True
    while there_are_more_new_terms or terms_size < sum(len(ts) for ts in old_terms.values()):
        there_are_more_new_terms = False
        terms_size = sum(len(ts) for ts in old_terms.values())
        generation = generation + 1
        for n, rhs in grammar.as_tuples():
            out_iter, avoid_iter = itertools.tee(
                merge(
                    *(
                        filter(
                            lambda new_term: hash(new_term)
                            not in already_checked[
                                n
                            ]  # Skip already generated terms for a specific symbol n
                            and validate_term(rule, new_term),  # Check the predicates
                            sorted_product(  # Build the new terms in a sorted iterator.
                                *(
                                    (
                                        old_terms[m]
                                        if not isinstance(m, Literal)
                                        else [Tree(m.value, ())]
                                    )  # Build new terms from old terms and literals
                                    for m in rule.all_args()
                                ),
                                key=tree_size,  # Sort them by size
                                combine=partial(
                                    lambda c, args: Tree(c, tuple(args)), rule.terminal
                                ),  # Construct a new term from the arguments
                            ),
                        )
                        # for c, ms in sorted(exprs, key=lambda expr: len(expr[1]))
                        for rule in rhs
                    ),
                    key=tree_size,
                ),
            )

            if n == start:
                for i in out_iter:
                    if tree_size(i) <= generation:
                        yield i
                    else:
                        there_are_more_new_terms = True
                        break

            for i in avoid_iter:
                if tree_size(i) <= generation:
                    already_checked[n].add(hash(i))
                    if max_count is None or len(old_terms[n]) <= max_count:
                        old_terms[n].append(i)
                else:
                    there_are_more_new_terms = True
                    break


def bounded_union(old_elements: set[S], new_elements: Iterable[S], max_count: int) -> set[S]:
    """Return the union of old_elements and new_elements up to max_count elements as a new set."""

    result: set[S] = old_elements.copy()
    for element in new_elements:
        if len(result) >= max_count:
            return result
        elif element not in result:
            result.add(element)
    return result


def new_terms_max_count(
    rule: RHSRule[S, T],
    existing_terms: dict[S, set[Tree[T]]],
    max_count: Optional[int] = None,
    hashing_function: Optional[Callable[["Tree[T]"], int]] = None,
) -> set[Tree[T]]:
    output_set: set[Tree[T]] = set()
    list_of_params = list(rule.binder.keys())

    for params in itertools.product(
        *(existing_terms[rule.binder[name]] for name in list_of_params)
    ):
        params_dict = {list_of_params[i]: param for i, param in enumerate(params)}
        if all((predicate.eval(params_dict) for predicate in rule.predicates)):
            for args in itertools.product(
                *(
                    (
                        existing_terms[arg]
                        if not isinstance(arg, Literal)
                        else [Tree(arg.value, (), hashing_function=hashing_function)]
                    )
                    for arg in rule.args
                )
            ):
                output_set.add(
                    Tree(
                        rule.terminal,
                        tuple(
                            itertools.chain(
                                (
                                    (
                                        Tree(parameter.value, ())
                                        if isinstance(parameter, Literal)
                                        else params_dict[parameter.name]
                                    )
                                    for parameter in rule.parameters
                                ),
                                args,
                            )
                        ),
                        hashing_function=hashing_function,
                    )
                )
                if max_count is not None and len(output_set) >= max_count:
                    return output_set
    return output_set


def new_terms(
    rhs: Iterable[RHSRule[S, T]],
    existing_terms: dict[S, set[Tree[T]]],
    max_count: Optional[int] = None,
) -> set[Tree[T]]:
    output_set: set[Tree[T]] = set()
    for rule in rhs:
        list_of_params = list(rule.binder.keys())

        for params in itertools.product(
            *(existing_terms[rule.binder[name]] for name in list_of_params)
        ):
            params_dict = {list_of_params[i]: param for i, param in enumerate(params)}
            if all((predicate.eval(params_dict) for predicate in rule.predicates)):
                for args in itertools.product(
                    *(
                        (
                            existing_terms[arg]
                            if not isinstance(arg, Literal)
                            else [Tree(arg.value, ())]
                        )
                        for arg in rule.args
                    )
                ):
                    output_set.add(
                        Tree(
                            rule.terminal,
                            tuple(
                                itertools.chain(
                                    (
                                        (
                                            Tree(parameter.value, ())
                                            if isinstance(parameter, Literal)
                                            else params_dict[parameter.name]
                                        )
                                        for parameter in rule.parameters
                                    ),
                                    args,
                                )
                            ),
                        )
                    )
    return output_set


def enumerate_terms_old(
    start: S,
    grammar: ParameterizedTreeGrammar[S, T],
    max_count: Optional[int] = 100,
) -> Iterable[Tree[T]]:
    """Given a start symbol and a tree grammar, enumerate at most max_count ground terms derivable
    from the start symbol ordered by (depth, term size).
    """

    if start not in grammar.nonterminals():
        return

    # accumulator for previously seen terms
    result: set[Tree[T]] = set()
    terms: dict[S, set[Tree[T]]] = {n: set() for n in grammar.nonterminals()}
    terms_size: int = -1
    while terms_size < sum(len(ts) for ts in terms.values()):
        terms_size = sum(len(ts) for ts in terms.values())

        if max_count is None:
            # new terms are built from previous terms according to grammar
            terms = {n: new_terms(exprs, terms) for (n, exprs) in grammar.as_tuples()}
        else:
            terms = {
                n: (
                    terms[n]
                    if len(terms[n]) >= max_count
                    else bounded_union(
                        terms[n],
                        sorted(new_terms(exprs, terms), key=tree_size),
                        max_count,
                    )
                )
                for (n, exprs) in grammar.as_tuples()
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

        new_terms: Callable[[Iterable[tuple[T, list[S]]]], set[Tree[T]]] = lambda exprs: {
            Tree(c, tuple(args))
            for (c, ms) in exprs
            for args in itertools.product(*(terms[m] for m in ms))
        }

        terms = {
            n: (
                terms[n]
                if len(terms[n]) >= max_count * (terms_size + 1)
                else grouped_bounded_union(
                    group_by_tree_size(terms[n]),
                    group_by_tree_size(new_terms(exprs)),
                    max_count,
                    term_size,
                )
            )
            for (n, exprs) in grammar.items()
        }

        for term in terms[start]:
            # yield term if not seen previously
            if tree_size(term) == term_size and term not in result:
                result.add(term)
                yield term


def interpret_term(term: Tree[T], interpretation: Optional[dict[T, Any]] = None) -> Any:
    """Recursively evaluate given term."""

    terms: deque[Tree[T]] = deque((term,))
    combinators: deque[tuple[T, int]] = deque()
    # decompose terms
    while terms:
        t = terms.pop()
        combinators.append((t.root, len(t.children)))
        terms.extend(reversed(t.children))
    results: deque[Any] = deque()

    # apply/call decomposed terms
    while combinators:
        (c, n) = combinators.pop()
        parameters_of_c: Sequence[Parameter] = []
        current_combinator: partial[Any] | T | Callable[..., Any] = (
            c if interpretation is None or c not in interpretation else interpretation[c]
        )

        if callable(current_combinator):
            try:
                parameters_of_c = list(signature(current_combinator).parameters.values())
            except ValueError:
                raise RuntimeError(
                    f"Interpretation of combinator {c} does not expose a signature. "
                    "If it's a built-in, you can simply wrap it in another function."
                )

            if n == 0 and len(parameters_of_c) == 0:
                current_combinator = current_combinator()

        arguments = deque((results.pop() for _ in range(n)))

        while arguments:
            if not callable(current_combinator):
                raise RuntimeError(
                    f"Interpretation of combinator {c} is applied to {n} argument(s), "
                    f"but can only be applied to {n - len(arguments)}"
                )

            use_partial = False

            simple_arity = len(list(filter(lambda x: x.default == _empty, parameters_of_c)))
            default_arity = len(list(filter(lambda x: x.default != _empty, parameters_of_c)))

            # if any parameter is marked as var_args, we need to use all available arguments
            pop_all = any(map(lambda x: x.kind == _ParameterKind.VAR_POSITIONAL, parameters_of_c))

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
    # d: Mapping[str, list[tuple[str, list[str]]]] = {
    #     "X": [("a", []), ("b", ["X", "Y"])],
    #     "Y": [("c", []), ("d", ["Y", "X"])],
    # }
    d: ParameterizedTreeGrammar[str, str] = ParameterizedTreeGrammar()
    d.update(
        {
            "X": deque(
                [
                    RHSRule({}, [], "a", [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], "b", [GVar("x"), GVar("y")], []),
                ]
            )
        }
    )
    d.update(
        {
            "Y": deque(
                [
                    RHSRule({}, [], "c", [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], "d", [GVar("y"), GVar("x")], []),
                ]
            )
        }
    )
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

    for i, r in enumerate(itertools.islice(enumerate_terms("X", d, max_count=100), 1000000)):
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

    # d: dict[str, list[tuple[A | B | C | D | str, list[str]]]] = {
    #     "X": [(A(), []), (B(), ["X", "Y"]), ("Z", [])],
    #     "Y": [(C(), []), (D(), ["Y", "X"])],
    # }
    d: ParameterizedTreeGrammar[str, A | B | C | D | str] = ParameterizedTreeGrammar()
    d.update(
        {
            "X": deque(
                [
                    RHSRule({}, [], A(), [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], B(), [GVar("x"), GVar("y")], []),
                ]
            )
        }
    )
    d.update(
        {
            "Y": deque(
                [
                    RHSRule({}, [], "Z", [], []),
                    RHSRule({"x": "Y", "y": "Y"}, [], D(), [GVar("y"), GVar("x")], []),
                ]
            )
        }
    )

    import timeit

    start = timeit.default_timer()

    for i, r in enumerate(
        enumerate_terms("X", d, max_count=100_000, hashing_function=lambda t: t.size)
    ):
        print(i)
        # pass
        # print(i, interpret_term(r))

    print("Time: ", timeit.default_timer() - start)


def test3() -> None:
    grammar: ParameterizedTreeGrammar[str, str] = ParameterizedTreeGrammar()
    grammar.add_rule("X", RHSRule({"y": "Y"}, [], "y", [], []))
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: True)], "y1", [], []))
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False)], "y2", [], []))
    print(grammar.show())


if __name__ == "__main__":
    test2()
