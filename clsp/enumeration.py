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
from typing import Any, Generic, Optional, TypeVar, overload
import typing
from queue import PriorityQueue
from dataclasses import dataclass, field
import random


from .grammar import (
    GVar,
    ParameterizedTreeGrammar,
    Predicate,
    RHSRule,
)

from .types import Literal

NT = TypeVar("NT")  # non-terminals
T = TypeVar("T", covariant=True, bound=Hashable)


# Tree: TypeAlias = tuple[T, tuple["Tree[T]", ...]]
@dataclass(slots=True)
class Tree(Generic[NT, T]):
    root: T
    derived_from: NT | str
    rhs_rule: RHSRule[NT, T]

    children: tuple["Tree[NT, T]", ...] = field(default=())
    variable_names: list[str] = field(default_factory=list)
    frozen: bool = field(default=False)

    size: int = field(init=False, compare=True, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)
    # logic_goals: list[Predicate] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.size = 1 + sum(child.size for child in self.children)
        self._hash = hash((self.root, self.children))
        # self.logic_goals = self.rhs_rule.predicates

    @property
    def parameters(self) -> dict[str, "Tree[NT, T]"]:
        return {name: self.children[i] for i, name in enumerate(self.variable_names)}

    @property
    def arguments(self) -> tuple["Tree[NT, T]", ...]:
        return tuple(self.children[len(self.variable_names):])

    @overload
    def __getitem__(self, i: typing.Literal[0]) -> T: ...
    @overload
    def __getitem__(self, i: typing.Literal[1]) -> tuple["Tree[NT, T]", ...]: ...

    def __getitem__(self, i: typing.Literal[0] | typing.Literal[1]) -> T | tuple["Tree[NT, T]", ...]:
        match i:
            case 0:
                return self.root
            case 1:
                return self.children
        raise IndexError()

    # def __iter__(self) -> Iterator[tuple[T, tuple["Tree[T]", ...]]]:
    #     x = iter((self.root, self.children))
    #     return x

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "Tree[NT, T]") -> bool:
        return self.size < other.size

    def __rec_to_str__(self, outermost: bool) -> str:
        str_root = [f"{str(self.root)}"]
        str_params = [
            f"{{{name}={subtree.__rec_to_str__(True)}}}"
            for name, subtree in self.parameters.items()
        ]
        str_args = [f"{subtree.__rec_to_str__(False)}" for subtree in self.arguments]

        strings = str_root + str_params + str_args
        if not outermost and len(strings) > 1:
            return f"({' '.join(strings)})"
        return " ".join(strings)

    def __str__(self) -> str:
        return self.__rec_to_str__(True)

    # subtrees() returns a list of all subtrees and their contexts.
    # The context is its path in the primary tree, the variable name of the subtree,
    # its siblings as a substitution and a list of predicates.
    # If the subtree is an argument and not a parameter, the context is empty, because there are no constraints.
    def subtrees(self, prefix: list[int]) -> typing.Generator[tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]]]:
        for i, child in list(enumerate(self.children)):
            if not child.frozen:
                if i < len(self.variable_names):
                    params: dict[str, "Tree[NT, T]"] = {name: self.parameters[name] for name, _ in self.rhs_rule.binder.items()}
                    preds: list[Predicate] = self.rhs_rule.predicates
                    yield child, prefix + [i], self.variable_names[i], params, preds
                else:
                    params: dict[str, "Tree[NT, T]"] = {}
                    preds: list[Predicate] = []
                    yield child, prefix + [i], "", params, preds
                yield from list(child.subtrees(prefix + [i]))
                #for t, p, names, param, pred in child.subtrees([i]):
                    # the path need to be updated, this way without being an argument to the recursive call
                #    yield t, [i] + p, names, param, pred



    def is_valid(self, p_subtree: tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]],
                 s_subtree: "Tree[NT, T]") -> bool:
        substitution = p_subtree[3]
        substitution.update({p_subtree[2]: s_subtree})
        return all(pred.eval(substitution) for pred in p_subtree[4])

    # TODO: is_consistent currently traverses the whole tree top down, but it should be more efficient to just traverse bottom up from the crossover point.
    # TODO: this function is buggy!!!!
    def is_consistent(self) -> bool:
        result: list[bool] = []
        for i, child in enumerate(self.children):
            if i < len(self.variable_names):
                substitution = {name: self.parameters[name] for name, _ in self.rhs_rule.binder.items()}
                result.append(all(pred.eval(substitution) for pred in self.rhs_rule.predicates))
            result.append(child.is_consistent())
        return all(result)

    # replace the subtree at the given path with the new subtree
    def replace(self, path: list[int], new_subtree: "Tree[NT, T]") -> "Tree[NT, T]":
        if not path:
            return new_subtree
        i = path.pop(0)
        if i < len(self.children):
            return Tree(self.root, self.derived_from, self.rhs_rule, tuple(
                (child.replace(path, new_subtree)) if j == i else child
                for j, child in enumerate(self.children)), self.variable_names, self.frozen)
        else:
            return self

    # crossover function
    def crossover(self, secondary_derivation_tree: "Tree[NT, T]"):  # -> "Tree[NT, T]" | None:
        # 1.
        primary_sub_trees: list[tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]]] = (
            list(self.subtrees([])))
        # 2.
        secondary_sub_trees: list["Tree[NT, T]"] = list(map(lambda x: x[0], secondary_derivation_tree.subtrees([])))
        secondary_sub_trees.insert(0, secondary_derivation_tree)
        # 3.
        while primary_sub_trees:
            # 4.
            sel_primary_subtree: tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]] = (
                random.choice(primary_sub_trees))
            primary_crossover_point: NT = sel_primary_subtree[0].derived_from
            primary_sub_trees.remove(sel_primary_subtree)
            # 5.
            temp_secondary_subtrees: list[Tree[NT, T]] = secondary_sub_trees.copy()
            temp_secondary_subtrees = list(filter(lambda x: x.derived_from == primary_crossover_point,
                                                  temp_secondary_subtrees))
            # 6.
            while temp_secondary_subtrees:
                # 7.
                sel_secondary_subtree: Tree[NT, T] = random.choice(temp_secondary_subtrees)
                temp_secondary_subtrees.remove(sel_secondary_subtree)
                if self.is_valid(sel_primary_subtree, sel_secondary_subtree):
                    offspring = self.replace(sel_primary_subtree[1].copy(), sel_secondary_subtree)
                    if (offspring.derived_from == self.derived_from) and offspring.is_consistent():
                        print(f"crossover-point: {primary_crossover_point}")
                        print(f"sel_primary_subtree path: {sel_primary_subtree[1]}")
                        print(f"sel_secondary_subtree derived from: {sel_secondary_subtree.derived_from}")
                        return offspring
        return None

    # mutating the tree by selecting a random subtree and replacing it with a new subtree inhabiting the same type.
    # Therefore, mutation needs the grammar as an extra argument to inhabit the mutations.
    # This should be more memory efficient than taking the grammar as a field in each node
    def mutate(self, grammar: ParameterizedTreeGrammar[NT, T]):  # -> "Tree[NT, T]" | None:
        # include an optional parameter maximum depth and ensure, that no tree produced by mutation exceeds this depth
        # 1.
        sub_trees: list[tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]]] = (
            list(self.subtrees([])))
        # 2.
        while sub_trees:
            # 3.
            mutated_subtree: tuple["Tree[NT, T]", list[int], str, dict[str, "Tree[NT, T]"], list[Predicate]] = (
                random.choice(sub_trees))
            mutate_point: T = mutated_subtree[1]
            sub_trees.remove(mutated_subtree)
            non_terminal: NT = mutated_subtree[0].derived_from
            # 6.
            new_sub_tree: Tree[NT, T] = random.choice(list(enumerate_terms(non_terminal, grammar, 300)))
            # 7.
            offspring = self.replace(mutate_point, new_sub_tree)
            if (offspring.derived_from == self.derived_from) and offspring.is_consistent():
                return offspring
        mutate_point = []
        non_terminal = self.derived_from
        new_sub_tree: Tree[NT, T] = random.choice(list(enumerate_terms(non_terminal, grammar, 300)))
        offspring = self.replace(mutate_point, new_sub_tree)
        if offspring.is_consistent():
            return offspring
        return None


def tree_size(tree: Tree[NT, T]) -> int:
    """The number of nodes in a tree."""

    # result = 0
    # trees: deque[Tree[T]] = deque((tree,))
    # while trees:
    #     result += 1
    #     trees.extendleft(trees.pop().nodes)
    return tree.size


# def takewhile_inclusive(pred: Callable[[T], bool], it: Iterable[T]) -> Iterable[T]:
#     """Like takewhile, but also returns the first element not satisfying `pred`"""
#     for elem in it:
#         yield elem
#         if not pred(elem):
#             return


def enumerate_term_vectors(
    non_terminals: Sequence[NT],
    existing_terms: Mapping[NT, set[Tree[NT, T]]],
    nt_term: Optional[tuple[NT, Tree[NT, T]]] = None,
) -> Iterable[tuple[Tree[NT, T], ...]]:
    """Enumerate possible term vectors for a given list of non-terminals and existing terms. Use nt_term at least once (if given)."""
    if nt_term is None:
        yield from itertools.product(*(existing_terms[n] for n in non_terminals))
    else:
        nt, term = nt_term
        for i, n in enumerate(non_terminals):
            if n == nt:
                yield from itertools.product(
                    *([term] if i == j else existing_terms[m] for j, m in enumerate(non_terminals))
                )


def generate_new_terms(
    lhs:  NT,  # the non-terminal of the rule
    rule: RHSRule[NT, T],
    existing_terms: Mapping[NT, set[Tree[NT, T]]],
    max_count: Optional[int] = None,
    nt_old_term: Optional[tuple[NT, Tree[NT, T]]] = None,
) -> set[Tree[NT, T]]:
    # Genererate new terms for rule `rule` from existing terms up to `max_count`
    # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

    output_set: set[Tree[NT, T]] = set()
    if max_count == 0:
        return output_set

    names: tuple[str, ...]
    param_nts: tuple[NT, ...]

    names, param_nts = zip(*rule.binder.items()) if len(rule.binder) > 0 else ((), ())
    literals: list[Tree[NT, T] | str] = [
        Tree(p.value, p.group, rule) if isinstance(p, Literal) else p.name for p in rule.parameters
    ]
    interleave: Callable[[Mapping[str, Tree[NT, T]]], tuple[Tree[NT, T], ...]] = lambda substitution: tuple(
        substitution[t] if isinstance(t, str) else t for t in literals
    )
    new_term: Callable[[tuple[Tree[NT, T], ...]], Tree[NT, T]] = lambda params_args: Tree(
        rule.terminal,
        lhs,
        rule,
        params_args,
        variable_names=rule.variable_names,
    )

    if nt_old_term is None:
        all_args = list(enumerate_term_vectors(rule.args, existing_terms, None))
        all_params: list[tuple[Tree[NT, T], ...]] = [
            interleave(substitution)
            for param_terms in enumerate_term_vectors(param_nts, existing_terms, None)
            for substitution in (dict(zip(names, param_terms)),)
            if all(predicate.eval(substitution) for predicate in rule.predicates)
        ]
        for params in all_params:
            for args in all_args:
                output_set.add(new_term(params + args))
                if max_count is not None and len(output_set) >= max_count:
                    return output_set
    else:
        nt, old_term = nt_old_term
        if nt in param_nts:
            cached_all_args = None
            for param_terms in enumerate_term_vectors(param_nts, existing_terms, (nt, old_term)):
                substitution = dict(zip(names, param_terms))
                if all(predicate.eval(substitution) for predicate in rule.predicates):
                    cached_all_args = (
                        list(enumerate_term_vectors(rule.args, existing_terms, None))
                        if cached_all_args is None
                        else cached_all_args
                    )
                    for args in cached_all_args:
                        output_set.add(new_term(interleave(substitution) + args))
                        if max_count is not None and len(output_set) >= max_count:
                            return output_set

        if nt in rule.args:
            cached_all_params = None
            for args in enumerate_term_vectors(rule.args, existing_terms, (nt, old_term)):
                cached_all_params = (
                    [
                        interleave(substitution)
                        for param_terms in enumerate_term_vectors(param_nts, existing_terms, None)
                        for substitution in (dict(zip(names, param_terms)),)
                        if all(predicate.eval(substitution) for predicate in rule.predicates)
                    ]
                    if cached_all_params is None
                    else cached_all_params
                )
                for params in cached_all_params:
                    output_set.add(new_term(params + args))
                    if max_count is not None and len(output_set) >= max_count:
                        return output_set

    return output_set


def enumerate_terms(
    start: NT,
    grammar: ParameterizedTreeGrammar[NT, T],
    max_count: Optional[int] = None,
    max_bucket_size: Optional[int] = None,
) -> Iterable[Tree[NT, T]]:
    return itertools.islice(
        enumerate_terms_fast(start, grammar, max_count, max_bucket_size),
        max_count,
    )


def enumerate_terms_fast(
    start: NT,
    grammar: ParameterizedTreeGrammar[NT, T],
    max_count: Optional[int] = None,
    max_bucket_size: Optional[int] = None,
) -> Iterable[Tree[NT, T]]:
    """
    Enumerate terms as an iterator efficiently - all terms are enumerated, no guaranteed term order.
    """
    if start not in grammar.nonterminals():
        return

    queues: dict[NT, PriorityQueue[Tree[NT, T]]] = {n: PriorityQueue() for n in grammar.nonterminals()}
    existing_terms: dict[NT, set[Tree[NT, T]]] = {n: set() for n in grammar.nonterminals()}
    inverse_grammar: dict[NT, deque[tuple[NT, RHSRule[NT, T]]]] = {
        n: deque() for n in grammar.nonterminals()
    }
    all_results: set[Tree[NT, T]] = set()

    for n, exprs in grammar.as_tuples():
        for expr in exprs:
            for m in expr.non_terminals():
                inverse_grammar[m].append((n, expr))
            for new_term in generate_new_terms(n, expr, existing_terms):
                queues[n].put(new_term)
                if n == start and new_term not in all_results:
                    if max_count is not None and len(all_results) >= max_count:
                        return
                    yield new_term
                    all_results.add(new_term)

    current_bucket_size = 1

    while (max_bucket_size is None or current_bucket_size <= max_bucket_size) and any(
        not queue.empty() for queue in queues.values()
    ):
        non_terminals = {n for n in grammar.nonterminals() if not queues[n].empty()}

        while non_terminals:
            n = non_terminals.pop()
            results = existing_terms[n]
            while len(results) < current_bucket_size and not queues[n].empty():
                term = queues[n].get()
                if term in results:
                    continue
                results.add(term)
                for m, expr in inverse_grammar[n]:
                    if len(existing_terms[m]) < current_bucket_size:
                        non_terminals.add(m)
                    if m == start:
                        for new_term in generate_new_terms(
                            m, expr, existing_terms, max_count, (n, term)
                        ):
                            if new_term not in all_results:
                                if max_count is not None and len(all_results) >= max_count:
                                    return
                                yield new_term
                                all_results.add(new_term)
                                queues[start].put(new_term)
                    else:
                        for new_term in generate_new_terms(
                            m, expr, existing_terms, max_bucket_size, (n, term)
                        ):
                            queues[m].put(new_term)
        current_bucket_size += 1
    return


# def validate_term(rule: RHSRule[S, T], term: Tree[T]) -> bool:
#     arguments = term.children
#     substitution = {
#         param.name: subterm
#         for subterm, param in zip(arguments, rule.parameters)
#         if isinstance(param, GVar)
#     }
#     return all(predicate.eval(substitution) for predicate in rule.predicates)


# def enumerate_terms_iter(
#     start: S, grammar: ParameterizedTreeGrammar[S, T], max_count: Optional[int] = None
# ) -> Iterable[Tree[T]]:
#     """
#     Enumerate terms as an iterator in an ascending way.
#     """
#     if start not in grammar.nonterminals():
#         return
#
#     if max_count is not None:
#         max_count += 1
#
#     old_terms: dict[S, list[Tree[T]]] = {n: [] for n in grammar.nonterminals()}
#     already_checked: dict[S, set[int]] = {n: set() for n in grammar.nonterminals()}
#
#     terms_size: int = -1
#
#     generation = 0
#
#     there_are_more_new_terms = True
#     while there_are_more_new_terms or terms_size < sum(len(ts) for ts in old_terms.values()):
#         there_are_more_new_terms = False
#         terms_size = sum(len(ts) for ts in old_terms.values())
#         generation = generation + 1
#         for n, rhs in grammar.as_tuples():
#             out_iter, avoid_iter = itertools.tee(
#                 merge(
#                     *(
#                         filter(
#                             lambda new_term: hash(new_term)
#                             not in already_checked[
#                                 n
#                             ]  # Skip already generated terms for a specific symbol n
#                             and validate_term(rule, new_term),  # Check the predicates
#                             sorted_product(  # Build the new terms in a sorted iterator.
#                                 *(
#                                     (
#                                         old_terms[m]
#                                         if not isinstance(m, Literal)
#                                         else [Tree(m.value, ())]
#                                     )  # Build new terms from old terms and literals
#                                     for m in rule.all_args()
#                                 ),
#                                 key=tree_size,  # Sort them by size
#                                 combine=partial(
#                                     lambda c, args: Tree(c, tuple(args)), rule.terminal
#                                 ),  # Construct a new term from the arguments
#                             ),
#                         )
#                         # for c, ms in sorted(exprs, key=lambda expr: len(expr[1]))
#                         for rule in rhs
#                     ),
#                     key=tree_size,
#                 ),
#             )
#
#             if n == start:
#                 for i in out_iter:
#                     if tree_size(i) <= generation:
#                         yield i
#                     else:
#                         there_are_more_new_terms = True
#                         break
#
#             for i in avoid_iter:
#                 if tree_size(i) <= generation:
#                     already_checked[n].add(hash(i))
#                     if max_count is None or len(old_terms[n]) <= max_count:
#                         old_terms[n].append(i)
#                 else:
#                     there_are_more_new_terms = True
#                     break


# def enumerate_terms_with_iter(
#     start: S,
#     grammar: ParameterizedTreeGrammar[S, T],
#     max_count: Optional[int] = None,
# ) -> Iterable[Tree[T]]:
#     return itertools.islice(enumerate_terms_iter(start, grammar, max_count), max_count)
#
#
# def bounded_union(old_elements: set[S], new_elements: Iterable[S], max_count: int) -> set[S]:
#     """Return the union of old_elements and new_elements up to max_count elements as a new set."""
#
#     result: set[S] = old_elements.copy()
#     for element in new_elements:
#         if len(result) >= max_count:
#             return result
#         elif element not in result:
#             result.add(element)
#     return result
#
#
# def new_terms_max_count(
#     rule: RHSRule[S, T], existing_terms: dict[S, set[Tree[T]]], max_count: Optional[int] = None
# ) -> set[Tree[T]]:
#     output_set: set[Tree[T]] = set()
#     list_of_params = list(rule.binder.keys())
#
#     for params in itertools.product(
#         *(existing_terms[rule.binder[name]] for name in list_of_params)
#     ):
#         params_dict = {list_of_params[i]: param for i, param in enumerate(params)}
#         if all((predicate.eval(params_dict) for predicate in rule.predicates)):
#             for args in itertools.product(
#                 *(
#                     (existing_terms[arg] if not isinstance(arg, Literal) else [(arg.value, ())])
#                     for arg in rule.args
#                 )
#             ):
#                 output_set.add(
#                     (
#                         rule.terminal,
#                         tuple(
#                             itertools.chain(
#                                 (
#                                     (
#                                         (parameter.value, ())
#                                         if isinstance(parameter, Literal)
#                                         else params_dict[parameter.name]
#                                     )
#                                     for parameter in rule.parameters
#                                 ),
#                                 args,
#                             )
#                         ),
#                     )
#                 )
#                 if max_count is not None and len(output_set) >= max_count:
#                     return output_set
#     return output_set
#
#
# def new_terms(
#     rhs: Iterable[RHSRule[S, T]],
#     existing_terms: dict[S, set[Tree[T]]],
#     max_count: Optional[int] = None,
# ) -> set[Tree[T]]:
#     output_set: set[Tree[T]] = set()
#     for rule in rhs:
#         list_of_params = list(rule.binder.keys())
#
#         for params in itertools.product(
#             *(existing_terms[rule.binder[name]] for name in list_of_params)
#         ):
#             params_dict = {list_of_params[i]: param for i, param in enumerate(params)}
#             if all((predicate.eval(params_dict) for predicate in rule.predicates)):
#                 for args in itertools.product(
#                     *(
#                         (existing_terms[arg] if not isinstance(arg, Literal) else [(arg.value, ())])
#                         for arg in rule.args
#                     )
#                 ):
#                     output_set.add(
#                         (
#                             rule.terminal,
#                             tuple(
#                                 itertools.chain(
#                                     (
#                                         (
#                                             (parameter.value, ())
#                                             if isinstance(parameter, Literal)
#                                             else params_dict[parameter.name]
#                                         )
#                                         for parameter in rule.parameters
#                                     ),
#                                     args,
#                                 )
#                             ),
#                         )
#                     )
#     return output_set
#
#
# def enumerate_terms_old(
#     start: S,
#     grammar: ParameterizedTreeGrammar[S, T],
#     max_count: Optional[int] = 100,
# ) -> Iterable[Tree[T]]:
#     """Given a start symbol and a tree grammar, enumerate at most max_count ground terms derivable
#     from the start symbol ordered by (depth, term size).
#     """
#
#     if start not in grammar.nonterminals():
#         return
#
#     # accumulator for previously seen terms
#     result: set[Tree[T]] = set()
#     terms: dict[S, set[Tree[T]]] = {n: set() for n in grammar.nonterminals()}
#     terms_size: int = -1
#     while terms_size < sum(len(ts) for ts in terms.values()):
#         terms_size = sum(len(ts) for ts in terms.values())
#
#         if max_count is None:
#             # new terms are built from previous terms according to grammar
#             terms = {n: new_terms(exprs, terms) for (n, exprs) in grammar.as_tuples()}
#         else:
#             terms = {
#                 n: (
#                     terms[n]
#                     if len(terms[n]) >= max_count
#                     else bounded_union(
#                         terms[n],
#                         sorted(new_terms(exprs, terms), key=tree_size),
#                         max_count,
#                     )
#                 )
#                 for (n, exprs) in grammar.as_tuples()
#             }
#         for term in sorted(terms[start], key=tree_size):
#             # yield term if not seen previously
#             if term not in result:
#                 result.add(term)
#                 yield term
#
#
# def group_by_tree_size(terms: Iterable[Tree[T]]) -> dict[int, set[Tree[T]]]:
#     """Groups terms by tree_size as a dictionary mapping size to sets of terms."""
#
#     result: dict[int, set[Tree[T]]] = dict()
#     for term in terms:
#         size = tree_size(term)
#         ts = result.get(size, set())
#         ts.add(term)
#         result[size] = ts
#     return result
#
#
# def grouped_bounded_union(
#     grouped_old_terms: dict[int, set[Tree[T]]],
#     grouped_new_terms: dict[int, set[Tree[T]]],
#     max_count: int,
#     term_size: int,
# ) -> set[Tree[T]]:
#     return set(
#         itertools.chain.from_iterable(
#             bounded_union(
#                 grouped_old_terms.get(i, set()),
#                 grouped_new_terms.get(i, set()),
#                 max_count,
#             )
#             for i in range(term_size + 1)
#         )
#     )
#
#
# def enumerate_terms_of_size(
#     start: S,
#     grammar: Mapping[S, Iterable[tuple[T, list[S]]]],
#     term_size: int,
#     max_count: int,
# ) -> Iterable[Tree[T]]:
#     """Given a start symbol, a tree grammar, and term size, enumerate at most max_count ground terms
#     of specified term size derivable from the start symbol."""
#
#     # accumulator for previously seen terms
#     result: set[Tree[T]] = set()
#     terms: dict[S, set[Tree[T]]] = {n: set() for n in grammar.keys()}
#     terms_size: int = -1
#     while terms_size < sum(len(ts) for ts in terms.values()):
#         terms_size = sum(len(ts) for ts in terms.values())
#
#         new_terms: Callable[[Iterable[tuple[T, list[S]]]], set[Tree[T]]] = lambda exprs: {
#             (c, tuple(args))
#             for (c, ms) in exprs
#             for args in itertools.product(*(terms[m] for m in ms))
#         }
#
#         terms = {
#             n: (
#                 terms[n]
#                 if len(terms[n]) >= max_count * (terms_size + 1)
#                 else grouped_bounded_union(
#                     group_by_tree_size(terms[n]),
#                     group_by_tree_size(new_terms(exprs)),
#                     max_count,
#                     term_size,
#                 )
#             )
#             for (n, exprs) in grammar.items()
#         }
#
#         for term in terms[start]:
#             # yield term if not seen previously
#             if tree_size(term) == term_size and term not in result:
#                 result.add(term)
#                 yield term


def interpret_term(term: Tree[NT, T], interpretation: Optional[dict[T, Any]] = None) -> Any:
    """Recursively evaluate given term."""

    terms: deque[Tree[NT, T]] = deque((term,))
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
                    RHSRule({}, [], "a", [], [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], "b", [GVar("x"), GVar("y")], ["x", "y"], []),
                ]
            )
        }
    )
    d.update(
        {
            "Y": deque(
                [
                    RHSRule({}, [], "c", [], [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], "d", [GVar("y"), GVar("x")], ["x", "y"], []),
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
                    RHSRule({}, [], A(), [], [], []),
                    RHSRule({"x": "X", "y": "Y"}, [], B(), [GVar("x"), GVar("y")], ["x", "y"], []),
                ]
            )
        }
    )
    d.update(
        {
            "Y": deque(
                [
                    RHSRule({}, [], "Z", [], [], []),
                    RHSRule({"x": "Y", "y": "Y"}, [], D(), [GVar("y"), GVar("x")], ["y", "x"], []),
                ]
            )
        }
    )

    import timeit

    start = timeit.default_timer()

    for i, r in enumerate(
        itertools.islice(
            enumerate_terms_fast("X", d, max_count=10_000),
            10,
        )
    ):
        print(r)
        # pass
        # print(i, interpret_term(r), r.size)

    print("Time: ", timeit.default_timer() - start)


def test3() -> None:
    grammar: ParameterizedTreeGrammar[str, str] = ParameterizedTreeGrammar()
    grammar.add_rule("X", RHSRule({"y": "Y"}, [], "y", [], [], []))
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: True)], "y1", [], [], []))
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False)], "y2", [], [], []))
    print(grammar.show())


if __name__ == "__main__":
    test2()
