from clsp.dsl import DSL
from clsp.enumeration import Tree
from clsp.fcl import Contains

from clsp.types import Constructor, Literal, LVar

from typing import Any, Iterable


class DAGRepository:
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

    def __init__(self, dimension_upper_bound: int, dimension_lower_bound: int = 0):
        self.dimension_upper_bound: int = dimension_upper_bound
        self.dimension_lower_bound: int = dimension_lower_bound
        self.dimension: Iterable[int] = range(self.dimension_lower_bound, self.dimension_upper_bound + 1)

    def recognize_beside_associativity(self, term: Tree[Any, str]) -> bool:
        """
        Recognizes the left-hand side of the beside associativity rewriting rule.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # beside(beside(x,y),z)
        children: list[Tree[Any, str]] = list(term.children)
        non_literal_children: list[Tree[Any, str]] = [c for c in children if not c.is_literal]
        if not term.root == "beside":
            return all([self.recognize_beside_associativity(t) for t in non_literal_children])
        assert (
                len(non_literal_children) == 2
        )
        left_term = non_literal_children[0]
        if left_term.root == "beside":
            return True
        return all([self.recognize_beside_associativity(t) for t in non_literal_children])

    def recognize_before_associativity(self, term: Tree[Any, str]) -> bool:
        """
        Recognizes the left-hand side of the before associativity rewriting rule.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # before(before(x,y),z)
        children: list[Tree[Any, str]] = list(term.children)
        non_literal_children: list[Tree[Any, str]] = [c for c in children if not c.is_literal]
        if not term.root == "before":
            return all([self.recognize_before_associativity(t) for t in non_literal_children])
        assert (
                len(non_literal_children) == 2
        )
        left_term = non_literal_children[0]
        if left_term.root == "before":
            return True
        return all([self.recognize_before_associativity(t) for t in non_literal_children])

    def recognize_abiding(self, term: Tree[Any, str]) -> bool:
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

    def recognize_left_hand_sides(self, term: Tree[Any, str]) -> bool:
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
    def base_delta(self) -> dict[str, list[Any]]:
        return {"nat": self.Nat(),
                "bool": [True, False]}

    def delta(self) -> dict[str, Any]:
        return self.base_delta() | {
            "dimension": self.dimension
        }

    def gamma(self):
        return {
            "edge": DSL()
            .In(Constructor("graph",
                            Constructor("input", Literal(1, "dimension")) &
                            Constructor("output", Literal(1, "dimension")) &
                            Constructor("size", Literal(1, "nat")))),  # Constructor("size", Literal(0, "nat")))),
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
            .As(lambda s3: {x for x in range(0, s3 + 1)}, multi_value=True)
            .Use("s2", "nat")
            .As(lambda s3, s1: s3 - s1)
            .Use("x", Constructor("graph",
                                  Constructor("input", LVar("m")) &
                                  Constructor("output", LVar("n")) &
                                  Constructor("size", LVar("s1"))))
            .Use("y", Constructor("graph",
                                  Constructor("input", LVar("p")) &
                                  Constructor("output", LVar("q")) &
                                  Constructor("size", LVar("s2"))))
            .In(Constructor("graph",
                            Constructor("input", LVar("i")) &
                            Constructor("output", LVar("o")) &
                            Constructor("size", LVar("s3")))),
            "before": DSL()
            .Use("m", "dimension")
            .Use("n", "dimension")
            .Use("p", "dimension")
            .Use("s3", "nat")
            .Use("s1", "nat")
            .As(lambda s3: {x for x in range(0, s3 + 1)}, multi_value=True)
            .Use("s2", "nat")
            .As(lambda s3, s1: s3 - s1)
            .Use("x", Constructor("graph",
                                  Constructor("input", LVar("m")) &
                                  Constructor("output", LVar("n")) &
                                  Constructor("size", LVar("s1"))))
            .Use("y", Constructor("graph",
                                  Constructor("input", LVar("n")) &
                                  Constructor("output", LVar("p")) &
                                  Constructor("size", LVar("s2"))))
            .In(Constructor("graph",
                            Constructor("input", LVar("m")) &
                            Constructor("output", LVar("p")) &
                            Constructor("size", LVar("s3")))),
            "swap": DSL()
            .Use("io", "dimension")
            .Use("m", "dimension")
            .With(lambda io, m: 0 < m < io)  # swapping zero connections is neutral
            .Use("n", "dimension")
            .As(lambda io, m: io - m)
            .In(Constructor("graph",
                            Constructor("input", LVar("io")) &
                            Constructor("output", LVar("io")) &
                            Constructor("size", Literal(1, "nat")))),  # Constructor("size", Literal(0, "nat")))),
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
