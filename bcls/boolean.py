"""
BooleanTerms
============

This module provides functions and objects to create and evaluate boolean terms, as well as a
function to generate a minimal DNF/CNF for a given boolean Term using the Quine Mccluskey 
algorithm.

The algorithm itself uses the following steps:
    1) Generate all variable mappings, that evaluate the term to "true" (called minterms). We
        group this list by the number of variables, that are mapped to "true".
    2) Minimize these mappings in the sense, that if two variable mappings differ in the value of
        exactly one variable, this variable is unnecessary, and both mappings can be merged into
        one mapping, that does not set this variable to any value.
        We can use the fact, that every potential term, that differs in one bit, needs to be either
        in the group before or in the group after this term, to limit the search space. In fact it
        is enough to only compare with the mappings in the group above.

        We repeat this until no mappings can be merged anymore. We can limit the mappings of each
        repetition to the mappings, that were created in the round before. The mappings, that cannot
        be further merged, and were themself not merged with another term are called prime
        implicants.
    3) Minimize the set of prime implicants, so that it covers the initial minterms (called
        essential prime implicants)
    4) Each mapping in the essential prime implicants, is transformed into a conjuction. All
        these conjuntions are fed into one disjuntion. This is a minimal dnf

For a CNF, the term is first negated, then (on the negated term) the DNF is computed. Negating
the output and using de-Morgan, we can obtain a minimal CNF from that DNF.

A mapping can be encoded as two boolean vectors (represented by two integers). One that indicates
the boolean value of each variable, and another one, that indicates which variables are actually
used. This way, we can compute most of the comparison operations by using simple bitwise operations.
Since this does not store the acutal variables, we need to ensure, that the amount and order of the
variables used anywhere in any term does not change. We calculate the list of variables used
beforehand and we call this list of variables a signature.
The most significant bit of the vectors corresponds to the first variable in the signature.

Usage
-----

    The boolean terms are generic over a type variable T. The only constraint on T is, that it is
    hashable. To satisfy the python type checker, it is enough to specify the type variable T any
    level.

    term = And[str](Or(Var("A"), And(Not(Var("D")), Var("B"))), Var("C"))
    print(term) # ((A | (~D & B)) & C)
    print(minimal_dnf(term)) # ((C & B & ~D) | (C & A))
    print(minimal_cnf(term)) # ((C) & (B | A) & (~D | A))

    Additionally, we can (optionally) omit the `Var` constructor to write terms in a shorter way

    term: BooleanTerm[str] = And(Or("A", And(Not("D"), "B")), "C")

    As a final way to declare boolean terms, we can use the &, | and ~ symbols. Since they are
    explicitly declared for boolean terms, we cannot omit the Var Constructor.

    term: BooleanTerm[str] = Var("A") | (~(Var("D") & Var("B"))) & Var("C")

Example for the encoding
------------------------

Given the signature ['A', 'B', 'C', 'D', 'E'], the mapping (13,16,5) would correspond to the
following table:

A: Unused
B: True
C: True
D: False
F: True

Example of the algorithm
------------------------

Given the Term ((A & ~B) | C) & (~C | ~A). We have the following signature: ['A', 'B', 'C'].
We first calculate all mappings and check if they evaluate to true. This can be done by counting
from 0 to 2**len(variables)-1 for the first vector. At this time no variable is ignored in the
mapping, so the second vector is always 0. The third integer is 3, since we have 3 variables.

0) All possible mappings are (grouped by number of ones in the binary representation):
[(0,0,3)],[(1,0,3),(2,0,3),(4,0,3)],[(3,0,3),(5,0,3),(6,0,3)],[(7,0,3)]

1) We now filter for the mappings, that evaluate to true, the minterms:
[(1,0,3),(4,0,3)],[(3,0,3)]

2) Now we try to find the prime impicants. We first compare (1,0,3) and (3,0,3), and we see, that
they differ in exactly the value for B, so we merge them to (1,2,3).
Note, that we could also use (3,2,3) as a merge result. Since the 2 means, that the second bit is
ignored, it follows, that (1,2,3) = (3,2,3). W.l.o.g we use the smaller one in these situations.

We now compare (4,0,3) to (3,0,3). Both mappings differ on more than one variable, so we do not
merge. We could not merge (4,0,3) with any other mapping, making this a prime implicant.

Next, We cannot compare (3,0,3) to any mapping. Since this was already part of a merge, it is no
prime implicant. This concludes the first round.

Since only one new mapping was created, it cannot be compared to any other mapping, making it
a prime implicant.

We now have the two prime implicants [(4,0,3),(1,2,3)].

3) Using a set cover algorithm we can see, that [(4,0,3), (1,2,3)] is the minimal amount of mappings
to cover all minterms.

4) Using the signature ['A', 'B', 'C'], we can calculate the DNF: (A & ~B & ~C) | (C & ~A).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, compress, groupby
from typing import Generic, Optional, TypeAlias, TypeVar

from .combinatorics import minimal_covers

T = TypeVar("T", bound=Hashable, covariant=True)

Mapping: TypeAlias = tuple[int, int]
"""Mappings are encoded as a tuple of integers. The binary representation of the first integer
corresponds to the values of the variables, the binary representation of the second integer
corresponds to the variables actually used."""


class BooleanTerm(Generic[T], ABC):
    """Parent class for all constructors of boolean terms.

    The generic paramater T is the type of the variabes. The only constraint on T is, that it is
    hashable.

    The following constructors exist:

        * And
        * Or
        * Not
        * Var

    The constructors And and Or can be of any arity, Not and Var take exactly one parameter.
    When building a formula, any subterm, that is not explicitly a BooleanTerm (or a child object),
    is interpreted as a Var, so the following calls are equivalient:

        And(Or(Var("a"), Var("b")), Not(Var("c")))

    """

    def _mask_signature(self, signature: list[T]) -> int:
        """int: Given a signature, return a bit vector indicating which variables are in the term

        Args:
            signature (list[T]): List of variable names to match against
        """
        mask = 0
        # for each variable in the signature
        for i, variable in enumerate(signature):
            # if this variable is used in this term
            if variable in self.variables:
                # set the bit at the corresponding position (from left) to 1
                mask |= 2 ** (len(signature) - 1 - i)
        return mask

    @cached_property
    @abstractmethod
    def variables(self) -> frozenset[T]:
        """frozenset[T]: Return the variables used in this term"""

    def evaluate(
        self,
        mapping: Mapping,
        signature: list[T],
        evaluate_cache: Optional[dict[tuple[BooleanTerm[T], int], bool]] = None,
    ) -> bool:
        """bool: Evaluate the term given a variable mapping

        Results of this evaluation are cached.

        Note:
            - The second field of a mapping is completely ignored in this evaluation.
              To guarantee a correct result, it should be 0.
            - This could have been an evaluate method for each child class. But to incorporate the
              caching and limit the amount of function calls, it was done at the parent class.


        Args:
            mapping (Mapping): The mapping containing the truth values for the variables
                                 in the signature
            signature (list[T]): The list of all variables in the "domain" of the mapping
        """
        # We are only interested in variables, that actually occur in the term
        interesting_variables = mapping[0] & self._mask_signature(signature)

        # Use a cached result, if such an evaluation was already queried
        if evaluate_cache is None:
            evaluate_cache = {}
        if (self, interesting_variables) in evaluate_cache:
            return evaluate_cache[(self, interesting_variables)]

        value = False
        match self:
            case And(inner):
                # an And-term evaluates to true iff all of its subterms evaluate to true
                value = all(subterm.evaluate(mapping, signature) for subterm in inner)
            case Or(inner):
                # an Or-term evaluates to true iff any of its subterms evaluate to true
                value = any(subterm.evaluate(mapping, signature) for subterm in inner)
            case Not(inner):
                # a Not-Term evaluates to true iff its subterm evaluate to false
                value = not inner.evaluate(mapping, signature)
            case Var(_):
                # interesting_variables has exactly one 1, if the only variable of this subterm is
                # true, 0 otherwise.
                value = interesting_variables != 0

        # cache the result for future lookups
        evaluate_cache[(self, interesting_variables)] = value

        return value

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    def __invert__(self) -> BooleanTerm[T]:
        return Not[T](self)

    def __or__(self, other: BooleanTerm[T]) -> BooleanTerm[T]:
        if not isinstance(other, BooleanTerm):
            raise RuntimeError(
                "Only an instance of BooleanTerm can be used in a disjuntion with a BooleanTerm"
            )

        return Or[T](self, other)

    def __and__(self, other: BooleanTerm[T]) -> BooleanTerm[T]:
        if not isinstance(other, BooleanTerm):
            raise RuntimeError(
                "Only an instance of BooleanTerm can be used in a conjuntion with a BooleanTerm"
            )

        return And[T](self, other)


class And(BooleanTerm[T]):
    """And constructor for boolean terms, takes a list of subterms"""

    __match_args__ = ("inner",)

    inner: frozenset[BooleanTerm[T]]

    def __init__(self, *inner: BooleanTerm[T] | T):
        self.inner = frozenset(
            i if isinstance(i, BooleanTerm) else Var(i) for i in inner
        )

    @cached_property
    def variables(self) -> frozenset[T]:
        return frozenset(
            chain.from_iterable((subterm.variables for subterm in self.inner))
        )

    def __str__(self) -> str:
        return f"({' & '.join(str(subterm) for subterm in self.inner)})"

    def __hash__(self) -> int:
        return hash((And, self.inner))


class Or(BooleanTerm[T]):
    """Or constructor for boolean terms, takes a list of subterms"""

    __match_args__ = ("inner",)

    inner: frozenset[BooleanTerm[T]]

    def __init__(self, *inner: BooleanTerm[T] | T):
        self.inner = frozenset(
            i if isinstance(i, BooleanTerm) else Var(i) for i in inner
        )

    @cached_property
    def variables(self) -> frozenset[T]:
        return frozenset(
            chain.from_iterable((subterm.variables for subterm in self.inner))
        )

    def __str__(self) -> str:
        return f"({' | '.join(str(subterm) for subterm in self.inner)})"

    def __hash__(self) -> int:
        return hash((Or, self.inner))


@dataclass(frozen=True)
class Var(BooleanTerm[T]):
    """Variable constructor for boolean terms. The name can be of any hashable type T"""

    name: T

    @cached_property
    def variables(self) -> frozenset[T]:
        return frozenset({self.name})

    def __str__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return hash((Var, self.name))


class Not(BooleanTerm[T]):
    """Negation constructor for boolean terms."""

    __match_args__ = ("inner",)

    inner: BooleanTerm[T]

    def __init__(self, inner: BooleanTerm[T] | T):
        if isinstance(inner, BooleanTerm):
            self.inner = inner
        else:
            self.inner = Var(inner)

    @cached_property
    def variables(self) -> frozenset[T]:
        return self.inner.variables

    def __str__(self) -> str:
        return f"~{self.inner}"

    def __hash__(self) -> int:
        return hash((Not, self.inner))


def mapping_lt(mapping: Mapping, other: Mapping) -> bool:
    """bool: Compute the inclusion operation of mappings for the set cover algorithm."""

    positives_a = mapping[0] & ~mapping[1]
    positives_b = other[0] & ~other[1]

    negatives_a = ~mapping[0] & ~mapping[1]
    negatives_b = ~other[0] & ~other[1]

    return (positives_a & positives_b == positives_a) and (
        negatives_a & negatives_b == negatives_a
    )


def to_clause(
    mapping: Mapping, signature: list[T], invert: bool = False
) -> list[BooleanTerm[T]]:
    """list[BooleanTerm[T]]: Generate a list of literals for a mapping."""

    positives = (
        (mapping[0] & ~mapping[1] & 2**x) != 0
        for x in range(len(signature) - 1, -1, -1)
    )
    negatives = (
        (~mapping[0] & ~mapping[1] & 2**x) != 0
        for x in range(len(signature) - 1, -1, -1)
    )

    if invert:
        tmp = positives
        positives = negatives
        negatives = tmp

    positive_literals = (Var(x) for x in compress(signature, positives))
    negative_literals: Iterable[BooleanTerm[T]] = (
        Not(Var(x)) for x in compress(signature, negatives)
    )

    return list(chain(positive_literals, negative_literals))


def generate_all_variable_mappings(length_of_signature: int) -> Iterable[Mapping]:
    """Iterable[Mapping]: Generate all possible variable mappings for a signature of a given length

    Since a mapping is encoded by an integer, this is done by counting to 2**length_of_signature.
    The result is sorted by the amount of 1s in the binary representation of the integer.
    """

    return (
        (x, 0)
        for x in sorted(range(2**length_of_signature), key=lambda x: x.bit_count())
    )


def get_minterms(term: BooleanTerm[T], signature: list[T]) -> Iterable[Mapping]:
    """Iterable[Mapping]: Generate all mappings for a term, that evaluate to true

    The result is sorted by the amount of variables, that are mapped to true.
    """
    all_variable_mappings = list(generate_all_variable_mappings(len(signature)))

    evaluate_cache: dict[tuple[BooleanTerm[T], int], bool] = {}

    return filter(
        lambda mapping: term.evaluate(mapping, signature, evaluate_cache),
        all_variable_mappings,
    )


def get_prime_implicants(
    minterms: list[Mapping], length_of_signature: int
) -> Iterable[Mapping]:
    """Iterable[Mapping]: Computes the prime implicants for given minterms.

    The minterms must be sorted by the number of positive mappings.
    """

    # First group the mappings by the number of positive mappings. We call the number of
    # positive mapping the class level.
    classes_of_minterms = {
        class_level: set(mappings)
        for class_level, mappings in groupby(minterms, key=lambda x: x[0].bit_count())
    }

    # We will store the mappings, that were already merged in a set to efficiently decide
    # if a mapping was already mapped
    merged: set[Mapping] = set()

    # Indicator, if the last run merged any mapping
    mergable_mappings = True

    # pylint: disable=R1702
    # If repeat until no more terms can be merged
    while mergable_mappings:
        mergable_mappings = False

        # In each run, it is enough to compare the merge results of the previous run
        new_table: dict[int, set[Mapping]] = {
            x: set() for x in classes_of_minterms.keys()
        }

        # iterate over all mappings and their respective amount of true values
        for class_level, mappings in classes_of_minterms.items():
            for mapping in mappings:
                # Indicator, if there is a mapping, that differs at exactly one position
                found_similar_mapping = False

                # These are all the mappings, that are similar to the current mapping
                # Note1: Since we are only comparing to mappings, that have one true value more,
                #        than this mapping, it is enough to generate mappings, that have a position
                #        set to 1.
                # Note2: Apparently it is faster to also generate "useless" mappings (like a mapping
                #        that has been obtained by setting a bit to 1, that was already set to 1, or
                #        setting bits to 1, that are part of the "ignored" vector), than filtering
                #        them out.
                similar_mappings = (
                    (mapping[0] | 2**position, mapping[1])
                    for position in range(length_of_signature)
                )

                # if we access a class level, that has is not set, a KeyError is raised. This
                # correspondes to the fact, that there are no mappings with this amount of 1s. In
                # that case no merging will occur, and we will continue with the next level, if it
                # exists.
                try:
                    for similar_mapping in similar_mappings:
                        # if the next level has mapping, that is similar to the current one,...
                        if similar_mapping in classes_of_minterms[class_level + 1]:
                            # ...we will generate a new mapping, that has the differing bit set in
                            # the ignore part of the mapping.
                            merged_mapping = (
                                mapping[0],
                                mapping[1] | (mapping[0] ^ similar_mapping[0]),
                            )

                            # mark the new mapping as merged
                            merged.add(similar_mapping)
                            # and add it to the table for the next run.
                            # Note: We know, that the class level of the new mapping must be the
                            #       same as the current level
                            new_table[class_level].add(merged_mapping)

                            mergable_mappings = True
                            found_similar_mapping = True
                except KeyError:
                    pass

                # If a mapping has not a similar mapping, and was never merged in a previous run,
                # it is considered a prime implicant
                if not found_similar_mapping and mapping not in merged:
                    yield mapping

        # reset for the next run
        classes_of_minterms = new_table
        merged = set()


def get_min_prime_implicants(
    term: BooleanTerm[T], signature: list[T]
) -> list[list[Mapping]]:
    minterms = list(get_minterms(term, signature))
    primes = get_prime_implicants(minterms, len(signature))
    return minimal_covers(list(primes), minterms, mapping_lt)


def minimal_dnf(term: BooleanTerm[T]) -> Or[T]:
    """BooleanTerm[T]: Compute the minimal dnf for a given boolean term

    The result is in dnf. If the result is "Or(And([]))", it corresponds to the value True, if the
    result is "Or([])" it corresponds to the value of False
    """

    signature = list(term.variables)
    minimal_primes = get_min_prime_implicants(term, signature)

    if len(minimal_primes) == 0:
        return Or()

    return Or[T](
        *(And[T](*to_clause(implicant, signature)) for implicant in minimal_primes[0])
    )


def minimal_cnf(term: BooleanTerm[T]) -> And[T]:
    """BooleanTerm[T]: Compute the minimal dnf for a given boolean term

    The result is in dnf. If the result is "Or(And([]))", it corresponds to the value True, if the
    result is "Or([])" it corresponds to the value of False
    """

    nterm: BooleanTerm[T] = Not(term)
    signature = list(nterm.variables)
    minimal_primes = get_min_prime_implicants(nterm, signature)

    if len(minimal_primes) == 0:
        return And()

    return And[T](
        *(
            Or[T](*to_clause(implicant, signature, invert=True))
            for implicant in minimal_primes[0]
        )
    )


def minimal_dnf_as_list(term: BooleanTerm[T]) -> list[list[tuple[bool, T]]]:
    """list[list[tuple[bool, T]]]: Computes the minimal dnf and returns the result stripped of all
                                   Constructors.

    Since the format of the dnf is well known, this makes it easier to iterate over the clauses
    of the dnf of a term."""

    dnf_term = minimal_dnf(term)

    if not isinstance(dnf_term, Or):
        raise RuntimeError(
            "minimal_dnf, did not return a dnf. This is most likely a bug."
        )

    output_list = []
    for clause in dnf_term.inner:
        if not isinstance(clause, And):
            raise RuntimeError(
                "minimal_dnf, did not return a dnf. This is most likely a bug."
            )

        clause_list = []
        for literal in clause.inner:
            match literal:
                case Var(name):
                    clause_list.append((True, name))
                case Not(Var(name)):
                    clause_list.append((False, name))
                case _:
                    raise RuntimeError(
                        "minimal_dnf, did not return a dnf. This is most likely a bug."
                    )
        output_list.append(clause_list)
    return output_list


def minimal_cnf_as_list(term: BooleanTerm[T]) -> list[list[tuple[bool, T]]]:
    """list[list[tuple[bool, T]]]: Computes the minimal cnf and returns the result stripped of all
                                   Constructors.

    Since the format of the cnf is well known, this makes it easier to iterate over the clauses
    of the cnf of a term."""

    cnf_term = minimal_cnf(term)

    if not isinstance(cnf_term, And):
        raise RuntimeError(
            "minimal_cnf, did not return a cnf. This is most likely a bug."
        )

    output_list = []
    for clause in cnf_term.inner:
        if not isinstance(clause, Or):
            raise RuntimeError(
                "minimal_cnf, did not return a cnf. This is most likely a bug."
            )

        clause_list = []
        for literal in clause.inner:
            match literal:
                case Var(name):
                    clause_list.append((True, name))
                case Not(Var(name)):
                    clause_list.append((False, name))
                case _:
                    raise RuntimeError(
                        "minimal_dnf, did not return a dnf. This is most likely a bug."
                    )
        output_list.append(clause_list)
    return output_list
