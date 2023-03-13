from collections import deque
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Optional, TypeVar
from .subtypes import Subtypes
from .types import Type, Omega, Constructor, Product, Arrow, Intersection
from .enumeration import enumerate_terms, interpret_term
from .boolean import BooleanTerm, And, Var, Or, Not
from .bfcl import Clause, FiniteCombinatoryLogic

__all__ = [
    "Subtypes",
    "Type",
    "Omega",
    "Constructor",
    "Product",
    "Arrow",
    "Intersection",
    "enumerate_terms",
    "interpret_term",
    "BooleanTerm",
    "And",
    "Var",
    "Or",
    "Not",
    "FiniteCombinatoryLogic",
]

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")


def inhabit_and_interpret(
    repository: Mapping[C, Type[T]],
    query: list[BooleanTerm[Type[T]] | Type[T]] | BooleanTerm[Type[T]] | Type[T],
    max_count: Optional[int] = 100,
    subtypes: Optional[Subtypes[T]] = None,
) -> Iterable[Any]:
    fcl = FiniteCombinatoryLogic(
        repository, Subtypes(dict()) if subtypes is None else subtypes
    )
    if not isinstance(query, list):
        query = [query]
    grammar: dict[
        Clause[T] | BooleanTerm[Type[T]] | Type[T],
        deque[tuple[C, list[Clause[T] | BooleanTerm[Type[T]]]]],
    ] = fcl.inhabit(*query)
    for q in query:
        enumerated_terms = enumerate_terms(q, grammar, max_count)
        for term in enumerated_terms:
            yield interpret_term(term)
