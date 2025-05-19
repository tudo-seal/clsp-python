from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeVar, Generic

from .solution_space import SolutionSpace

from .subtypes import Subtypes, Taxonomy
from .types import Type, Omega, Constructor, Arrow, Intersection, Literal, Var
from .synthesizer import Synthesizer, Specification, ParameterSpace
from .dsl import DSL

__all__ = [
    "DSL",
    "Literal",
    "Var",
    "Subtypes",
    "Type",
    "Omega",
    "Constructor",
    "Arrow",
    "Intersection",
    "Synthesizer",
    "SolutionSpace",
]

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")

class CoSy(Generic[C]):
    componentSpecifications: Mapping[C, Specification]
    parameterSpace: ParameterSpace | None = None
    taxonomy: Taxonomy = {}
    _synthesizer : Synthesizer
    
    def __init__(
        self,
        componentSpecifications: Mapping[C, Specification],
        parameterSpace: ParameterSpace | None = None,
        taxonomy: Taxonomy = {},
    ) -> None:
        self.componentSpecifications = componentSpecifications
        self.parameterSpace = parameterSpace
        self.taxonomy = taxonomy
        self._synthesizer = Synthesizer(componentSpecifications, parameterSpace, taxonomy)


    def solve(self, query: Type, max_count: int = 100) -> Iterable[Any]:
        """
        Solves the given query by constructing a solution space and enumerating and interpreting the resulting trees.

        :param query: The query to solve.
        :param max_count: The maximum number of trees to enumerate.
        :return: An iterable of interpreted trees.
        """
        if not isinstance(query, Type):
            raise TypeError("Query must be of type Type")
        solutionSpace = self._synthesizer.constructSolutionSpace(query)

        trees = solutionSpace.enumerate_trees(query, max_count=max_count)
        for tree in trees:
            yield tree.interpret()
