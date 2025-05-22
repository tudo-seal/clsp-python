"""Auxiliary functions for inspecting the specification.
   The method `inspect` analyses the given  component specifications, parameter space, and taxonomy,
   and provides info, warnings, and reports errors if any.
"""
import logging

from collections.abc import Callable, Mapping, Sequence, Hashable, Iterable
from typing import TypeVar, Optional, Any, overload
from .synthesizer import Specification, ParameterSpace, Taxonomy
from .types import LiteralParameter, TermParameter, Predicate, Abstraction, Implication, Type, Literal, Var, Constructor, Arrow, Intersection, Omega
from collections import deque
from itertools import chain

# type of components
C = TypeVar("C", bound=Hashable)

class Inspector:
    """Inspector class for analyzing component specifications, parameter space, and taxonomy."""
    _logger: logging.Logger

    def __init__(self, logger = None):
        if logger is None:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.DEBUG)
            #handler = logging.StreamHandler()
            #handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
            #self._logger.addHandler(handler)
        else:
            self._logger = logger

    @staticmethod
    def _constructors(type: Type) -> set[str]:
        """
        Get the constructors of a type.
        """
        constructors = set()
        stack: deque[Type] = deque([type])
        while stack:
            match stack.pop():
                case Intersection(l, r):
                    stack.extend((l, r))
                case Arrow(src, tgt):
                    stack.extend((src, tgt))
                case Constructor(name, arg):
                    constructors.add(name)
                    stack.append(arg)

        return constructors

    def inspect(
        self,
        componentSpecifications: Mapping[C, Specification],
        parameterSpace: ParameterSpace | None = None,
        taxonomy: Taxonomy = {},):
        """
        Inspect the component specifications, parameter space, and taxonomy.
        A `ValueError` is raised if the specifications are not well-formed, which includes:
        - a component has two parameters/arguments with the same name (shadowing)
        - a parameter name is used in the specification of a component but not abstracted via a parameter
        - a group is used in a parameter but not defined in the parameter space

        An info is logged if:
        - a name is bound to different groups in different components
        - a parameter is abstracted but not used in the specification (caveat: constraints cannot be checked)
        - a group is not used in any component
        - a concept is used only in one component
        - a concept in the taxonomy is not used in any component
        """

        if parameterSpace is None:
            parameterSpace = {}

        all_groups: set[tuple[str, str | Type]] = set()
        all_constructors: list[set[str]] = []

        for component, specification in componentSpecifications.items():
            prefix: list[LiteralParameter | TermParameter] = []
            # mapping from variable names to groups
            groups: dict[str, str | Type] = {}
            # set of parameter names occurring in bodies
            vars: set[str] = set()
            parameterizedType = specification
            # set of constructors occurring in the specification
            constructors: set[str] = set()

            while not isinstance(parameterizedType, Type):
                if isinstance(parameterizedType, Abstraction):
                    param = parameterizedType.parameter
                    if isinstance(param, LiteralParameter) or isinstance(param, TermParameter):
                        prefix.append(param)
                        if param.name in groups:
                            # check if parameter names are unique
                            raise ValueError(f"Duplicate name: {param.name}")
                        groups[param.name] = param.group
                        for n, g in all_groups:
                            if n == param.name and g != param.group:
                                self._logger.info(f"{param.name} is used both as {param.group} and {g}")
                        all_groups.add((param.name, param.group))
                    if isinstance(param, LiteralParameter):
                        if param.group not in parameterSpace:
                            # check if group is defined in the parameter space
                            raise ValueError(f"Group {param.group} is not defined in the parameter space")
                    if isinstance(param, TermParameter):
                        vars.update(param.group.free_vars)
                        constructors.update(Inspector._constructors(param.group))
                    parameterizedType = parameterizedType.body
                elif isinstance(parameterizedType, Implication):
                    parameterizedType = parameterizedType.body

            vars.update(parameterizedType.free_vars)
            constructors.update(Inspector._constructors(parameterizedType))
            all_constructors.append(constructors)
            # check if every variable in the body is abstracted
            for var in vars:
                if var not in groups:
                    raise ValueError(f"Variable {var} is not abstracted via a parameter")
                
            # check if every abstracted variable is used
            for var, group in groups.items():
                if isinstance(group, str) and var not in vars:
                    self._logger.info(f"Variable {var} is abstracted via a parameter but not used")

        all_group_names = set(g for n, g in all_groups if isinstance(g, str))

        # check if every group is used
        for group in parameterSpace.keys():
            if group not in all_group_names:
                self._logger.info(f"Group {group} is not used in any component")

        # check is some constructor is used only in one component
        for constructors in all_constructors:
            for constructor in constructors:
                if sum(1 for cs in all_constructors if constructor in cs) == 1:
                    self._logger.info(f"Concept {constructor} is used in only one component")

        # check if every concept in the taxonomy is used
        for name, subtypes in taxonomy.items():
            for c in chain([name], subtypes):
                if c not in set.union(*all_constructors):
                    self._logger.info(f"Concept {c} is not used in any component")

        # further ideas:
        # check if each parameter of a non-iterable group without candidate values is in the codomain
        return