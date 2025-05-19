# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.

from functools import partial
from inspect import Parameter, signature, _ParameterKind, _empty
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Optional, TypeVar, overload
import typing
from dataclasses import dataclass, field

T = TypeVar("T", bound=Hashable)

# Tree: TypeAlias = tuple[T, tuple["Tree[T]", ...]]
@dataclass(slots=True)
class Tree(Generic[T]):
    root: T
    children: tuple["Tree[T]", ...] = field(default=())
    # has exactly the same length as children and contains the names of the children
    child_names: tuple[str | None, ...] = field(default=())

    size: int = field(init=False, compare=True, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.size = 1 + sum(child.size for child in self.children)
        self._hash = hash((self.root, self.children))

    @property
    def parameters(self) -> dict[str, "Tree[T]"]:
        return {name: self.children[i] for i, name in enumerate(self.child_names) if name is not None}

    @property
    def arguments(self) -> tuple["Tree[T]", ...]:
        return tuple(self.children[i] for i, name in enumerate(self.child_names) if name is None)

    @overload
    def __getitem__(self, i: typing.Literal[0]) -> T: ...
    @overload
    def __getitem__(self, i: typing.Literal[1]) -> tuple["Tree[T]", ...]: ...

    def __getitem__(self, i: typing.Literal[0] | typing.Literal[1]) -> T | tuple["Tree[T]", ...]:
        match i:
            case 0:
                return self.root
            case 1:
                return self.children
        raise IndexError()

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "Tree[T]") -> bool:
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


    def interpret(self, interpretation: Optional[dict[T, Any]] = None) -> Any:
        """Recursively evaluate given term."""

        terms: deque[Tree[T]] = deque((self,))
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
