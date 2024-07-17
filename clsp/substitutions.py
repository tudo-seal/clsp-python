from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Iterable
from .types import LitParamSpec, SetTo
import copy


class SubstitutionSpace:
    def __init__(self, literals: Mapping[str, Sequence[Any]]) -> None:
        self.vars = deque()
        self.literals = literals

        self._root = {}

    def add_var(self, var: LitParamSpec) -> None:
        new_root = {}
        self.vars.append(var)

        for value in self.literals[var.group]:
            new_root = new_root | {
                value: self.filter_tree(
                    self._root, var.predicate, {var.name: value}, self.vars
                )
            }

        self._root = new_root

    def filter_tree(self, root, predicates, substitution, variables):
        return copy.deepcopy(root)
        # if len(variables) == 1:
        #     varspec = variables[0]
        #     for val in root.keys():
        #         complete_substitution = {varspec.name : val} | variables
        #
        #
        # for var, value in root.items()

    def filter_predicates(self) -> None:
        nodes = [self._root]
        while nodes:
            pass

    def _get_substitutions(
        self, root, variables: list[LitParamSpec]
    ) -> Iterable[dict[str, Any]]:
        if not variables:
            yield {}
            return
        head = variables[0]
        print(f"getting {head}")
        for val, sub in root.items():
            print(f" {val=}")
            for subst in self._get_substitutions(sub, variables[1:]):
                yield {head.name: val} | subst

    def get_substitutions(self) -> Iterable[dict[str, Any]]:
        return self._get_substitutions(self._root, list(self.vars))


def test():
    literals = {"int": [1, 2, 3, 4], "char": ["a", "b", "c"]}
    s = SubstitutionSpace(literals)

    a = LitParamSpec("a", "int", [])
    b = LitParamSpec("b", "char", [lambda vars: vars["a"] != 1])
    c = LitParamSpec("c", "int", [lambda vars: vars["a"] == vars["c"] + 11])

    s.add_var(a)
    s.add_var(c)
    s.add_var(b)

    print(list(s.get_substitutions()))
    print(s._root)
    print(s._root["a"][1])
    del s._root["a"][1][1]
    print(s._root["a"][1])
    print(s._root)


if __name__ == "__main__":
    test()
