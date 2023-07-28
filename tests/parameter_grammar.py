from picls.enumeration import enumerate_terms
from picls.grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule
from picls.types import Literal


def main() -> None:
    grammar: ParameterizedTreeGrammar[str, str] = ParameterizedTreeGrammar()
    grammar.add_rule(
        "X",
        RHSRule({"y": "Y"}, [], "x", [GVar("y"), GVar("y")], []),
    )
    grammar.add_rule(
        "Y", RHSRule({}, [Predicate(lambda _: True, "⊤")], "y1", [Literal(3, int)], [])
    )
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False, "⊥")], "y2", [], []))
    print(grammar.show())

    enumeration = enumerate_terms("X", grammar)

    for t in enumeration:
        print(t)


if __name__ == "__main__":
    main()
