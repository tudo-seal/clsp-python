from cls.enumeration import enumerate_terms
from cls.grammar import GVar, ParameterizedTreeGrammar, Predicate, RHSRule
from cls.types import Literal


def main():
    grammar = ParameterizedTreeGrammar()
    grammar.add_rule(
        "X",
        RHSRule(
            {"y": "Y"},
            [],
            "x",
            [GVar("y"), GVar("y")],
        ),
    )
    grammar.add_rule(
        "Y", RHSRule({}, [Predicate(lambda _: True, "⊤")], "y1", [Literal(3, int)])
    )
    grammar.add_rule("Y", RHSRule({}, [Predicate(lambda _: False, "⊥")], "y2", []))
    print(grammar.show())

    enumeration = enumerate_terms("X", grammar)

    for t in enumeration:
        print(t)


if __name__ == "__main__":
    main()
