from random import choice
from time import perf_counter

import lark

from cls_python.boolean import (And, BooleanTerm, Not, Or, Var,
                                generate_all_variable_mappings, minimal_dnf)

parser = lark.Lark(
    r"""
    ?term : negation | disjunction | conjunction | variable
    negation : "~" term
    disjunction : "(" [term ("|" term)*] ")"
    conjunction : "(" [term ("&" term)*] ")"
    variable : IDENT

    %import common.CNAME -> IDENT
    %import common.WS
    %ignore WS
    """,
    start="term",
)


class BoolTransformer(lark.Transformer):
    def negation(self, inner):
        return Not(inner[0])

    def disjunction(self, inner):
        return Or(*inner)

    def conjunction(self, inner):
        return And(*inner)

    def variable(self, name):
        return Var(name[0])


def parse(inpt):
    return BoolTransformer().transform(parser.parse(inpt))


benchmark_term1 = "((g & ((((~c & ~f & b & a) & (d & ~g) & (c & ~c & ~c)) | f) | ((b & (b & ~d & ~g)) | ~b | ~h) | c | ~g) & (b & (((a | b | e) & (~f & ~g) & (a & a & ~d) & (g | ~e | ~d)) & ((g | g) & f)))) | ((~a & (~g | f | (~b | (~f & g & a & ~d) | (h | ~c) | (g | h | ~e | ~f))) & ~g & (b | ~e)) | ((f | ((f & ~a & a) & ~c) | (h & (~f & d & b & ~b) & (d | g))) & (((a & ~g & c) | (a & a & f & ~b)) & ~b) & (((~e | ~c | g | h) | (~d | ~h | ~a) | (~a & d & h) | (~d | d)) | ((d & h) & (~h | ~h) & (f & f & ~b & h) & a))) | ((((~e & ~d & a) | ~d) & (~c | (f & ~g & e & ~f)) & ((~a & ~f & ~c & ~d) | (~b & ~f) | (~b & ~h & ~a & ~h) | (a & ~d & ~a))) & (((~e | c | ~f | a) | (~a & c & d & ~h) | a) & ((~g | d) | b | (~a & a)) & d & (g & ~g & ~b & (~g | b | ~g))) & ((~f | (~a & ~g) | (~h & ~h & b)) | ~e))) | c)"
benchmark_term2 = "(~x13 & ~x1 & ((((x12 & x1) | ((~x1 & ~x2) & (~x3 & x3 & ~x1 & x3)) | (~x5 | (x3 | ~x12)) | ((x3 & x1 & ~x4 & ~x11) & (~x9 | x7 | x6) & (~x7 | x13) & ~x8)) & ~x9 & x7 & ~x11) | ((((x13 & x1) | x8 | (x10 & ~x2 & x8)) | x11 | (~x11 & (~x9 | x10) & (x2 & ~x3))) | ~x13 | (x1 & ~x0 & ~x10) | (((~x13 | ~x5 | ~x8) & (~x12 | x6 | x13 | ~x6) & (~x7 & ~x9) & (x8 & x7 & x4)) | ((x3 & ~x10 & x12) & x9 & x12) | ~x12 | ~x10)) | (((x13 & ~x9) | (x3 & (~x10 | ~x7) & x13 & (x0 & ~x8)) | ~x7) | (((x11 | ~x7 | ~x8) & (~x12 | ~x2 | ~x4 | ~x3)) | ((~x8 | x0 | ~x7 | x1) | ~x1 | ~x12 | (x3 | ~x12)) | x4 | ((~x4 | x5 | x7 | x3) & x5)) | ((x1 & (x13 & x5) & (~x6 | x2 | x9 | x7)) | (~x9 | x3 | (x1 & x6 & x1 & ~x2) | (x4 | ~x6 | ~x8)) | (~x5 & (x13 | ~x8 | x1) & (~x1 & ~x11))) | (((~x11 & ~x1 & x13 & ~x1) & (~x12 | ~x12 | x8) & x8 & x10) | ~x0 | x9)) | (~x4 | (x2 | (~x9 | ~x3)) | x13 | (((x13 & x9 & x3) & ~x11 & (x10 | ~x8 | x1)) & x9 & (x5 & (~x3 & ~x10 & ~x10) & x5) & ~x4))))"
benchmark_term3 = "(((A & ~B) | C) & (~C | ~A))"
benchmark_term4 = "((x7 | x4) & x3 & ~x8)"
benchmark_term5 = "(((x2 & ((~x4 & ~x4) | (~x11 | x1 | x9) | x7) & ((~x3 | ~x7 | x7 | ~x12) | (~x10 | x11 | x1 | x5) | (x1 & x8 & ~x10)) & ((~x12 & ~x2 & x13) & x5)) | ~x1 | (~x8 | ((~x2 & x1 & ~x4) & (~x4 | ~x3 | ~x12) & (x10 | ~x7 | ~x4 | x12)) | ((~x5 & x3 & ~x1) & (x8 & ~x10) & (~x12 | x13) & (~x2 & ~x13))) | ((x9 & (~x4 & ~x13) & (~x3 & ~x5 & ~x3 & ~x9) & (x12 | ~x13 | x13)) | ((x10 | x5) & (~x1 & x13) & (x7 & x12 & ~x13 & x12) & ~x4) | (~x6 | (x8 & ~x12 & ~x4) | (x2 & x11 & ~x0) | (~x7 | x9)) | ~x3)))"


def wikipedia_example():
    return Or[str](
        And(Not(Var("A")), Var("B"), Not(Var("C")), Not(Var("D"))),  # 0100
        And(Var("A"), Not(Var("B")), Not(Var("C")), Not(Var("D"))),  # 1000
        And(Var("A"), Not(Var("B")), Not(Var("C")), Var("D")),  # 1001
        And(Var("A"), Var("B"), Not(Var("C")), Not(Var("D"))),  # 1100
        And(Var("A"), Not(Var("B")), Var("C"), Var("D")),  # 1011
        And(Var("A"), Var("B"), Var("C"), Not(Var("D"))),  # 1110
        And(Var("A"), Var("B"), Var("C"), Var("D")),  # 1111
    )


def random_example():
    return Or[str](
        And(Not(Var("x2")), Not(Var("x1")), Not(Var("x0"))),
        And(Not(Var("x2")), Not(Var("x1")), (Var("x0"))),
        And(Not(Var("x2")), (Var("x1")), (Var("x0"))),
        And((Var("x2")), Not(Var("x1")), Not(Var("x0"))),
        And((Var("x2")), (Var("x1")), Not(Var("x0"))),
        And((Var("x2")), (Var("x1")), (Var("x0"))),
    )


def generate_random_term(depth, max_width, variables, can_be_var=False):
    negated = choice([True, False])
    if depth == 0:
        term = Var(choice(variables))
        if negated:
            term = Not(term)
        return term

    constructor = choice([And, Or] + ([Var] if can_be_var else []))
    if constructor == Var:
        term = Var(choice(variables))
        if negated:
            term = Not(term)
        return term

    subterms = []
    num_of_subterms = 2 + choice(range(max_width - 1))

    for _ in range(num_of_subterms):
        subterms.append(generate_random_term(depth - 1, max_width, variables, True))

    term = constructor(subterms)

    return term


def check_equiv(term_1: BooleanTerm[str], term_2: BooleanTerm[str]) -> bool:
    variables = term_1.variables

    variable_list = list(variables)

    mappings = generate_all_variable_mappings(len(variable_list))

    for mapping in mappings:
        if term_1.evaluate(mapping, variable_list) != term_2.evaluate(
            mapping, variable_list
        ):
            return False

    return True


def main():
    term = parse(benchmark_term2)
    # term = generate_random_term(4, 4, [f"x{i}" for i in range(14)])
    print(term)
    timer = perf_counter()
    dnf = minimal_dnf(term)
    timer2 = perf_counter()
    print(dnf)
    print(f"Runtime: {timer2 - timer}s")
    # print(f"Testing equivilance: {check_equiv(term, dnf)}")


if __name__ == "__main__":
    main()
