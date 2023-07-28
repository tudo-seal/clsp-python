from tests.example_pi_counting import counting
from tests.example_pi_labyrinth import labyrinth
from tests.example_pi_labyrinth_set_to import labyrinth_set_to


# def termparams():
#     global count_i
#     count_i = 0
#
#     def print_then_true(vars):
#         global count_i
#         count_i = count_i + 1
#         print(
#             f"({count_i}) I can decide on the following variables: {list(vars.keys())}"
#         )
#         return True
#
#     repo = {
#         "Y": Param(
#             "a",
#             int,
#             lambda _: True,
#             Param(
#                 "x",
#                 Constructor("b"),
#                 print_then_true,
#                 Param("z", int, lambda _: True, Constructor("a")),
#             ),
#         ),
#         "X": Constructor("b"),
#     }
#
#     literals = {int: [0, 1, 2]}
#
#     fcl: FiniteCombinatoryLogic[str, str] = FiniteCombinatoryLogic(
#         repo, literals=literals
#     )
#     grammar = fcl.inhabit(Constructor("a"))
#     for i, term in enumerate(enumerate_terms(Constructor("a"), grammar)):
#         print(f"{i}, {term}")
#
#
# def set_to_test():
#     X = lambda a, b: f"X {a} {b}"
#     repo = {
#         X: Param(
#             "a",
#             int,
#             lambda _: True,
#             Param("b", int, SetTo(lambda vars: vars["a"].value + 1), Constructor("x")),
#         )
#     }
#
#     literals = {int: [0, 1, 2]}
#
#     fcl: FiniteCombinatoryLogic[str, Any] = FiniteCombinatoryLogic(
#         repo, literals=literals
#     )
#
#     grammar = fcl.inhabit(Constructor("x"))
#
#     for term in enumerate_terms(Constructor("x"), grammar):
#         print(interpret_term(term))
#

if __name__ == "__main__":
    print("Counting Example:")
    counting()
    print("Labyrinth Example (SetTo):")
    labyrinth_set_to()
    print("Labyrinth Example:")
    labyrinth()
