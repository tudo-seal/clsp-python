from cls_python.boolean import BooleanTerm, Var
from cls_python.enumeration_new import enumerate_terms
from cls_python.pfcl import FiniteCombinatoryLogic
from cls_python.subtypes import Subtypes
from cls_python.types import Arrow, Constructor, Intersection, Type


def test() -> None:
    a: Type = Constructor("a")
    b: Type = Constructor("b")
    c: Type = Constructor("c")

    repository: dict[object, Type] = dict[object, Type](
        {
            "C": Intersection(Arrow(a, b), Intersection(a, Intersection(b, c))),
        }
    )
    environment: dict[object, set] = dict[object, set]()
    subtypes: Subtypes = Subtypes(environment)

    # target: BooleanTerm[Type] = And(b, Not(And(a, c)))

    target: BooleanTerm[Type] = Var(b) & ~(Var(a) & Var(c))

    fcl: FiniteCombinatoryLogic = FiniteCombinatoryLogic(repository, subtypes)
    result = fcl.inhabit(target)

    enumerated_result = enumerate_terms(target, result)

    for real_result in enumerated_result:
        print(real_result)

    # if result.check_empty(target_type):
    #     print("No inhabitants")
    # else:
    #     for tree in result[target_type][0:10]:
    #         print(tree)
    #         print("")


if __name__ == "__main__":
    test()
