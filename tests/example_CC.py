from bcls import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    BooleanTerm,
    Var,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)


def test() -> None:
    a: Type[str] = Constructor("a")
    b: Type[str] = Constructor("b")
    c: Type[str] = Constructor("c")

    repository: dict[str, Type[str]] = dict(
        {
            "C": Intersection(Arrow(a, b), Intersection(a, Intersection(b, c))),
        }
    )
    environment: dict[str, set[str]] = dict()
    subtypes = Subtypes(environment)

    # target: BooleanTerm[Type] = And(b, Not(And(a, c)))

    target: BooleanTerm[Type[str]] = Var(b) & ~(Var(a) & Var(c))

    fcl = FiniteCombinatoryLogic(repository, subtypes)
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
