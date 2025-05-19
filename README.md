# Combinatory Logic Synthesizer with Predicates in Python (CLSP)

Combinatory Logic Synthesizer (CLS) is a family of synthesis frameworks based in combinatory logic and type theory.
This implementation focuses on performance and implements inhabitation in Finite Combinatory Logic with Predicates.

This repository is a fork of [cls-python](https://github.com/cls-python/cls-python).
## Installation

Since this project does not need any additional dependencies, you can simply clone this repository to a location, where your
code can access it. In case you *want* to install it, you can do this by either run

    pip install .

in the cloned repository, or

    pip install git+https://github.com/tudo-seal/clsp-python

without the need to clone it beforehand.

## Usage

To use CLS for your use-case, you need to model your problem domain in the form of:

  * *Combinators*, that can be any `Hashable` object. These are the building blocks, used to compose a solution. If they implement `Callable`, they are applied to on each other in the interpretation step, otherwise they are used as constants.
  * *Types*, that represent specifications of how combinators can be applied.
  * A *repository* of type `dict[Combinators, Types]` that maps combinators to their respective specifications.
  * A *literal environment* of `dict[str, Any]`, listing all literals (of type `Any`) that are considered in solutions for the problem, grouped by a collection identifier of type `str`
  * A *subtype taxonomy* of `dict[str,set[str]]`, representing prior knowledge of a taxonomy between type constructors.
  * A *query* of type `Type`, representing the specification, the synthesis results should conform to.

Once you specified a repository `gamma`, a literal environment `delta` and a query `q`, you can start the inhabitation procedure by

    results = FiniteCombinatoryLogic(repository=gamma, literals=delta, subtypes=Subtyping(subtypes)).inhabit(q)

`results` contains a representation of solutions that were build by the inhabitation procedure. You can access the terms, that inhabit `q` by

    terms = enumerate_trees(q, results)

**Note:** Since the enumerated results are potentially infinite, `enumerated_results` returns a lazy `Generator`.

The last step is to interpret the terms. This simply calls a term iff it is `Callable` with its assigned (and evaluated) parameters.
If it is not callable, the object in itself is returned. This is useful for constants like simple strings or numbers.

    for term in terms:
        do_sth_with(interpret_term(term))

## Types

To give combinators a specification and query for a result, we use an approach based in intersection type theory.
Intersection types can be one of the following:

  * A *function type* "A → B", signifying that a combinator transforms an A into a B. These can be chained e.g. "A → B → C", meaning a something that transforms an A and a B into a C, or even nested e.g "(A → B) → C → D", meaning something that takes something of type A → B and of type C, and constructs a D.
  * A *Constructor* "c(T)" with a name "c" and an inner type "T", e.g. "list(A)"
  * An *Intersection* "A ∩ B", signifying that something is both an A and a B.
  * ω, signifying a type, that every term inhabits (This is mostly used as an argument inside a constructor, when it is used as a nullary type constant, e.g. A(ω))
  * Any literal.
  * Combinations of the above

When constructing types inside CLS, you can use the following Notations:

  * "A → B" = `A ** B` (unfortunately the `**` operator is the only available right associative operator)
  * "c(T)" = `('c'@T)`
  * "A ∩ B" = `A & B`
  * ω = `Omega()`
  * A literal l = `Literal(l)`

**Example:** `('list'@('wheel'@Omega())) ** ('Motor'@(('Electric'@Omega) & ('Combustion'@Omega()))) ** ('Car'@Omega())` is a valid type

A query can only be specified by an intersection type, but a combinator can be specified by a *parameterized type*.

**Parameterized types** add quantifiers and predicates in front of an intersection type and allow for literal variables inside the intersection type.
Usage is best explained in an example:

    DSL().Use('x', 'int').Use('y', 'int').SuchThat(lambda x, y: x > y).In(LVar('x') ** LVar('y'))

This corresponds to each type in the form x → y where x and y are numbers and x > y.

Further examples can be found in the tests folder.
