# (Vanilla) Combinatory Logic Synthesis in Python

This repository is a fork of [cls-python](https://github.com/cls-python/cls-python). Advancements made to the core system from the boolean fork (see main branch) are backported here.
## Installation

Since this project does not need any additional dependencies, you can simply clone this repository to a location, where your
code can access it. In case you *want* to install it, you can do this by either run

    pip install .

in the cloned repository, or

    pip install git+https://github.com/tudo-seal/bcls-python@vanilla

without the need to clone it beforehand.

## Usage

A *repository* is a `dict`, that maps Combinators to `Type`s.
A *combinator* can be any `Hashable` object. Special treatment occurs, if it is `Callable`.
A *query* is a `Type`

Once you specified a repository `gamma` and a query `q`, you can start the inhabitation procedure by

    grammar = FiniteCombinatoryLogic(gamma, Subtypes({})).inhabit(q)

**Note:** `Subtypes` can contain a priori subtype information in the form of a `dict` from a constructor name to a `set` of constructor names.

`grammar` contains a *tree grammar* that was build by the inhabitation procedure. You can access the terms, that inhabit `q` by

    terms = enumerate_terms(q, grammar)

**Note:** Since the enumerated results are potentially infinite, `enumerated_results` returns a lazy `Generator`.

The last step is to evaluate the terms. This simply calls a term iff it is `Callable` with its assigned (and evaluated) parameters.
If it is not callable, the object in itself is returned. This is useful for constants like simple strings or numbers.

    for term in terms:
        do_sth_with(interpret_term(term))

See the examples on how to construct repositories, combinators and queries.
