# (Boolean enhanced) Combinatory Logic Synthesis in Python

This repository is a fork of [cls-python](https://github.com/cls-python/cls-python), that extends
the query language of CLS to Boolean queries.

**Note:** This does not affect the specification language for combinators in your repository.
The types of combinators are still limited to intersection types.

## Installation

Since this project does not need any additional dependencies, you can simply clone this repository to a location, where your 
code can access it. In case you *want* to install it, you can do this by either run 

    pip install . 

in the cloned repository, or

    pip install git+https://github.com/tudo-seal/bcls-python
    
without the need to clone it beforehand.

## Running the examples

Examples are stored in the `tests/` directory. Currently the following examples are provided:

- `example_1.py`, this is a simple example with three combinators, `X`, `Y` and `F`
- `example_CC.py`, this is another simple example with only one combinator `C`

If you have cloned the repository, you can run the examples from the projects root with

    python -m tests.example_1
    
and

    python -m tests.example_CC

## Usage

A *repository* is a `dict`, that maps Combinators to `Type`s.
A *combinator* can be any `Hashable` object. Special treatment occurs, if it is `Callable`.
A *query* is a `BooleanTerm` over `Types` 

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
