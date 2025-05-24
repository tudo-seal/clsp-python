from collections.abc import Callable, Mapping
import timeit
from itertools import product
from random import Random
from clsp.combinatorics import maximal_elements


def main(bound = 20, dimension = 10, count = 5000, output: bool = False):
    """Benchmark maximal_elements function."""
    rand = Random(0)
    def random_element() -> tuple[int, ...]:
        return tuple(rand.randint(0, bound) for _ in range(dimension))

    elements = [random_element() for _ in range(count)]
    print(f"# elements: {len(elements)}")
    compare = lambda x, y: all(a <= b for a, b in zip(x, y))
    start = timeit.default_timer()
    maximal = maximal_elements(elements, compare)
    print(timeit.default_timer() - start)
    print(f"# maximal elements: {len(maximal)}")
    if output:
        print(f"Elements: {elements}")
        print(f"Maximal elements: {maximal}")
    return

if __name__ == "__main__":
    main()
