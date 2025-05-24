# regression tests combinatorics.py

import logging
import unittest
from random import Random
from itertools import combinations
from clsp.combinatorics import maximal_elements, minimal_covers

class TestDSLUse(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
    )

    def setUp(self) -> None:
        self.rand = Random(0)

    def test_maximal_elements(self) -> None:
        """Test maximal_elements function."""
        bound = 10
        dimension = 4
        def random_element() -> tuple[int, ...]:
            return tuple(self.rand.randint(0, bound) for _ in range(dimension))

        count = 200
        elements = [random_element() for _ in range(count)]
        compare = lambda x, y: all(a <= b for a, b in zip(x, y))
        maximal = maximal_elements(elements, compare)
        self.logger.debug(f"Maximal elements: {maximal}")
        self.assertTrue(
            all(
                any(compare(x, y) for y in maximal)
                for x in elements
            ),
            "Some element is not dominated by maximal elements",
        )

        for i, x in enumerate(maximal):
            for j, y in enumerate(maximal):
                if i != j:
                    self.assertFalse(
                        compare(x, y),
                        f"Maximal elements {x} and {y} are not incomparable",
                    )

    def test_minimal_covers(self) -> None:
        """Test minimal_covers function."""
        bound = 15
        size = 4
        def random_set() -> frozenset[int]:
            return frozenset(self.rand.randint(0, bound) for _ in range(size))
        count = 20
        sets = set([random_set() for _ in range(count)])
        elements = set(range(bound))
        covers = minimal_covers(
            list(sets),
            elements,
            lambda s, e: e in s,
        )

        self.logger.debug(f"Minimal covers: {covers}")
        self.assertTrue(
            all(
                any(e in s for s in cover)
                for cover in covers
                for e in elements
            ),
            "Some element is not covered by minimal covers",
        )
        
        self.assertTrue(
            all(
                s in sets
                for cover in covers for s in cover
            ),
            "Some set in a cover is not included in sets",
        )

        for i, cover1 in enumerate(covers):
            for j, cover2 in enumerate(covers):
                if i != j:
                    self.assertFalse(
                        all(s in cover2 for s in cover1),
                        f"Minimal covers {cover1} and {cover2} are not incomparable",
                    )

        # check if all covers are found
        for i in range(len(sets)+1):
            for cover in combinations(sets, i):
                if all(
                    any(e in s for s in cover)
                    for e in elements
                ):
                    self.assertTrue(
                        any(
                            all(s in cover for s in cover2)
                            for cover2 in covers
                        ),
                        f"Cover {cover} is not found in minimal covers",
                    )
        

if __name__ == "__main__":
    unittest.main()
