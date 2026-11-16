"""Check the paired interval against Newcombe (1998), Table III."""
import unittest

from prepare_final_excel import paired_score_interval


class PairedIntervalTests(unittest.TestCase):
    def test_published_examples(self):
        examples = [
            ((20, 12, 2, 16), (.0562, .3292)),
            ((18, 12, 2, 18), (.0562, .3290)),
            ((35, 14, 0, 1), (.1461, .4175)),
        ]
        for counts, expected in examples:
            with self.subTest(counts=counts):
                interval = paired_score_interval(*counts)
                self.assertEqual(tuple(round(value, 4) for value in interval), expected)

    def test_reversing_systems_reverses_interval(self):
        lower, upper = paired_score_interval(74, 129, 117, 180)
        reverse_lower, reverse_upper = paired_score_interval(74, 117, 129, 180)
        self.assertAlmostEqual(lower, -reverse_upper)
        self.assertAlmostEqual(upper, -reverse_lower)

    def test_zero_marginals_remain_finite(self):
        lower, upper = paired_score_interval(0, 0, 0, 50)
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        self.assertGreaterEqual(lower, -1)
        self.assertLessEqual(upper, 1)


if __name__ == '__main__':
    unittest.main()
