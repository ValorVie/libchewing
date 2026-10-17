import unittest
from d512_eval import select_point


def point(w, tau, helpful, harmful):
    return dict(w=w, tau=tau, delta_pp=helpful-harmful,
                helpful_overrides=helpful, harmful_overrides=harmful)


class SelectionTests(unittest.TestCase):
    def test_rejects_best_delta_if_harm_exceeds_one_third(self):
        grid = [point(2, 0, 100, 34), point(1, .5, 90, 30)]
        self.assertEqual(select_point(grid), grid[1])

    def test_ties_prefer_smallest_weight_then_margin(self):
        grid = [point(1, 0, 90, 30), point(.5, 1, 90, 30), point(.5, .5, 90, 30)]
        self.assertEqual(select_point(grid), grid[2])

    def test_no_admissible_point_is_an_error(self):
        with self.assertRaises(ValueError):
            select_point([point(1, 0, 3, 2)])


if __name__ == '__main__':
    unittest.main()
