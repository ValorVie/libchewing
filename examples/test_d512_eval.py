import unittest
import d512_eval
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


class PresetTests(unittest.TestCase):
    def test_balanced_stays_frozen_and_aggressive_maximizes_admissible_gain(self):
        balanced = point(.25, .25, 60, 16)
        conservative = point(.1, 1, 30, 4)  # exactly half the balanced ratio
        aggressive = point(.2, 0, 90, 30)  # exactly the admissible ratio
        invalid = point(1, 0, 120, 41)
        chosen = d512_eval.select_presets([balanced, conservative, aggressive, invalid])
        self.assertEqual(chosen, dict(conservative=conservative,
                                      balanced=balanced, aggressive=aggressive))

    def test_conservative_rejects_ratio_just_above_half_and_nonpositive_gain(self):
        balanced = point(.25, .25, 60, 16)
        good = point(.2, 1, 30, 4)
        grid = [balanced, good, point(.1, 0, 60, 9), point(0, 0, 0, 0)]
        self.assertEqual(d512_eval.select_presets(grid)['conservative'], good)

    def test_preset_ties_keep_existing_weight_then_margin_order(self):
        grid = [point(.25, .25, 60, 16), point(.2, 1, 30, 2),
                point(.1, 1.5, 30, 2), point(.1, 1, 30, 2)]
        self.assertEqual(d512_eval.select_presets(grid)['conservative'], grid[-1])

    def test_missing_balanced_or_conservative_point_fails(self):
        for grid in ([point(.1, 0, 10, 0)], [point(.25, .25, 60, 16)]):
            with self.assertRaises(ValueError):
                d512_eval.select_presets(grid)


if __name__ == '__main__':
    unittest.main()
