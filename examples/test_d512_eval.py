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


def suggestion_point(tau, total, correct, w=.25):
    # Native-wrong/suggestion-wrong changes still count in the denominator.
    return dict(w=w, tau=tau, suggestions=total, correct_suggestions=correct,
                helpful_overrides=correct, harmful_overrides=0,
                neutral_overrides=total-correct)


class SuggestionPresetTests(unittest.TestCase):
    def test_precision_counts_all_suggestions_and_uses_three_targets(self):
        aggressive = suggestion_point(.1, 100, 80)
        balanced = suggestion_point(.2, 50, 45)
        conservative = suggestion_point(.3, 20, 19)
        self.assertEqual(d512_eval.select_suggestion_presets(
            [aggressive, balanced, conservative]), dict(conservative=conservative,
                balanced=balanced, aggressive=aggressive))

    def test_maximizes_correct_count_then_prefers_smaller_margin_not_precision(self):
        preferred = suggestion_point(.1, 100, 95)
        grid = [suggestion_point(.3, 94, 94), suggestion_point(.2, 99, 95), preferred]
        selected = d512_eval.select_suggestion_presets(grid)
        self.assertTrue(all(p == preferred for p in selected.values()))

    def test_weight_is_fixed_and_zero_suggestions_do_not_qualify(self):
        valid = suggestion_point(.2, 20, 19)
        selected = d512_eval.select_suggestion_presets(
            [suggestion_point(0, 100, 100, w=.5), suggestion_point(.1, 0, 0), valid])
        self.assertTrue(all(p == valid for p in selected.values()))

    def test_precision_boundary_is_inclusive_without_rounding(self):
        valid = suggestion_point(2, 20, 19)
        selected = d512_eval.select_suggestion_presets(
            [suggestion_point(1, 10000, 9499), valid])
        self.assertEqual(selected['conservative'], valid)

    def test_missing_tier_stops_without_relaxing_precision(self):
        for grid in ([suggestion_point(1, 10, 9)], [suggestion_point(1, 0, 0)]):
            with self.assertRaisesRegex(ValueError, 'conservative'):
                d512_eval.select_suggestion_presets(grid)


if __name__ == '__main__':
    unittest.main()
