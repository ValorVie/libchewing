// SPDX-License-Identifier: LGPL-2.1-or-later
//! Behavioral ports of the Mac candidate ranking and personal preference tests.
use super::*;
use crate::syl;
use crate::{
    dictionary::StringTableBuilder,
    input::{
        keycode,
        keymap::{QWERTY_MAP, map_ascii},
    },
    lm::StaticDictBuilder,
    zhuyin::Bopomofo::*,
};

fn enabled(e: &mut Editor, value: bool) {
    e.set_editor_options(|o| o.personal_candidate_preference = value);
}
fn confirm(e: &mut Editor, unchanged: bool) {
    e.confirm_personal_candidate_commit(unchanged);
}
fn count(e: &Editor, keys: &[Syllable], text: &str) -> u64 {
    e.shared.ranking.count(
        &keys
            .iter()
            .map(|s| s.to_string().trim_end_matches('ˉ').to_owned())
            .collect::<Vec<_>>(),
        text,
    )
}

fn fixture() -> Editor {
    let mut st = StringTableBuilder::new();
    for word in [
        "試",
        "世",
        "事",
        "試試",
        "逝世",
        "世事",
        "世事事",
        "蟲",
        "重",
    ] {
        st.insert(word);
    }
    let st = st.build();
    let mut dict = StaticDictBuilder::new();
    for (n, words) in [
        (1, vec!["試", "世", "事"]),
        (2, vec!["試試", "逝世", "世事"]),
        (3, vec!["世事事"]),
    ] {
        for word in words {
            dict.insert(&vec![syl![SH, TONE4]; n], st.get_wid(word).unwrap());
        }
    }
    for word in ["蟲", "重"] {
        dict.insert(&[syl![CH, U, ENG, TONE2]], st.get_wid(word).unwrap());
    }
    let mut rare = StaticDictBuilder::new();
    rare.insert(&[syl![CH, U, ENG, TONE2]], st.get_wid("重").unwrap());
    let mut e = EditorBuilder::new()
        .string_table(st)
        .static_dict(dict.build())
        .rare_dict(rare.build())
        .build();
    e.set_editor_options(|o| {
        o.phrase_choice_rearward = true;
        o.disable_auto_learn_phrase = true;
        o.esc_clear_all_buffer = true;
    });
    enabled(&mut e, true);
    e
}
fn type_keys(e: &mut Editor, keys: &[u8]) {
    for k in keys {
        e.process_keyevent(map_ascii(&QWERTY_MAP, *k));
    }
}
fn press(e: &mut Editor, code: keycode::Keycode, sym: Keysym) {
    e.process_keyevent(KeyboardEvent::builder().code(code).ksym(sym).build());
}
fn choose(e: &mut Editor, word: &str) {
    e.start_selecting().unwrap();
    let i = e
        .all_candidates()
        .unwrap()
        .iter()
        .position(|v| v == word)
        .unwrap();
    e.select(i).unwrap();
}
fn commit_choice(e: &mut Editor, keys: &[u8], word: &str) {
    e.clear();
    type_keys(e, keys);
    choose(e, word);
    e.commit().unwrap();
    assert_eq!(e.display_commit(), word);
    confirm(e, true);
}
fn words(e: &mut Editor) -> Vec<String> {
    e.start_selecting().unwrap();
    let v = e.all_candidates().unwrap();
    e.cancel_selecting().unwrap();
    v
}

#[test]
fn personal_rare_reading_recent_choice_changes_preedit_preserves_single_window() {
    let mut e = fixture();
    type_keys(&mut e, b"tj/6");
    let original_window = words(&mut e);
    e.clear();
    commit_choice(&mut e, b"tj/6", "重");
    type_keys(&mut e, b"tj/6");
    assert_eq!(e.display(), "重");
    assert_eq!(words(&mut e), original_window);
}
#[test]
fn personal_mac_001_only_confirmed_manual_commits_learn_once() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
    e.commit().unwrap();
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
    confirm(&mut e, true);
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
}
#[test]
fn personal_mac_003_replaced_and_cancelled_choices_do_not_learn() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "逝世");
    choose(&mut e, "世事");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "逝世"), 0);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "逝世");
    e.clear();
    type_keys(&mut e, b"g4g4");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "逝世"), 0);
}
#[test]
fn personal_mac_004_delete_and_reinsert_does_not_resurrect_choice() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "試試");
    press(&mut e, keycode::KEY_BACKSPACE, SYM_BACKSPACE);
    type_keys(&mut e, b"g4");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "試試"), 0);
}
#[test]
fn personal_mac_006_changed_or_unconfirmed_output_does_not_learn() {
    for confirmed in [false, true] {
        let mut e = fixture();
        type_keys(&mut e, b"g4g4");
        choose(&mut e, "世事");
        e.commit().unwrap();
        if !confirmed {
            confirm(&mut e, false);
        } else {
            e.ack();
        }
        confirm(&mut e, true);
        assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
    }
}
#[test]
fn personal_mac_009_latest_then_counts_and_frequency_switch_independent() {
    for frequency in [false, true] {
        let mut e = fixture();
        e.set_editor_options(|o| o.sort_candidates_by_frequency = frequency);
        for (word, n) in [("試試", 10), ("逝世", 5), ("世事", 1)] {
            for _ in 0..n {
                commit_choice(&mut e, b"g4g4", word);
            }
        }
        type_keys(&mut e, b"g4g4");
        assert_eq!(words(&mut e), ["世事", "試試", "逝世"]);
        assert_eq!(
            ["試試", "逝世", "世事"].map(|w| count(&e, &[syl![SH, TONE4]; 2], w)),
            [10, 5, 1]
        );
    }
}
#[test]
fn personal_mac_010_036_switch_off_withdraws_without_committing_or_erasing() {
    let mut e = fixture();
    commit_choice(&mut e, b"tj/6", "重");
    type_keys(&mut e, b"tj/6");
    assert_eq!(e.display(), "重");
    assert_eq!(words(&mut e)[0], "蟲");
    enabled(&mut e, false);
    assert_eq!(e.display(), "蟲");
    assert_eq!(words(&mut e)[0], "蟲");
    assert_eq!(e.len(), 1);
    assert_eq!(count(&e, &[syl![CH, U, ENG, TONE2]], "重"), 1);
    enabled(&mut e, true);
    assert_eq!(e.display(), "重");
}
#[test]
fn personal_automatic_composition_and_previews_never_learn() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    let text = e.display();
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], &text), 0);
    commit_choice(&mut e, b"g4g4", "世事");
    type_keys(&mut e, b"g4g4");
    assert_eq!(e.display(), "世事");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
}
#[test]
fn personal_longer_words_first_and_single_does_not_split_original_word() {
    let mut e = fixture();
    commit_choice(&mut e, b"g4", "世");
    type_keys(&mut e, b"g4g4");
    assert_eq!(e.display(), "試試");
    commit_choice(&mut e, b"g4g4", "世事");
    type_keys(&mut e, b"g4g4");
    assert_eq!(e.display(), "世事");
    commit_choice(&mut e, b"g4g4g4", "世事事");
    type_keys(&mut e, b"g4g4g4");
    assert_eq!(e.display(), "世事事");
}
#[test]
fn personal_manual_choice_protected_from_latest_preference() {
    let mut e = fixture();
    commit_choice(&mut e, b"g4g4", "世事");
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "逝世");
    assert_eq!(e.display(), "逝世");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "逝世"), 1);
}
#[test]
fn personal_disabled_epoch_cancels_pending_choice_without_losing_manual_output() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    enabled(&mut e, false);
    enabled(&mut e, true);
    assert_eq!(e.display(), "世事");
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
}

#[test]
fn personal_mac_007_whole_overflow_learns_only_after_client_confirmation() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    e.set_editor_options(|o| o.auto_commit_threshold = 2);
    type_keys(&mut e, b"g4");
    assert_eq!(e.display_commit(), "世事");
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
    e.clear();
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
}
#[test]
fn personal_mac_007_008_split_overflow_requires_surviving_tail_and_all_receipts() {
    for cancel in [false, true] {
        let mut e = fixture();
        type_keys(&mut e, b"g4g4");
        choose(&mut e, "世事");
        // The engine currently emits whole intervals; exercise the shared receipt boundary
        // with a synthetic partial client delivery as Mac's secure buffer does.
        let prefix = e.shared.com.consume_personal(1, "世事");
        e.shared.com.remove_front(1);
        prefix.confirm(true, &e.shared.ranking);
        assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
        if cancel {
            e.clear();
        } else {
            let tail = e.shared.com.consume_personal(1, "事");
            tail.confirm(true, &e.shared.ranking);
            assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
        }
        if cancel {
            assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
        }
    }
}

#[cfg(feature = "d512")]
mod d512_personal {
    use super::*;
    use crate::d512::{
        D512Calibration, D512Error, D512Mode, D512Options, SentenceRequest, SentenceScorer,
    };
    struct Model;
    impl SentenceScorer for Model {
        fn score(
            &mut self,
            r: &SentenceRequest,
            _: std::time::Instant,
        ) -> Result<Vec<f64>, D512Error> {
            Ok(r.candidates
                .iter()
                .map(|s| if s == "逝世" { 0. } else { -20. })
                .collect())
        }
    }
    // Each mode reads its own bonus; stored history is gated by the editor switch.
    fn options(mode: D512Mode, automatic: f64, suggestion: f64) -> D512Options {
        D512Options {
            mode,
            margin: 0.5,
            personal_bonus: automatic,
            suggestion_personal_bonus: suggestion,
            ..Default::default()
        }
    }
    fn metrics(c: &D512Calibration, word: &str) -> (usize, f64) {
        (
            c.candidates
                .iter()
                .find(|v| v.text == word)
                .unwrap()
                .personal_hits,
            c.bonus,
        )
    }
    #[test]
    fn personal_d512_mac_005_bonus_and_a28_independent_modes() {
        for mode in [D512Mode::Automatic, D512Mode::Suggestion] {
            for b in [2., 8.] {
                let mut e = fixture();
                commit_choice(&mut e, b"g4g4", "世事");
                let (a, s) = if mode == D512Mode::Automatic {
                    (b, 100.)
                } else {
                    (100., b)
                };
                e.set_sentence_scorer(Some(Box::new(Model)), options(mode, a, s))
                    .unwrap();
                type_keys(&mut e, b"g4g4");
                let shown = e.display();
                let c = e.d512_calibration().unwrap();
                assert_eq!(metrics(&c, "世事"), (1, b));
                assert_eq!(
                    shown,
                    if mode == D512Mode::Automatic && b == 2. {
                        "逝世"
                    } else {
                        "世事"
                    }
                );
                assert_eq!(e.d512_report().unwrap().suggestion.is_some(), b == 2.);
                let saved = count(&e, &[syl![SH, TONE4]; 2], "世事");
                e.commit().unwrap();
                confirm(&mut e, true);
                assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), saved);
            }
        }
    }
    #[test]
    fn personal_d512_off_switch_excludes_saved_history_from_fusion() {
        let mut e = fixture();
        commit_choice(&mut e, b"g4g4", "世事");
        enabled(&mut e, false);
        e.set_sentence_scorer(
            Some(Box::new(Model)),
            options(D512Mode::Automatic, 100., 100.),
        )
        .unwrap();
        type_keys(&mut e, b"g4g4");
        assert_eq!(e.display(), "逝世");
        assert_eq!(metrics(&e.d512_calibration().unwrap(), "世事"), (0, 0.));
    }
}

#[cfg(feature = "d512")]
#[test]
fn personal_temporary_pin_has_no_rare_reading_cost() {
    let mut e = fixture();
    let original = e.shared.conv.convert_native(&{
        let mut c = crate::conversion::Composition::new();
        c.push(Symbol::from(syl![CH, U, ENG, TONE2]));
        c
    });
    let rare_cost = original
        .iter()
        .find(|o| {
            o.intervals
                .iter()
                .map(|i| i.text.as_ref())
                .collect::<String>()
                == "重"
        })
        .unwrap()
        .cost;
    let normal_cost = original
        .iter()
        .find(|o| {
            o.intervals
                .iter()
                .map(|i| i.text.as_ref())
                .collect::<String>()
                == "蟲"
        })
        .unwrap()
        .cost;
    assert_eq!(rare_cost, normal_cost);
    commit_choice(&mut e, b"tj/6", "重");
    type_keys(&mut e, b"tj/6");
    assert_eq!(e.hypotheses()[0].cost, normal_cost);
}

#[test]
fn personal_mac_039_040_deletion_keeps_manual_remainders_without_learning_them() {
    for forward in [false, true] {
        let mut e = fixture();
        commit_choice(&mut e, b"g4", "試");
        type_keys(&mut e, b"g4g4");
        choose(&mut e, "世事");
        if forward {
            press(&mut e, keycode::KEY_HOME, SYM_HOME);
            press(&mut e, keycode::KEY_DELETE, SYM_DELETE);
        } else {
            press(&mut e, keycode::KEY_BACKSPACE, SYM_BACKSPACE);
        }
        let remainder = if forward { "事" } else { "世" };
        assert_eq!(e.display(), remainder);
        e.commit().unwrap();
        confirm(&mut e, true);
        assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
        assert_eq!(count(&e, &[syl![SH, TONE4]], remainder), 0);
    }
}
#[test]
fn personal_mac_011_038_open_window_keeps_numeric_snapshot() {
    for index in 0..3 {
        let mut e = fixture();
        commit_choice(&mut e, b"g4g4", "世事");
        type_keys(&mut e, b"g4g4");
        e.start_selecting().unwrap();
        let shown = e.all_candidates().unwrap();
        e.shared.ranking.record(
            &["ㄕˋ".into(), "ㄕˋ".into()],
            "逝世",
            e.shared.ranking.next_order(),
        );
        enabled(&mut e, false);
        assert_eq!(e.all_candidates().unwrap(), shown);
        enabled(&mut e, true);
        assert_eq!(e.all_candidates().unwrap(), shown);
        e.select(index).unwrap();
        assert_eq!(e.display(), shown[index]);
    }
}
#[test]
fn personal_mac_017_019_keyboard_tab_learns_final_choice_only() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    press(&mut e, keycode::KEY_TAB, SYM_TAB);
    let first = e.display();
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], &first), 0);
    press(&mut e, keycode::KEY_TAB, SYM_TAB);
    let last = e.display();
    assert_ne!(last, first);
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], &first), 0);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], &last), 1);
}

#[test]
fn personal_user_revision_single_window_is_fixed_in_both_frequency_modes() {
    for frequency in [false, true] {
        let mut e = fixture();
        e.set_editor_options(|o| o.sort_candidates_by_frequency = frequency);
        type_keys(&mut e, b"tj/6");
        let original = words(&mut e);
        e.clear();
        commit_choice(&mut e, b"tj/6", "重");
        type_keys(&mut e, b"tj/6");
        assert_eq!(e.display(), "重");
        assert_eq!(words(&mut e), original);
    }
}

// The owned receipt can outlive the next key and the normal composition clear.
fn take_receipt(e: &mut Editor) -> Option<Box<dyn FnOnce(bool)>> {
    e.take_personal_candidate_commit()
        .map(|r| Box::new(move |unchanged| r.confirm(unchanged)) as Box<dyn FnOnce(bool)>)
}
#[test]
fn personal_mac_005_032_owned_receipts_survive_clear_and_reverse_delivery() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    e.commit().unwrap();
    let earlier = take_receipt(&mut e).unwrap();
    e.clear();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "逝世");
    e.commit().unwrap();
    let later = take_receipt(&mut e).unwrap();
    later(true);
    earlier(true);
    type_keys(&mut e, b"g4g4");
    assert_eq!(e.display(), "逝世");
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 1);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "逝世"), 1);
}
#[test]
fn personal_mac_032_owned_receipt_is_invalidated_by_switch_epoch() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    e.commit().unwrap();
    let receipt = take_receipt(&mut e).unwrap();
    enabled(&mut e, false);
    enabled(&mut e, true);
    receipt(true);
    assert_eq!(count(&e, &[syl![SH, TONE4]; 2], "世事"), 0);
}

#[cfg(feature = "d512")]
#[test]
fn personal_d512_hits_are_exact_intervals_and_single_hits_do_not_split_words() {
    let mut e = fixture();
    commit_choice(&mut e, b"g4", "世");
    commit_choice(&mut e, b"g4g4", "世事");
    struct Zero;
    impl crate::d512::SentenceScorer for Zero {
        fn score(
            &mut self,
            r: &crate::d512::SentenceRequest,
            _: std::time::Instant,
        ) -> Result<Vec<f64>, crate::d512::D512Error> {
            Ok(vec![0.; r.candidates.len()])
        }
    }
    e.set_sentence_scorer(Some(Box::new(Zero)), Default::default())
        .unwrap();
    type_keys(&mut e, b"g4g4");
    let _ = e.display();
    let c = e.d512_calibration().unwrap();
    for row in &c.candidates {
        assert_eq!(row.personal_hits, usize::from(row.text == "世事"));
        assert_eq!(row.fused, Some(row.raw + row.personal_hits as f64 * 2.));
    }
    let _ = format!("{:?}", e.take_personal_candidate_commit());
}
#[test]
fn personal_default_off_and_simple_choices_do_not_learn() {
    assert!(!EditorOptions::default().personal_candidate_preference);
    let mut e = fixture();
    enabled(&mut e, false);
    commit_choice(&mut e, b"tj/6", "重");
    type_keys(&mut e, b"tj/6");
    assert_eq!(e.display(), "蟲");
    assert_eq!(count(&e, &[syl![CH, U, ENG, TONE2]], "重"), 0);
    e.clear();
    enabled(&mut e, true);
    e.set_editor_options(|o| o.conversion_engine = ConversionEngineKind::SimpleEngine);
    type_keys(&mut e, b"tj/6");
    let _ = e.display();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![CH, U, ENG, TONE2]], "重"), 0);
}

#[cfg(feature = "d512")]
#[test]
fn personal_d512_newly_confirmed_history_invalidates_old_calibration() {
    let mut e = fixture();
    type_keys(&mut e, b"g4g4");
    choose(&mut e, "世事");
    e.commit().unwrap();
    let pending = e.take_personal_candidate_commit().unwrap();
    struct Zero;
    impl crate::d512::SentenceScorer for Zero {
        fn score(
            &mut self,
            r: &crate::d512::SentenceRequest,
            _: std::time::Instant,
        ) -> Result<Vec<f64>, crate::d512::D512Error> {
            Ok(vec![0.; r.candidates.len()])
        }
    }
    e.set_sentence_scorer(Some(Box::new(Zero)), Default::default())
        .unwrap();
    type_keys(&mut e, b"g4g4");
    let _ = e.display();
    assert!(e.d512_calibration().is_some());
    pending.confirm(true);
    assert!(e.d512_calibration().is_none());
    assert!(e.d512_report().is_none());
    assert_eq!(e.display(), "世事");
    assert!(e.d512_calibration().is_some());
}

#[cfg(feature = "d512")]
#[test]
fn personal_d512_acceptance_learns_only_changed_suggestion_words() {
    struct Prefer(String);
    impl crate::d512::SentenceScorer for Prefer {
        fn score(
            &mut self,
            r: &crate::d512::SentenceRequest,
            _: std::time::Instant,
        ) -> Result<Vec<f64>, crate::d512::D512Error> {
            Ok(r.candidates
                .iter()
                .map(|s| if s == &self.0 { 0. } else { -1000. })
                .collect())
        }
    }
    let mut e = fixture();
    type_keys(&mut e, b"tj/6g4");
    let original = e.display();
    assert!(original.ends_with("試"));
    let prefix = original.chars().next().unwrap().to_string();
    let target = format!("{prefix}世");
    e.set_sentence_scorer(
        Some(Box::new(Prefer(target.clone()))),
        crate::d512::D512Options {
            mode: crate::d512::D512Mode::Suggestion,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(e.display(), original);
    assert!(e.accept_d512_suggestion());
    assert_eq!(e.display(), target);
    e.commit().unwrap();
    confirm(&mut e, true);
    assert_eq!(count(&e, &[syl![SH, TONE4]], "世"), 1);
    assert_eq!(count(&e, &[syl![CH, U, ENG, TONE2]], &prefix), 0);
}
