// SPDX-License-Identifier: LGPL-2.1-or-later
//! Optional D512 sentence reranking; tensor protocol follows the MIT model card.
use crate::{
    conversion::{Composition, Outcome, Selection},
    dictionary::StringTable,
};
use std::{
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(feature = "d512-ort")]
mod onnx;
#[cfg(feature = "d512-ort")]
pub use onnx::OrtSentenceScorer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D512Mode {
    Off,
    Automatic,
    Suggestion,
}

#[derive(Debug, Clone, Copy)]
pub struct D512Options {
    pub mode: D512Mode,
    pub weight: f64,
    pub margin: f64,
    pub timeout: Duration,
}
impl Default for D512Options {
    fn default() -> Self {
        Self {
            mode: D512Mode::Automatic,
            weight: 0.25,
            margin: 0.5,
            timeout: Duration::from_millis(100),
        }
    }
}

pub struct SentenceRequest {
    pub context: String,
    pub reading: Vec<String>,
    pub candidates: Vec<String>,
}
impl fmt::Debug for SentenceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SentenceRequest")
            .field("syllables", &self.reading.len())
            .field("candidates", &self.candidates.len())
            .finish_non_exhaustive()
    }
}

/// Return one finite natural-log score per candidate, in the given order.
/// Implementations must bound transport/inference by `deadline`, return an
/// error without retries, and must not log or persist request text.
pub trait SentenceScorer: Send {
    fn score(
        &mut self,
        request: &SentenceRequest,
        deadline: Instant,
    ) -> Result<Vec<f64>, D512Error>;
}

#[derive(Clone)]
pub struct D512Suggestion {
    pub text: String,
    pub outcome_index: usize,
    pub fused_gap: f64,
}
impl fmt::Debug for D512Suggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512Suggestion")
            .field("outcome_index", &self.outcome_index)
            .field("fused_gap", &self.fused_gap)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, Clone)]
pub struct D512Report {
    pub selected_index: usize,
    pub suggestion: Option<D512Suggestion>,
    pub fallback: Option<D512Error>,
    pub candidate_indices: Vec<usize>,
    pub model_scores: Vec<f64>,
    pub elapsed: Duration,
}

/// Candidate scores in first-occurrence native text order.
#[derive(Clone)]
pub struct D512CalibrationCandidate {
    pub text: String,
    pub raw: f64,
    pub model_logp: Option<f64>,
    pub fused: Option<f64>,
}
impl fmt::Debug for D512CalibrationCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512CalibrationCandidate")
            .field("raw", &self.raw)
            .field("model_logp", &self.model_logp)
            .field("fused", &self.fused)
            .finish_non_exhaustive()
    }
}

/// Latest actual scoring attempt; contains sensitive text, never printed by Debug.
#[derive(Clone)]
pub struct D512Calibration {
    pub reading: Vec<String>,
    pub candidates: Vec<D512CalibrationCandidate>,
    pub default_index: usize,
    pub winner_index: usize,
    pub applied: bool,
    pub suggested: bool,
    pub weight: f64,
    pub margin: f64,
    pub mode: D512Mode,
}
impl fmt::Debug for D512Calibration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512Calibration")
            .field("syllables", &self.reading.len())
            .field("candidates", &self.candidates.len())
            .field("default_index", &self.default_index)
            .field("winner_index", &self.winner_index)
            .field("applied", &self.applied)
            .field("suggested", &self.suggested)
            .field("weight", &self.weight)
            .field("margin", &self.margin)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

// Input identity of the last completed scoring attempt; never printed by Debug.
struct ReportSnapshot {
    composition: Composition,
    committed_prefix: u64,
}
impl ReportSnapshot {
    fn matches(&self, current: &Composition, committed_prefix: u64) -> bool {
        self.committed_prefix == committed_prefix
            && self.composition.symbols() == current.symbols()
            && self.composition.gaps() == current.gaps()
            && self
                .composition
                .selections()
                .iter()
                .map(|s| (s.start, s.end, s.wid))
                .eq(current.selections().iter().map(|s| (s.start, s.end, s.wid)))
    }
}

/// Shared by an editor and its conversion engine, including after engine switches.
#[derive(Default)]
pub struct D512State {
    scorer: Option<Box<dyn SentenceScorer + Send>>,
    options: D512Options,
    context: String,
    last: Option<D512Report>,
    calibration: Option<D512Calibration>,
    report_snapshot: Option<ReportSnapshot>,
    editor_committed_prefix: u64,
    suggested_path: Option<Vec<Selection>>,
}
impl fmt::Debug for D512State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512State")
            .field("configured", &self.scorer.is_some())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl D512State {
    pub(crate) fn configure(
        &mut self,
        scorer: Option<Box<dyn SentenceScorer + Send>>,
        options: D512Options,
    ) -> Result<(), D512Error> {
        self.scorer = None;
        self.invalidate();
        if !options.weight.is_finite()
            || !options.margin.is_finite()
            || options.weight < 0.
            || options.margin < 0.
            || options.timeout.is_zero()
            || options.timeout > Duration::from_secs(5)
        {
            return Err(D512Error::InvalidConfig);
        }
        self.options = options;
        self.scorer = scorer;
        Ok(())
    }
    pub(crate) fn set_context(&mut self, context: &str) {
        self.context = context
            .chars()
            .rev()
            .take(64)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        self.invalidate();
    }
    pub(crate) fn report(&self) -> Option<D512Report> {
        self.last.clone()
    }
    pub(crate) fn set_editor_prefix(&mut self, committed_prefix: u64) {
        self.editor_committed_prefix = committed_prefix;
    }
    pub(crate) fn report_for(
        &self,
        current: &Composition,
        committed_prefix: u64,
    ) -> Option<D512Report> {
        if !self.snapshot_matches(current, committed_prefix) {
            return None;
        }
        self.report()
    }
    pub(crate) fn calibration_for(
        &self,
        current: &Composition,
        committed_prefix: u64,
    ) -> Option<D512Calibration> {
        if !self.snapshot_matches(current, committed_prefix) {
            return None;
        }
        self.calibration.clone()
    }
    fn snapshot_matches(&self, current: &Composition, committed_prefix: u64) -> bool {
        self.report_snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.matches(current, committed_prefix))
    }
    pub(crate) fn suggestion_for(
        &self,
        current: &Composition,
        committed_prefix: u64,
    ) -> Option<(D512Suggestion, Vec<Selection>)> {
        if self.options.mode != D512Mode::Suggestion {
            return None;
        }
        let suggestion = self.report_for(current, committed_prefix)?.suggestion?;
        Some((suggestion, self.suggested_path.clone()?))
    }
    pub(crate) fn invalidate(&mut self) {
        self.last = None;
        self.calibration = None;
        self.report_snapshot = None;
        self.suggested_path = None;
    }
    pub(crate) fn rerank(
        &mut self,
        com: &Composition,
        strings: &StringTable,
        mut outcomes: Vec<Outcome>,
    ) -> Vec<Outcome> {
        self.invalidate();
        if self.options.mode == D512Mode::Off || self.scorer.is_none() {
            return outcomes;
        }
        let start = Instant::now();
        let deadline = start + self.options.timeout;
        let mut report = D512Report {
            selected_index: 0,
            suggestion: None,
            fallback: None,
            candidate_indices: vec![],
            model_scores: vec![],
            elapsed: Duration::ZERO,
        };
        let attempt = (|| -> Result<(), D512Error> {
            if com.is_empty() || com.len() > 32 || outcomes.is_empty() {
                return Err(D512Error::OutOfLimits);
            }
            let reading: Vec<String> = com
                .symbols()
                .iter()
                .map(|s| {
                    s.to_syllable()
                        .map(|s| s.to_string().trim_end_matches('ˉ').to_owned())
                        .ok_or(D512Error::BadReading)
                })
                .collect::<Result<_, _>>()?;
            let mut candidates = vec![];
            let mut engine = vec![];
            for (i, out) in outcomes.iter().enumerate() {
                let text: String = out.intervals.iter().map(|it| it.text.as_ref()).collect();
                if text.chars().count() != reading.len() {
                    return Err(D512Error::OutOfLimits);
                }
                for sel in com.selections() {
                    let fixed = strings.get_text(sel.wid).ok_or(D512Error::ProtectedRange)?;
                    let actual: String = text
                        .chars()
                        .skip(sel.start)
                        .take(sel.end.saturating_sub(sel.start))
                        .collect();
                    if actual != fixed || sel.end > reading.len() || sel.start >= sel.end {
                        return Err(D512Error::ProtectedRange);
                    }
                }
                if candidates.contains(&text) {
                    continue;
                }
                candidates.push(text);
                engine.push(-out.cost);
                report.candidate_indices.push(i);
            }
            if candidates.len() > 16 {
                return Err(D512Error::OutOfLimits);
            }
            if candidates.len() < 2 {
                return Ok(());
            }
            let req = SentenceRequest {
                context: self.context.clone(),
                reading,
                candidates,
            };
            self.calibration = Some(D512Calibration {
                reading: req.reading.clone(),
                candidates: req
                    .candidates
                    .iter()
                    .zip(&engine)
                    .map(|(text, raw)| D512CalibrationCandidate {
                        text: text.clone(),
                        raw: *raw,
                        model_logp: None,
                        fused: None,
                    })
                    .collect(),
                default_index: 0,
                winner_index: 0,
                applied: false,
                suggested: false,
                weight: self.options.weight,
                margin: self.options.margin,
                mode: self.options.mode,
            });
            let model = self
                .scorer
                .as_mut()
                .ok_or(D512Error::Inference)?
                .score(&req, deadline)?;
            if Instant::now() >= deadline {
                return Err(D512Error::Timeout);
            }
            let (chosen, gap) = choose(&engine, &model, self.options.weight, self.options.margin)?;
            if let Some(c) = self.calibration.as_mut() {
                for (candidate, model) in c.candidates.iter_mut().zip(&model) {
                    candidate.model_logp = Some(*model);
                    candidate.fused = Some(candidate.raw + c.weight * model);
                }
                c.winner_index = (1..c.candidates.len()).fold(0, |best, i| {
                    if c.candidates[i].fused > c.candidates[best].fused {
                        i
                    } else {
                        best
                    }
                });
                c.applied = self.options.mode == D512Mode::Automatic && chosen != 0;
                c.suggested = self.options.mode == D512Mode::Suggestion && chosen != 0;
            }
            report.model_scores = model;
            if chosen != 0 {
                report.selected_index = report.candidate_indices[chosen];
                report.suggestion = Some(D512Suggestion {
                    text: req.candidates[chosen].clone(),
                    outcome_index: report.selected_index,
                    fused_gap: gap,
                });
            }
            Ok(())
        })();
        if let Err(error) = attempt {
            report.fallback = Some(error);
            report.selected_index = 0;
            report.suggestion = None;
        }
        if report.selected_index != 0 {
            if self.options.mode == D512Mode::Suggestion {
                self.suggested_path = outcomes[report.selected_index]
                    .intervals
                    .iter()
                    .map(|interval| {
                        if !interval.is_phrase {
                            return None;
                        }
                        Some(Selection {
                            start: interval.start,
                            end: interval.end,
                            wid: strings.get_wid(&interval.text)?,
                        })
                    })
                    .collect();
            } else {
                let winner = outcomes.remove(report.selected_index);
                outcomes.insert(0, winner);
            }
        }
        self.report_snapshot = Some(ReportSnapshot {
            composition: com.clone(),
            committed_prefix: self.editor_committed_prefix,
        });
        report.elapsed = start.elapsed();
        self.last = Some(report);
        outcomes
    }
}

pub type SharedD512State = Arc<Mutex<D512State>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D512Error {
    InvalidConfig,
    MissingFiles,
    InvalidModel,
    RuntimeUnavailable,
    TelemetryNotDisabled,
    OutOfLimits,
    Oov,
    BadReading,
    ProtectedRange,
    Timeout,
    Busy,
    NonFinite,
    Inference,
}
impl fmt::Display for D512Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "D512 {self:?}")
    }
}
impl std::error::Error for D512Error {}

fn choose(
    engine: &[f64],
    model: &[f64],
    weight: f64,
    margin: f64,
) -> Result<(usize, f64), D512Error> {
    if engine.is_empty()
        || engine.len() != model.len()
        || !weight.is_finite()
        || !margin.is_finite()
        || weight < 0.
        || margin < 0.
    {
        return Err(D512Error::InvalidConfig);
    }
    if engine.iter().chain(model).any(|s| !s.is_finite()) {
        return Err(D512Error::NonFinite);
    }
    if engine.iter().any(|s| *s > engine[0]) {
        return Err(D512Error::InvalidConfig);
    }
    let scores: Vec<_> = engine
        .iter()
        .zip(model)
        .map(|(e, m)| e + weight * m)
        .collect();
    if scores.iter().any(|s| !s.is_finite()) {
        return Err(D512Error::NonFinite);
    }
    let best = (1..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
    let gap = scores[best] - scores[0];
    Ok((if best != 0 && gap > margin { best } else { 0 }, gap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversion::{Interval, Selection, Symbol},
        syl,
        zhuyin::Bopomofo::*,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fake {
        calls: Arc<AtomicUsize>,
        response: Result<Vec<f64>, D512Error>,
        delay: Duration,
    }
    impl SentenceScorer for Fake {
        fn score(&mut self, r: &SentenceRequest, _d: Instant) -> Result<Vec<f64>, D512Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(r.reading.len(), 2);
            std::thread::sleep(self.delay);
            self.response.clone()
        }
    }
    fn fixture(
        response: Result<Vec<f64>, D512Error>,
        delay: Duration,
    ) -> (
        D512State,
        Arc<AtomicUsize>,
        Composition,
        StringTable,
        Vec<Outcome>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut state = D512State::default();
        state
            .configure(
                Some(Box::new(Fake {
                    calls: calls.clone(),
                    response,
                    delay,
                })),
                D512Options::default(),
            )
            .unwrap();
        let mut com = Composition::new();
        com.push(Symbol::from(syl![D, AI, TONE4]));
        com.push(Symbol::from(syl![B, I, AU, TONE3]));
        let outs = [("代表", 2.), ("代錶", 3.)]
            .into_iter()
            .map(|(s, cost)| Outcome {
                cost,
                intervals: vec![Interval {
                    start: 0,
                    end: 2,
                    text: s.into(),
                    is_phrase: true,
                }],
            })
            .collect();
        (state, calls, com, StringTable::new(), outs)
    }
    fn text(out: &Outcome) -> String {
        out.intervals.iter().map(|i| i.text.as_ref()).collect()
    }

    #[test]
    fn adapter_applies_model_and_exposes_suggestion() {
        let (mut s, calls, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        let ranked = s.rerank(&com, &strings, out);
        assert_eq!(text(&ranked[0]), "代錶");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let report = s.report().unwrap();
        assert_eq!(report.selected_index, 1);
        assert_eq!(report.suggestion.unwrap().fused_gap, 1.0);
    }

    #[test]
    fn default_uses_mac_automatic_balanced_parameters() {
        let options = D512Options::default();
        assert_eq!(options.mode, D512Mode::Automatic);
        assert_eq!(options.weight, 0.25);
        assert_eq!(options.margin, 0.5);
        assert_eq!(options.timeout, Duration::from_millis(100));
    }

    #[test]
    fn default_margin_requires_more_than_half_point() {
        let (mut state, _, com, strings, out) = fixture(Ok(vec![-10., -4.]), Duration::ZERO);
        assert_eq!(state.rerank(&com, &strings, out.clone()), out);
        assert!(state.report().unwrap().suggestion.is_none());
    }

    #[test]
    fn single_unique_candidate_does_not_call_scorer() {
        for mode in [D512Mode::Automatic, D512Mode::Suggestion] {
            for duplicate in [false, true] {
                let (mut state, calls, com, strings, mut out) =
                    fixture(Err(D512Error::Inference), Duration::ZERO);
                state.options.mode = mode;
                out.truncate(1);
                if duplicate {
                    out.push(out[0].clone());
                }
                assert_eq!(state.rerank(&com, &strings, out.clone()), out);
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                let report = state.report().unwrap();
                assert_eq!(report.selected_index, 0);
                assert_eq!(report.candidate_indices, [0]);
                assert!(report.suggestion.is_none());
                assert!(report.fallback.is_none());
                assert!(report.model_scores.is_empty());
            }
        }
    }

    #[test]
    fn calibration_keeps_deduplicated_scores_and_pre_margin_winner() {
        for (mode, margin, applied, suggested) in [
            (D512Mode::Automatic, 0.5, true, false),
            (D512Mode::Suggestion, 0.5, false, true),
            (D512Mode::Automatic, 2.0, false, false),
            (D512Mode::Suggestion, 2.0, false, false),
        ] {
            let (mut state, calls, com, strings, mut out) =
                fixture(Ok(vec![-10., -2.]), Duration::ZERO);
            state.options.mode = mode;
            state.options.margin = margin;
            out.insert(1, out[0].clone());
            let ranked = state.rerank(&com, &strings, out.clone());
            let c = state.calibration_for(&com, 0).unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert_eq!(c.reading, ["ㄉㄞˋ", "ㄅㄧㄠˇ"]);
            assert_eq!((c.default_index, c.winner_index), (0, 1));
            assert_eq!((c.mode, c.weight, c.margin), (mode, 0.25, margin));
            assert_eq!((c.applied, c.suggested), (applied, suggested));
            assert_eq!(c.candidates.len(), 2);
            for (candidate, text, raw, model, fused) in [
                (&c.candidates[0], "代表", -2., -10., -4.5),
                (&c.candidates[1], "代錶", -3., -2., -3.5),
            ] {
                assert_eq!(candidate.text, text);
                assert_eq!(candidate.raw, raw);
                assert_eq!(candidate.model_logp, Some(model));
                assert_eq!(candidate.fused, Some(fused));
            }
            assert_eq!(text(&ranked[0]), if applied { "代錶" } else { "代表" });
            assert_eq!(
                state.report().unwrap().selected_index,
                if applied || suggested { 2 } else { 0 }
            );
        }
    }

    #[test]
    fn calibration_failed_scores_are_all_absent() {
        for (response, delay) in [
            (Err(D512Error::Inference), Duration::ZERO),
            (Err(D512Error::Timeout), Duration::ZERO),
            (Ok(vec![0.]), Duration::ZERO),
            (Ok(vec![0., f64::NAN]), Duration::ZERO),
            (Ok(vec![-10., -2.]), Duration::from_millis(5)),
        ] {
            let (mut state, _, com, strings, out) = fixture(response, delay);
            state.options.timeout = Duration::from_millis(1);
            assert_eq!(state.rerank(&com, &strings, out.clone()), out);
            let c = state.calibration_for(&com, 0).unwrap();
            assert_eq!(c.candidates.len(), 2);
            assert_eq!(c.candidates[0].raw, -2.);
            assert_eq!(c.candidates[1].raw, -3.);
            assert!(
                c.candidates
                    .iter()
                    .all(|c| c.model_logp.is_none() && c.fused.is_none())
            );
            assert_eq!((c.default_index, c.winner_index), (0, 0));
            assert!(!c.applied && !c.suggested);
        }
    }

    #[test]
    fn calibration_debug_redacts_readings_and_candidate_text() {
        let (mut state, _, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        state.rerank(&com, &strings, out);
        let c = state.calibration_for(&com, 0).unwrap();
        let debug = format!("{c:?} {:?} {state:?}", c.candidates);
        for sensitive in ["代表", "代錶", "ㄉㄞˋ", "ㄅㄧㄠˇ"] {
            assert!(!debug.contains(sensitive));
        }
    }

    #[test]
    fn calibration_skips_unscored_evaluations_and_invalidates_old_data() {
        let (mut state, _, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        state.rerank(&com, &strings, out.clone());
        assert!(state.calibration_for(&com, 0).is_some());
        for mode in [D512Mode::Off, D512Mode::Automatic] {
            state.options.mode = mode;
            state.rerank(&com, &strings, out[..1].to_vec());
            assert!(state.calibration_for(&com, 0).is_none());
        }
        state.rerank(&com, &strings, out.clone());
        assert!(state.calibration_for(&com, 0).is_some());
        state.set_context("新的前文");
        assert!(state.calibration_for(&com, 0).is_none());
        state.rerank(&com, &strings, out.clone());
        state.configure(None, D512Options::default()).unwrap();
        state.rerank(&com, &strings, out);
        assert!(state.calibration_for(&com, 0).is_none());
    }

    #[test]
    fn suggestion_mode_keeps_every_native_outcome_while_reporting_winner() {
        let (mut state, calls, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        state.options.mode = D512Mode::Suggestion;
        assert_eq!(state.rerank(&com, &strings, out.clone()), out);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let report = state.report().unwrap();
        assert_eq!(report.selected_index, 1);
        assert_eq!(report.suggestion.unwrap().outcome_index, 1);
    }

    #[test]
    fn scorer_errors_bad_lengths_nonfinite_and_late_results_keep_original_order() {
        for response in [
            Err(D512Error::Inference),
            Err(D512Error::Timeout),
            Ok(vec![0.]),
            Ok(vec![0., f64::NAN]),
        ] {
            let (mut s, _, com, strings, out) = fixture(response, Duration::ZERO);
            let original = out.clone();
            assert_eq!(s.rerank(&com, &strings, out), original);
            assert!(s.report().unwrap().fallback.is_some());
        }
        let (mut s, _, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::from_millis(5));
        s.options.timeout = Duration::from_millis(1);
        let original = out.clone();
        assert_eq!(s.rerank(&com, &strings, out), original);
        assert_eq!(s.report().unwrap().fallback, Some(D512Error::Timeout));
    }

    #[test]
    fn off_and_absent_scorer_do_not_call_transport() {
        let (mut s, calls, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        s.options.mode = D512Mode::Off;
        let original = out.clone();
        assert_eq!(s.rerank(&com, &strings, out), original);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn manual_ranges_are_checked_before_scoring() {
        let (mut s, calls, mut com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        let wid = strings.intern("表");
        com.push_selection(Selection {
            start: 1,
            end: 2,
            wid,
        });
        let original = out.clone();
        assert_eq!(s.rerank(&com, &strings, out), original);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.report().unwrap().fallback,
            Some(D512Error::ProtectedRange)
        );
    }

    #[test]
    fn fusion_uses_log10_cost_scale_and_strict_margin() {
        assert_eq!(choose(&[-2., -3.], &[-10., -2.], 0.25, 0.25), Ok((1, 1.0)));
        assert_eq!(choose(&[-2., -3.], &[-10., -5.], 0.25, 0.25), Ok((0, 0.25)));
        assert_eq!(choose(&[-2., -3.], &[-10., -2.], 0., 0.), Ok((0, 0.0)));
    }

    #[test]
    fn invalid_or_nonfinite_fusion_never_changes_baseline() {
        for (e, m, w, t) in [
            (vec![-2., -1.], vec![1., 2.], 0.25, 0.25),
            (vec![-2.], vec![f64::NAN], 0.25, 0.25),
            (vec![-2.], vec![], 0.25, 0.25),
            (vec![-2.], vec![1.], f64::INFINITY, 0.25),
            (vec![-2.], vec![1.], 0.25, -1.),
        ] {
            assert!(choose(&e, &m, w, t).is_err());
        }
    }
}
