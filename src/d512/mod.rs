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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D512Options {
    pub mode: D512Mode,
    pub weight: f64,
    pub margin: f64,
    /// Per latest-word hit in automatic mode; independent of the suggestion bonus.
    pub personal_bonus: f64,
    pub suggestion_personal_bonus: f64,
    pub timeout: Duration,
}
impl Default for D512Options {
    fn default() -> Self {
        Self {
            mode: D512Mode::Automatic,
            weight: 0.25,
            margin: 0.5,
            personal_bonus: 2.0,
            suggestion_personal_bonus: 2.0,
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
    pub changes: Vec<D512Change>,
}
/// Changed winning interval; start/end use composition symbol indices, end exclusive.
#[derive(Clone)]
pub struct D512Change {
    pub start: usize,
    pub end: usize,
    pub text: String,
}
impl fmt::Debug for D512Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512Change")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for D512Suggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512Suggestion")
            .field("outcome_index", &self.outcome_index)
            .field("fused_gap", &self.fused_gap)
            .field("changes", &self.changes.len())
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
    pub personal_hits: usize,
    pub model_logp: Option<f64>,
    pub fused: Option<f64>,
}
impl fmt::Debug for D512CalibrationCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D512CalibrationCandidate")
            .field("personal_hits", &self.personal_hits)
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
    pub bonus: f64,
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
            .field("bonus", &self.bonus)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

// Input identity of the last completed scoring attempt; never printed by Debug.
struct ReportSnapshot {
    composition: Composition,
    committed_prefix: u64,
    context: String,
    options: D512Options,
}
impl ReportSnapshot {
    fn matches(
        &self,
        current: &Composition,
        committed_prefix: u64,
        context: &str,
        options: D512Options,
    ) -> bool {
        self.committed_prefix == committed_prefix
            && self.context == context
            && self.options == options
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
    successful_native: Option<Vec<Outcome>>,
    editor_committed_prefix: u64,
    suggested_path: Option<Vec<Selection>>,
    personal: Option<crate::user::candidate_ranking::CandidateRankingStore>,
    personal_revision: u64,
    word_ranges: Vec<std::ops::Range<usize>>,
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
            || !options.personal_bonus.is_finite()
            || !options.suggestion_personal_bonus.is_finite()
            || options.personal_bonus < 0.
            || options.suggestion_personal_bonus < 0.
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
    pub(crate) fn prepare_personal(
        &mut self,
        store: Option<&crate::user::candidate_ranking::CandidateRankingStore>,
        ranges: Vec<std::ops::Range<usize>>,
    ) {
        let revision = store.map(|s| s.revision()).unwrap_or(0);
        if self.personal.is_some() != store.is_some()
            || self.personal_revision != revision
            || self.word_ranges != ranges
        {
            self.invalidate();
        }
        self.personal = store.cloned();
        self.personal_revision = revision;
        self.word_ranges = ranges;
    }
    fn personal_hits(&self, com: &Composition, out: &Outcome) -> usize {
        let Some(store) = &self.personal else {
            return 0;
        };
        out.intervals
            .iter()
            .filter(|i| {
                if !i.is_phrase
                    || i.end > com.len()
                    || i.text.chars().count() != i.len()
                    || (i.len() == 1
                        && self
                            .word_ranges
                            .iter()
                            .any(|r| r.start < i.end && i.start < r.end))
                {
                    return false;
                }
                let Some(keys) = crate::editor::personal::keys(&com.symbols()[i.start..i.end])
                else {
                    return false;
                };
                store.latest(&keys).as_deref() == Some(i.text.as_ref())
            })
            .count()
    }
    pub(crate) fn set_context(&mut self, context: &str) {
        let context: String = context
            .chars()
            .rev()
            .take(64)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if self.context != context {
            self.context = context;
            self.invalidate();
        }
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
        if self
            .personal
            .as_ref()
            .is_some_and(|store| store.revision() != self.personal_revision)
        {
            return false;
        }
        self.report_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.matches(current, committed_prefix, &self.context, self.options)
        })
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
        self.successful_native = None;
        self.suggested_path = None;
    }
    pub(crate) fn rerank(
        &mut self,
        com: &Composition,
        strings: &StringTable,
        mut outcomes: Vec<Outcome>,
    ) -> Vec<Outcome> {
        if self.snapshot_matches(com, self.editor_committed_prefix)
            && self.successful_native.as_ref() == Some(&outcomes)
            && let Some(report) = self.last.as_ref()
        {
            if self.options.mode == D512Mode::Automatic && report.selected_index != 0 {
                let winner = outcomes.remove(report.selected_index);
                outcomes.insert(0, winner);
            }
            return outcomes;
        }
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
            let mut personal_hits = vec![];
            let bonus = if self.personal.is_none() {
                0.
            } else if self.options.mode == D512Mode::Suggestion {
                self.options.suggestion_personal_bonus
            } else {
                self.options.personal_bonus
            };
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
                personal_hits.push(self.personal_hits(com, out));
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
                    .zip(&personal_hits)
                    .map(|((text, raw), hits)| D512CalibrationCandidate {
                        text: text.clone(),
                        raw: *raw,
                        personal_hits: *hits,
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
                bonus,
            });
            let model = self
                .scorer
                .as_mut()
                .ok_or(D512Error::Inference)?
                .score(&req, deadline)?;
            if Instant::now() >= deadline {
                return Err(D512Error::Timeout);
            }
            let adjusted: Vec<_> = engine
                .iter()
                .zip(&personal_hits)
                .map(|(e, h)| e + *h as f64 * bonus)
                .collect();
            let (chosen, gap) = if self.personal.is_some() {
                choose_adjusted(
                    &adjusted,
                    &model,
                    self.options.weight,
                    self.options.margin,
                    false,
                )?
            } else {
                choose(&engine, &model, self.options.weight, self.options.margin)?
            };
            if let Some(c) = self.calibration.as_mut() {
                for (candidate, model) in c.candidates.iter_mut().zip(&model) {
                    candidate.model_logp = Some(*model);
                    candidate.fused = Some(
                        candidate.raw + c.weight * model + candidate.personal_hits as f64 * c.bonus,
                    );
                }
                c.winner_index = (1..c.candidates.len()).fold(0, |best, i| {
                    if c.candidates[i].fused > c.candidates[best].fused {
                        i
                    } else {
                        best
                    }
                });
                c.applied = self.options.mode == D512Mode::Automatic && chosen != 0;
            }
            report.model_scores = model;
            if chosen != 0 {
                report.selected_index = report.candidate_indices[chosen];
                let changes: Vec<_> = outcomes[report.selected_index]
                    .intervals
                    .iter()
                    .filter(|interval| {
                        let same = outcomes[0].intervals.iter().any(|original| {
                            original.start == interval.start
                                && original.end == interval.end
                                && original.text == interval.text
                        });
                        !same
                            && !com.selections().iter().any(|selection| {
                                interval.start < selection.end && selection.start < interval.end
                            })
                    })
                    .map(|interval| D512Change {
                        start: interval.start,
                        end: interval.end,
                        text: interval.text.to_string(),
                    })
                    .collect();
                if !changes.is_empty() {
                    report.suggestion = Some(D512Suggestion {
                        text: req.candidates[chosen].clone(),
                        outcome_index: report.selected_index,
                        fused_gap: gap,
                        changes,
                    });
                }
            }
            if let Some(c) = self.calibration.as_mut() {
                c.suggested =
                    self.options.mode == D512Mode::Suggestion && report.suggestion.is_some();
            }
            Ok(())
        })();
        if let Err(error) = attempt {
            report.fallback = Some(error);
            report.selected_index = 0;
            report.suggestion = None;
        }
        if report.fallback.is_none() && !report.model_scores.is_empty() {
            self.successful_native = Some(outcomes.clone());
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
            context: self.context.clone(),
            options: self.options,
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
    choose_adjusted(engine, model, weight, margin, true)
}
fn choose_adjusted(
    engine: &[f64],
    model: &[f64],
    weight: f64,
    margin: f64,
    require_native_order: bool,
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
    if require_native_order && engine.iter().any(|s| *s > engine[0]) {
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

    fn intervals(parts: &[(usize, usize, &str)]) -> Vec<Interval> {
        parts
            .iter()
            .map(|(start, end, text)| Interval {
                start: *start,
                end: *end,
                text: (*text).into(),
                is_phrase: true,
            })
            .collect()
    }
    fn changes(suggestion: &D512Suggestion) -> Vec<(usize, usize, &str)> {
        suggestion
            .changes
            .iter()
            .map(|c| (c.start, c.end, c.text.as_str()))
            .collect()
    }

    #[test]
    fn hint_changes_single_word_in_both_modes_and_reused_result() {
        for mode in [D512Mode::Automatic, D512Mode::Suggestion] {
            let (mut state, calls, com, strings, out) =
                fixture(Ok(vec![-10., -2.]), Duration::ZERO);
            state.options.mode = mode;
            state.rerank(&com, &strings, out.clone());
            let suggestion = state.report().unwrap().suggestion.unwrap();
            assert_eq!(suggestion.text, "代錶");
            assert_eq!(changes(&suggestion), [(0, 2, "代錶")]);
            state.rerank(&com, &strings, out);
            assert_eq!(
                changes(&state.report().unwrap().suggestion.unwrap()),
                [(0, 2, "代錶")]
            );
            assert_eq!(calls.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn hint_changes_multiple_words_follow_winning_path_order() {
        let (mut state, _, com, strings, mut out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        out[0].intervals = intervals(&[(0, 1, "代"), (1, 2, "表")]);
        out[1].intervals = intervals(&[(0, 1, "帶"), (1, 2, "錶")]);
        state.rerank(&com, &strings, out);
        let suggestion = state.report().unwrap().suggestion.unwrap();
        assert_eq!(changes(&suggestion), [(0, 1, "帶"), (1, 2, "錶")]);
        let hint = suggestion
            .changes
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("／");
        assert_eq!(format!("⇥ {hint}"), "⇥ 帶／錶");
    }

    #[test]
    fn hint_changes_compare_word_ranges_for_split_and_merged_words() {
        let (mut state, _, com, strings, mut out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        out[1].intervals = intervals(&[(0, 1, "代"), (1, 2, "錶")]);
        state.rerank(&com, &strings, out.clone());
        assert_eq!(
            changes(&state.report().unwrap().suggestion.unwrap()),
            [(0, 1, "代"), (1, 2, "錶")]
        );
        out[0].intervals = intervals(&[(0, 1, "代"), (1, 2, "表")]);
        out[1].intervals = intervals(&[(0, 2, "代錶")]);
        state.rerank(&com, &strings, out);
        assert_eq!(
            changes(&state.report().unwrap().suggestion.unwrap()),
            [(0, 2, "代錶")]
        );
    }

    #[test]
    fn hint_changes_omit_same_words_at_same_ranges() {
        let (mut state, _, com, strings, mut out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        out[0].intervals = intervals(&[(0, 1, "代"), (1, 2, "表")]);
        out[1].intervals = intervals(&[(0, 1, "代"), (1, 2, "錶")]);
        state.rerank(&com, &strings, out);
        assert_eq!(
            changes(&state.report().unwrap().suggestion.unwrap()),
            [(1, 2, "錶")]
        );
    }

    #[test]
    fn hint_changes_exclude_protected_overlap_but_keep_adjacent_word() {
        let (mut state, _, mut com, strings, mut out) =
            fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        com.push_selection(Selection {
            start: 0,
            end: 1,
            wid: strings.intern("代"),
        });
        out[1].intervals = intervals(&[(0, 1, "代"), (1, 2, "錶")]);
        state.rerank(&com, &strings, out);
        assert_eq!(
            changes(&state.report().unwrap().suggestion.unwrap()),
            [(1, 2, "錶")]
        );
    }

    #[test]
    fn hint_changes_no_eligible_words_means_no_suggestion_without_changing_choice() {
        for mode in [D512Mode::Automatic, D512Mode::Suggestion] {
            let (mut state, _, mut com, strings, out) =
                fixture(Ok(vec![-10., -2.]), Duration::ZERO);
            state.options.mode = mode;
            com.push_selection(Selection {
                start: 0,
                end: 1,
                wid: strings.intern("代"),
            });
            let ranked = state.rerank(&com, &strings, out.clone());
            assert!(state.report().unwrap().suggestion.is_none());
            assert_eq!(state.report().unwrap().selected_index, 1);
            assert_eq!(
                text(&ranked[0]),
                if mode == D512Mode::Automatic {
                    "代錶"
                } else {
                    "代表"
                }
            );
            let calibration = state.calibration_for(&com, 0).unwrap();
            assert!(!calibration.suggested);
            assert_eq!(calibration.applied, mode == D512Mode::Automatic);
            let mut same = out;
            same[1].intervals = same[0].intervals.clone();
            state.rerank(&com, &strings, same);
            assert!(state.report().unwrap().suggestion.is_none());
        }
    }

    #[test]
    fn hint_changes_match_exact_start_for_repeated_readings() {
        let (mut state, _, mut com, strings, mut out) =
            fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        com.replace(1, Symbol::from(syl![D, AI, TONE4]));
        out[0].intervals = intervals(&[(0, 1, "代"), (1, 2, "戴")]);
        out[1].intervals = intervals(&[(0, 1, "戴"), (1, 2, "代")]);
        state.rerank(&com, &strings, out);
        assert_eq!(
            changes(&state.report().unwrap().suggestion.unwrap()),
            [(0, 1, "戴"), (1, 2, "代")]
        );
    }

    #[test]
    fn hint_changes_debug_redacts_standalone_and_nested_text() {
        let change = D512Change {
            start: 0,
            end: 2,
            text: "代錶".into(),
        };
        assert!(!format!("{change:?}").contains("代錶"));
        let (mut state, _, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        state.rerank(&com, &strings, out);
        let report = state.report().unwrap();
        let suggestion = report.suggestion.as_ref().unwrap();
        assert!(!format!("{report:?} {suggestion:?} {:?}", suggestion.changes).contains("代錶"));
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
    fn commit_reuse_retains_successful_report_and_calibration_in_both_modes() {
        for mode in [D512Mode::Automatic, D512Mode::Suggestion] {
            let (mut state, calls, com, strings, out) =
                fixture(Ok(vec![-10., -2.]), Duration::ZERO);
            state.options.mode = mode;
            state.set_context("已確認前文");
            let ranked = state.rerank(&com, &strings, out.clone());
            assert_eq!(
                text(&ranked[0]),
                if mode == D512Mode::Automatic {
                    "代錶"
                } else {
                    "代表"
                }
            );
            let report = state.report_for(&com, 0).unwrap();
            state.set_context("已確認前文");
            assert!(state.report_for(&com, 0).is_some());
            assert_eq!(state.rerank(&com, &strings, out), ranked);
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            let reused = state.report_for(&com, 0).unwrap();
            assert_eq!(reused.elapsed, report.elapsed);
            assert_eq!(reused.selected_index, 1);
            assert_eq!(reused.model_scores, [-10., -2.]);
            assert_eq!(state.calibration_for(&com, 0).unwrap().winner_index, 1);
        }
    }

    #[test]
    fn commit_reuse_compares_normalized_context_and_keeps_only_last_input() {
        let (mut state, calls, com, strings, out) = fixture(Ok(vec![-10., -2.]), Duration::ZERO);
        let suffix = "𠮷".repeat(64);
        state.set_context(&format!("舊{suffix}"));
        let original = state.rerank(&com, &strings, out.clone());
        state.set_context(&format!("新{suffix}"));
        assert_eq!(state.rerank(&com, &strings, out.clone()), original);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let mut changed = out.clone();
        changed[1].cost = 4.;
        state.rerank(&com, &strings, changed);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        state.rerank(&com, &strings, out);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn commit_reuse_rejects_changed_snapshot_context_options_or_native_paths() {
        for case in 0..11 {
            let (mut state, calls, mut com, strings, mut out) =
                fixture(Ok(vec![-10., -2.]), Duration::ZERO);
            state.rerank(&com, &strings, out.clone());
            match case {
                0 => state.set_context("不同前文"),
                1 => state.options.weight = 0.5,
                2 => state.options.margin = 2.,
                3 => state.options.mode = D512Mode::Suggestion,
                4 => state.options.timeout = Duration::from_millis(200),
                5 => state.set_editor_prefix(1),
                6 => com.replace(1, Symbol::from(syl![B, I, AU, TONE2])),
                7 => com.set_gap(1, crate::conversion::Gap::Break),
                8 => com.push_selection(Selection {
                    start: 0,
                    end: 1,
                    wid: strings.intern("代"),
                }),
                9 => out[0].cost = 2.1,
                _ => out[1].intervals[0].is_phrase = false,
            }
            state.rerank(&com, &strings, out);
            assert_eq!(calls.load(Ordering::Relaxed), 2, "stale reuse case {case}");
        }
    }

    #[test]
    fn commit_reuse_does_not_turn_failed_attempts_into_successful_hits() {
        let (mut state, calls, com, strings, out) =
            fixture(Err(D512Error::Inference), Duration::ZERO);
        assert_eq!(state.rerank(&com, &strings, out.clone()), out);
        assert_eq!(state.rerank(&com, &strings, out.clone()), out);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(state.report().unwrap().fallback, Some(D512Error::Inference));
        assert!(
            state
                .calibration_for(&com, 0)
                .unwrap()
                .candidates
                .iter()
                .all(|c| c.model_logp.is_none())
        );
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
