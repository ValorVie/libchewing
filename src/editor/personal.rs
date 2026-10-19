// SPDX-License-Identifier: LGPL-2.1-or-later
//! Delivery receipts and temporary, non-learning composition constraints.
use crate::{
    conversion::{Composition, Gap, Outcome, Selection, Symbol},
    dictionary::{CompositeDict, LookupStrategy, StringTable},
    model::Candidate,
    user::candidate_ranking::{CandidateRankingStore, WeakCandidateRankingStore},
    zhuyin::SyllableVec,
};
use std::{
    cell::{Cell, RefCell},
    fmt,
    ops::Range,
    rc::Rc,
};

pub(crate) fn keys(symbols: &[Symbol]) -> Option<Vec<String>> {
    symbols
        .iter()
        .map(|s| {
            s.to_syllable()
                .map(|s| s.to_string().trim_end_matches('ˉ').to_owned())
        })
        .collect()
}

/// Pins only a temporary copy. Live selections remain owned by the caller.
pub(crate) fn preferred(
    com: &Composition,
    base: &Outcome,
    store: &CandidateRankingStore,
    dict: &CompositeDict,
    strings: &StringTable,
) -> Option<Composition> {
    let mut candidates = vec![];
    for start in 0..com.len() {
        for end in start + 1..=com.len().min(start + SyllableVec::MAX_LEN) {
            if com
                .selections()
                .iter()
                .any(|s| start < s.end && s.start < end)
                || (start + 1..end).any(|i| com.gap(i) == Some(Gap::Break))
            {
                continue;
            }
            let Some(k) = keys(&com.symbols()[start..end]) else {
                continue;
            };
            let Some(word) = store.latest(&k) else {
                continue;
            };
            if word.chars().count() != end - start {
                continue;
            }
            let syllables: Vec<_> = com.symbols()[start..end]
                .iter()
                .filter_map(Symbol::to_syllable)
                .collect();
            if let Some(wid) = dict
                .lookup(&syllables, LookupStrategy::Standard)
                .into_iter()
                .find_map(|c| match c {
                    Candidate::Word { wid, .. }
                        if strings.get_text(wid).as_deref() == Some(word.as_str()) =>
                    {
                        Some(wid)
                    }
                    _ => None,
                })
            {
                candidates.push(Selection { start, end, wid });
            }
        }
    }
    candidates.sort_by_key(|s| (std::cmp::Reverse(s.len()), s.start));
    let mut pins = vec![];
    let mut proposed = com.clone();
    for s in candidates {
        if pins.iter().any(|p: &Selection| p.intersect(&s)) {
            continue;
        }
        if s.len() == 1
            && base
                .intervals
                .iter()
                .any(|i| i.len() > 1 && i.start < s.end && s.start < i.end)
        {
            continue;
        }
        proposed.push_selection(s);
        pins.push(s);
    }
    (!pins.is_empty()).then_some(proposed)
}

type SharedChoice = Rc<RefCell<Choice>>;
struct Choice {
    keys: Vec<String>,
    word: String,
    tokens: Vec<u64>,
    order: u64,
    delivered: usize,
    cancelled: bool,
    learned: bool,
    epoch: Rc<Cell<bool>>,
}
#[derive(Clone)]
pub(crate) struct SelectionLearning {
    tokens: Vec<u64>,
    next: u64,
    choices: Vec<SharedChoice>,
    epoch: Rc<Cell<bool>>,
}
impl Default for SelectionLearning {
    fn default() -> Self {
        Self {
            tokens: vec![],
            next: 0,
            choices: vec![],
            epoch: Rc::new(Cell::new(true)),
        }
    }
}
impl fmt::Debug for SelectionLearning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectionLearning")
            .field("choices", &self.choices.len())
            .finish_non_exhaustive()
    }
}
#[derive(Default)]
pub(crate) struct Receipt(Vec<(SharedChoice, usize)>);
impl fmt::Debug for Receipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Receipt")
            .field("fragments", &self.0.len())
            .finish()
    }
}
/// Owned confirmation for an exact commit buffer. Keep it through the edit-session
/// callback; call `confirm(true)` only when that text was inserted unchanged.
/// Dropping a receipt rejects it. Debug never exposes readings or text.
/// Confirmation after the editor's store has been dropped is ignored.
#[derive(Debug)]
pub struct PersonalCandidateCommit {
    receipt: Option<Receipt>,
    store: WeakCandidateRankingStore,
}
impl PersonalCandidateCommit {
    pub(crate) fn new(receipt: Receipt, store: WeakCandidateRankingStore) -> Option<Self> {
        (!receipt.0.is_empty()).then_some(Self {
            receipt: Some(receipt),
            store,
        })
    }
    pub fn confirm(mut self, inserted_unchanged: bool) {
        let Some(store) = self.store.upgrade() else {
            return;
        };
        if let Some(receipt) = self.receipt.take() {
            receipt.confirm(inserted_unchanged, &store);
            store.save();
        }
    }
}
impl Drop for PersonalCandidateCommit {
    fn drop(&mut self) {
        if let Some(receipt) = self.receipt.take() {
            receipt.cancel();
        }
    }
}
impl Receipt {
    pub(crate) fn confirm(self, unchanged: bool, store: &CandidateRankingStore) {
        for (choice, count) in self.0 {
            let mut c = choice.borrow_mut();
            if !unchanged {
                c.cancelled = true;
            }
            if c.cancelled || c.learned || !c.epoch.get() {
                continue;
            }
            c.delivered += count;
            if c.delivered == c.keys.len() {
                c.learned = true;
                store.record(&c.keys, &c.word, c.order);
            }
        }
    }
    pub(crate) fn cancel(self) {
        for (c, _) in self.0 {
            c.borrow_mut().cancelled = true;
        }
    }
}
impl SelectionLearning {
    pub(crate) fn insert(&mut self, at: usize) {
        self.next = self.next.wrapping_add(1);
        self.tokens.insert(at, self.next);
    }
    pub(crate) fn replace(&mut self, at: usize) {
        self.next = self.next.wrapping_add(1);
        self.tokens[at] = self.next;
    }
    pub(crate) fn remove(&mut self, at: usize) {
        if at < self.tokens.len() {
            self.tokens.remove(at);
        }
    }
    pub(crate) fn remove_front(&mut self, n: usize) {
        self.tokens.drain(..n);
    }
    pub(crate) fn clear(&mut self) {
        for c in self.choices.drain(..) {
            c.borrow_mut().cancelled = true;
        }
        self.tokens.clear();
    }
    pub(crate) fn reset_epoch(&mut self, enabled: bool) {
        self.epoch.set(false);
        self.epoch = Rc::new(Cell::new(enabled));
    }
    pub(crate) fn record(
        &mut self,
        s: Selection,
        com: &Composition,
        strings: &StringTable,
        store: &CandidateRankingStore,
    ) {
        if !self.epoch.get() || s.start >= s.end || s.end > self.tokens.len() {
            return;
        }
        let Some(k) = keys(&com.symbols()[s.start..s.end]) else {
            return;
        };
        let Some(word) = strings.get_text(s.wid) else {
            return;
        };
        if word.chars().count() != k.len() {
            return;
        }
        let tokens = self.tokens[s.start..s.end].to_vec();
        self.choices.retain(|c| {
            let mut c = c.borrow_mut();
            if c.tokens.iter().any(|t| tokens.contains(t)) {
                c.cancelled = true;
                false
            } else {
                true
            }
        });
        self.choices.push(Rc::new(RefCell::new(Choice {
            keys: k,
            word: word.to_string(),
            tokens,
            order: store.next_order(),
            delivered: 0,
            cancelled: false,
            learned: false,
            epoch: self.epoch.clone(),
        })));
    }
    fn range(tokens: &[u64], c: &Choice, com: &Composition, output: &str) -> Option<Range<usize>> {
        if c.cancelled || c.tokens.is_empty() {
            return None;
        }
        let start = tokens.iter().position(|t| *t == c.tokens[0])?;
        let end = start + c.tokens.len();
        if tokens.get(start..end)? != c.tokens {
            return None;
        }
        let consumed = c.keys.len() - c.tokens.len();
        if keys(&com.symbols()[start..end])? != c.keys[consumed..] {
            return None;
        }
        let actual: String = output.chars().skip(start).take(end - start).collect();
        if actual != c.word.chars().skip(consumed).collect::<String>() {
            return None;
        }
        // Full choices must retain their selected node; suffixes use surviving token identities.
        if consumed == 0
            && !com
                .selections()
                .iter()
                .any(|s| s.start == start && s.end == end)
        {
            return None;
        }
        Some(start..end)
    }
    pub(crate) fn consume(&mut self, n: usize, com: &Composition, output: &str) -> Receipt {
        let mut fragments = vec![];
        self.choices.retain(|choice| {
            let mut c = choice.borrow_mut();
            let Some(range) = Self::range(&self.tokens, &c, com, output) else {
                c.cancelled = true;
                return false;
            };
            let count = n.min(range.end).saturating_sub(range.start);
            if count > 0 {
                fragments.push((choice.clone(), count));
                c.tokens.drain(..count);
            }
            !c.tokens.is_empty()
        });
        Receipt(fragments)
    }
}
