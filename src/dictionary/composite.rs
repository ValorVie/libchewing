use std::sync::Arc;

use crate::{
    dictionary::LookupStrategy,
    lm::StaticDict,
    model::Candidate,
    user::{HistoryDict, UserDict},
    zhuyin::Syllable,
};

#[derive(Clone, Debug)]
pub struct CompositeDict {
    inner: Arc<CompositeDictInner>,
}

#[derive(Debug)]
struct CompositeDictInner {
    static_dict: StaticDict,
    rare_dict: StaticDict,
    history_dict: HistoryDict,
    user_dict: UserDict,
}

impl CompositeDict {
    pub fn new(
        static_dict: StaticDict,
        rare_dict: StaticDict,
        history_dict: HistoryDict,
        user_dict: UserDict,
    ) -> CompositeDict {
        CompositeDict {
            inner: Arc::new(CompositeDictInner {
                static_dict,
                rare_dict,
                history_dict,
                user_dict,
            }),
        }
    }

    pub fn lookup(&self, syllables: &[Syllable], strategy: LookupStrategy) -> Vec<Candidate> {
        // base value
        let mut res: Vec<_> = self
            .inner
            .static_dict
            .lookup(syllables, strategy)
            .into_iter()
            .map(|wid| (wid, f64::NEG_INFINITY, None))
            .collect();
        // ponytail: Keep rare-only readings available, without a rarity adjustment.
        for wid in self.inner.rare_dict.lookup(syllables, strategy) {
            if !res.iter().any(|cand| cand.0 == wid) {
                res.push((wid, f64::NEG_INFINITY, None));
            }
        }
        // history boost
        for (wid, hist_prob) in self.inner.history_dict.unigram(syllables, strategy) {
            if let Some(pos) = res.iter().position(|cand| cand.0 == wid) {
                res[pos].1 = hist_prob;
            } else {
                res.push((wid, hist_prob, None));
            }
        }
        // user boost
        for (wid, user_pref) in self.inner.user_dict.lookup(syllables, strategy) {
            if let Some(pos) = res.iter().position(|w| w.0 == wid) {
                res[pos].2 = Some(user_pref);
            } else {
                res.push((wid, f64::NEG_INFINITY, Some(user_pref)));
            }
        }
        let banned = self.inner.user_dict.lookup_banned(syllables, strategy);
        res.into_iter()
            .filter(|cand| !banned.contains(&cand.0))
            .map(|cand| Candidate::Word {
                wid: cand.0,
                hist_prob: cand.1,
                user_pref: cand.2,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversion::{Composition, Decoder, LatticeBuilder, Symbol},
        dictionary::StringTableBuilder,
        lm::{StaticDictBuilder, StaticLm},
        syl,
        zhuyin::Bopomofo::*,
    };

    #[test]
    fn rare_readings_use_normal_candidate_scores() {
        let mut strings = StringTableBuilder::new();
        strings.insert("冊");
        strings.insert("測");
        let strings = strings.build();
        let keys = [syl![C, E, TONE4]];
        let mut words = StaticDictBuilder::new();
        for word in ["冊", "測"] {
            words.insert(&keys, strings.get_wid(word).unwrap());
        }
        let words = words.build();
        let mut rare = StaticDictBuilder::new();
        rare.insert(&keys, strings.get_wid("測").unwrap());
        let history = HistoryDict::new(strings.clone());
        let user = UserDict::new(strings.clone());
        let decoder = Decoder {
            lm: StaticLm::new(),
            hist: history.clone(),
            lambda: Decoder::LAMBDA,
        };
        let mut com = Composition::new();
        com.push(Symbol::from(keys[0]));
        let costs = |rare| {
            let lattice = LatticeBuilder {
                dict: CompositeDict::new(words.clone(), rare, history.clone(), user.clone()),
                lookup_strategy: LookupStrategy::Standard,
            }
            .build_lattice(&com);
            decoder
                .decoden(lattice, 2)
                .into_iter()
                .map(|h| h.cost)
                .collect::<Vec<_>>()
        };
        let normal = costs(StaticDict::new());
        assert_eq!(normal.len(), 2);
        assert_eq!(normal[0], normal[1]);
        assert_eq!(costs(rare.build()), normal);
    }

    #[test]
    fn rare_only_candidates_remain_available_without_penalty() {
        let mut strings = StringTableBuilder::new();
        strings.insert("測");
        let strings = strings.build();
        let keys = [syl![C, E, TONE4]];
        let mut rare = StaticDictBuilder::new();
        let wid = strings.get_wid("測").unwrap();
        rare.insert(&keys, wid);
        let dict = CompositeDict::new(
            StaticDict::new(),
            rare.build(),
            HistoryDict::new(strings.clone()),
            UserDict::new(strings),
        );
        let candidates = dict.lookup(&keys, LookupStrategy::Standard);
        assert_eq!(candidates.len(), 1);
        assert!(
            matches!(candidates[0], Candidate::Word { wid: actual, user_pref: None, .. } if actual == wid)
        );
    }
}
