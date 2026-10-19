// SPDX-License-Identifier: LGPL-2.1-or-later
//! Independent, non-decaying manual-choice statistics (Mac Snapshot version 1).
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fmt,
    fs::{self, OpenOptions},
    hash::{BuildHasher, Hasher, RandomState},
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub(crate) const FILE_NAME: &str = "candidate-ranking.json";
#[derive(Clone, Default)]
pub(crate) struct CandidateRankingStore {
    inner: Arc<Mutex<State>>,
    path: Option<PathBuf>,
}
#[derive(Default)]
struct State {
    groups: BTreeMap<Vec<String>, Group>,
    order: u64,
    revision: u64,
    readonly: bool,
    pending: Vec<(Vec<String>, String, u64)>,
}
#[derive(Clone, PartialEq)]
struct Group {
    latest: String,
    counts: BTreeMap<String, u64>,
    order: u64,
}
impl fmt::Debug for CandidateRankingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CandidateRankingStore")
            .finish_non_exhaustive()
    }
}
impl CandidateRankingStore {
    pub(crate) fn new(path: Option<PathBuf>) -> Self {
        let mut state = State::default();
        if let Some(path) = &path {
            match fs::read(path) {
                Ok(bytes) => match Self::decode(&bytes) {
                    Some(groups) => state.groups = groups,
                    None => {
                        state.readonly = true;
                        log::warn!(
                            "Candidate ranking data could not be loaded; original file retained"
                        );
                    }
                },
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(_) => {
                    state.readonly = true;
                    log::warn!(
                        "Candidate ranking data could not be loaded; original file retained"
                    );
                }
            }
        }
        Self {
            inner: Arc::new(Mutex::new(state)),
            path,
        }
    }
    fn decode(bytes: &[u8]) -> Option<BTreeMap<Vec<String>, Group>> {
        let data: Value = serde_json::from_slice(bytes).ok()?;
        if data.get("version")?.as_u64()? != 1 {
            return None;
        }
        let mut result = BTreeMap::new();
        for row in data.get("groups")?.as_array()? {
            let keys: Vec<String> = row
                .get("keyArray")?
                .as_array()?
                .iter()
                .map(|v| v.as_str().map(str::to_owned))
                .collect::<Option<_>>()?;
            let latest = row.get("latest")?.as_str()?.to_owned();
            let counts: BTreeMap<String, u64> = row
                .get("counts")?
                .as_object()?
                .iter()
                .map(|(k, v)| Some((k.clone(), v.as_u64()?)))
                .collect::<Option<_>>()?;
            if keys.is_empty()
                || keys.iter().any(String::is_empty)
                || counts.is_empty()
                || counts.iter().any(|(w, n)| w.is_empty() || *n == 0)
                || !counts.contains_key(&latest)
                || result.contains_key(&keys)
            {
                return None;
            }
            result.insert(
                keys,
                Group {
                    latest,
                    counts,
                    order: 0,
                },
            );
        }
        Some(result)
    }
    pub(crate) fn next_order(&self) -> u64 {
        let mut s = self.inner.lock().unwrap();
        s.order = s.order.saturating_add(1);
        s.order
    }
    pub(crate) fn record(&self, keys: &[String], word: &str, order: u64) {
        if keys.is_empty() || keys.iter().any(String::is_empty) || word.is_empty() {
            return;
        }
        let mut s = self.inner.lock().unwrap();
        Self::apply(&mut s.groups, keys, word, order);
        if self.path.is_some() {
            s.pending.push((keys.to_vec(), word.to_owned(), order));
        }
        s.revision = s.revision.wrapping_add(1);
    }
    #[cfg(test)]
    pub(crate) fn count(&self, keys: &[String], word: &str) -> u64 {
        self.inner
            .lock()
            .unwrap()
            .groups
            .get(keys)
            .and_then(|g| g.counts.get(word).copied())
            .unwrap_or(0)
    }
    #[cfg(feature = "d512")]
    pub(crate) fn revision(&self) -> u64 {
        self.inner.lock().unwrap().revision
    }
    pub(crate) fn latest(&self, keys: &[String]) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .groups
            .get(keys)
            .map(|g| g.latest.clone())
    }
    pub(crate) fn ranked(
        &self,
        pairs: &[(Vec<String>, String)],
        including_single_characters: bool,
    ) -> Vec<(Vec<String>, String)> {
        let s = self.inner.lock().unwrap();
        let mut out = pairs.to_vec();
        let mut slots: BTreeMap<&[String], Vec<usize>> = BTreeMap::new();
        for (i, (keys, _)) in pairs.iter().enumerate() {
            slots.entry(keys).or_default().push(i);
        }
        for (keys, indices) in slots {
            if !including_single_characters && keys.len() == 1 {
                continue;
            }
            let Some(group) = s.groups.get(keys) else {
                continue;
            };
            let mut sorted = indices.clone();
            sorted.sort_by(|a, b| {
                let a = &pairs[*a].1;
                let b = &pairs[*b].1;
                (b == &group.latest)
                    .cmp(&(a == &group.latest))
                    .then_with(|| {
                        group
                            .counts
                            .get(b)
                            .unwrap_or(&0)
                            .cmp(group.counts.get(a).unwrap_or(&0))
                    })
            });
            for (slot, source) in indices.into_iter().zip(sorted) {
                out[slot] = pairs[source].clone();
            }
        }
        out
    }
    fn apply(groups: &mut BTreeMap<Vec<String>, Group>, keys: &[String], word: &str, order: u64) {
        let g = groups.entry(keys.to_vec()).or_insert_with(|| Group {
            latest: word.into(),
            counts: BTreeMap::new(),
            order: 0,
        });
        let n = g.counts.entry(word.into()).or_default();
        *n = n.saturating_add(1);
        if order >= g.order {
            g.latest = word.into();
            g.order = order;
        }
    }
    fn merged(&self, s: &mut State) -> Option<BTreeMap<Vec<String>, Group>> {
        let path = self.path.as_ref()?;
        let mut groups = match fs::read(path) {
            Ok(bytes) => match Self::decode(&bytes) {
                Some(g) => g,
                None => {
                    s.readonly = true;
                    log::warn!(
                        "Candidate ranking data could not be loaded; original file retained"
                    );
                    return None;
                }
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => BTreeMap::new(),
            Err(_) => {
                s.readonly = true;
                log::warn!("Candidate ranking data could not be loaded; original file retained");
                return None;
            }
        };
        // Session-local selection order survives disk reloads and delayed receipts.
        for (keys, g) in &mut groups {
            g.order = s.groups.get(keys).map(|old| old.order).unwrap_or(0);
        }
        for (keys, word, order) in &s.pending {
            Self::apply(&mut groups, keys, word, *order);
        }
        Some(groups)
    }
    pub(crate) fn refresh(&self) {
        if self.path.is_none() {
            return;
        }
        let mut s = self.inner.lock().unwrap();
        if s.readonly {
            return;
        }
        if let Some(groups) = self.merged(&mut s)
            && groups != s.groups
        {
            s.groups = groups;
            s.revision = s.revision.wrapping_add(1);
        }
    }
    pub(crate) fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let mut state = self.inner.lock().unwrap();
        if state.readonly || state.pending.is_empty() {
            return;
        }
        let Some(parent) = path.parent() else {
            return;
        };
        let mut created = false;
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_usize(state.pending.len());
        let temp = parent.join(format!(
            ".candidate-ranking-{}-{:x}.tmp",
            std::process::id(),
            hasher.finish()
        ));
        let write = (|| -> io::Result<()> {
            fs::create_dir_all(parent)?;
            let mut options = OpenOptions::new();
            options.write(true).read(true).create(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            // Lock a stable sidecar; replacing the snapshot never changes the lock inode.
            // Contention retains pending events for the next flush without blocking a key.
            let lock = options.open(path.with_extension("json.lock"))?;
            match lock.try_lock() {
                Ok(()) => {}
                Err(fs::TryLockError::WouldBlock) => return Ok(()),
                Err(fs::TryLockError::Error(error)) => return Err(error),
            }
            let Some(groups) = self.merged(&mut state) else {
                return Ok(());
            };
            let rows: Vec<_> = groups
                .iter()
                .map(|(keys, g)| json!({"keyArray":keys,"latest":g.latest,"counts":g.counts}))
                .collect();
            let bytes = serde_json::to_vec(&json!({"version":1,"groups":rows})).unwrap();
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            created = true;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, path)?;
            state.groups = groups;
            state.pending.clear();
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        })();
        if write.is_err() {
            if created {
                let _ = fs::remove_file(&temp);
            }
            log::warn!("Candidate ranking data could not be saved; original file retained");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn k() -> Vec<String> {
        vec!["ㄕˋ".into(), "ㄕˋ".into()]
    }
    fn record(s: &CandidateRankingStore, w: &str) {
        s.record(&k(), w, s.next_order());
    }
    fn pairs() -> Vec<(Vec<String>, String)> {
        ["試試", "逝世", "世事"].map(|w| (k(), w.into())).to_vec()
    }
    #[test]
    fn personal_store_mac_recent_counts_and_stable_ties() {
        let s = CandidateRankingStore::default();
        assert_eq!(s.ranked(&pairs(), true), pairs());
        for (w, n) in [("試試", 10), ("逝世", 5), ("世事", 1)] {
            for _ in 0..n {
                record(&s, w);
            }
        }
        assert_eq!(
            s.ranked(&pairs(), true),
            vec![pairs()[2].clone(), pairs()[0].clone(), pairs()[1].clone()]
        );
        assert_eq!(
            ["試試", "逝世", "世事"].map(|w| s.count(&k(), w)),
            [10, 5, 1]
        );
    }
    #[test]
    fn personal_store_mac_groups_keep_slots_and_absent_latest_is_not_added() {
        let s = CandidateRankingStore::default();
        record(&s, "逝世");
        record(&s, "世事");
        let mut p = pairs();
        p.insert(1, (vec!["ㄕˊ".into()], "時".into()));
        assert_eq!(
            s.ranked(&p, true),
            vec![p[3].clone(), p[1].clone(), p[2].clone(), p[0].clone()]
        );
        assert_eq!(
            s.ranked(&pairs()[..2], true),
            vec![pairs()[1].clone(), pairs()[0].clone()]
        );
    }
    #[test]
    fn personal_store_mac_delayed_receipts_follow_selection_order() {
        let s = CandidateRankingStore::default();
        let earlier = s.next_order();
        let later = s.next_order();
        s.record(&k(), "逝世", later);
        s.record(&k(), "世事", earlier);
        assert_eq!(s.latest(&k()).as_deref(), Some("逝世"));
    }
    #[test]
    fn personal_store_mac_snapshot_roundtrip_and_readonly_invalid_files() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("new/ranking.json");
        let s = CandidateRankingStore::new(Some(path.clone()));
        record(&s, "世事");
        s.save();
        let loaded = CandidateRankingStore::new(Some(path.clone()));
        assert_eq!(loaded.count(&k(), "世事"), 1);
        let json: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(json["version"], 1);
        assert_eq!(json["groups"][0]["keyArray"], serde_json::json!(k()));
        for bad in [
            "broken",
            "{\"version\":999,\"groups\":[]}",
            "{\"version\":1,\"groups\":[{\"keyArray\":[\"ㄕˋ\"],\"latest\":\"世\",\"counts\":{\"世\":0}}]}",
        ] {
            fs::write(&path, bad).unwrap();
            let s = CandidateRankingStore::new(Some(path.clone()));
            record(&s, "試試");
            s.save();
            assert_eq!(fs::read(&path).unwrap(), bad.as_bytes());
            assert_eq!(s.count(&k(), "試試"), 1);
        }
    }
    #[test]
    fn personal_store_mac_save_failure_preserves_previous_snapshot_and_recovers() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("dir");
        let saved = root.path().join("saved");
        let path = dir.join("ranking.json");
        let s = CandidateRankingStore::new(Some(path.clone()));
        record(&s, "試試");
        s.save();
        let old = fs::read(&path).unwrap();
        fs::rename(&dir, &saved).unwrap();
        fs::write(&dir, b"blocked").unwrap();
        record(&s, "逝世");
        s.save();
        assert_eq!(fs::read(saved.join("ranking.json")).unwrap(), old);
        fs::remove_file(&dir).unwrap();
        fs::rename(&saved, &dir).unwrap();
        record(&s, "世事");
        s.save();
        let r = CandidateRankingStore::new(Some(path));
        assert_eq!(
            ["試試", "逝世", "世事"].map(|w| r.count(&k(), w)),
            [1, 1, 1]
        );
    }
}

#[cfg(test)]
mod storage_boundary_tests {
    use super::*;
    #[test]
    fn personal_store_two_independent_writers_do_not_erase_counts() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let a = CandidateRankingStore::new(Some(path.clone()));
        let b = CandidateRankingStore::new(Some(path.clone()));
        let keys = vec!["ㄕˋ".into()];
        a.record(&keys, "試", a.next_order());
        a.save();
        b.record(&keys, "世", b.next_order());
        b.save();
        a.save();
        let loaded = CandidateRankingStore::new(Some(path));
        assert_eq!(loaded.count(&keys, "試"), 1);
        assert_eq!(loaded.count(&keys, "世"), 1);
    }
    #[test]
    fn personal_store_no_changes_does_not_create_a_file_and_later_corruption_is_readonly() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let s = CandidateRankingStore::new(Some(path.clone()));
        s.save();
        assert!(!path.exists());
        s.record(&["ㄕˋ".into()], "試", s.next_order());
        s.save();
        fs::write(&path, b"corrupted after loading").unwrap();
        s.record(&["ㄕˋ".into()], "世", s.next_order());
        s.save();
        assert_eq!(fs::read(&path).unwrap(), b"corrupted after loading");
    }
}

#[cfg(test)]
mod additional_storage_tests {
    use super::*;
    #[test]
    fn personal_store_busy_lock_retains_pending_and_flush_retries() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path.with_extension("json.lock"))
            .unwrap();
        lock.lock().unwrap();
        let s = CandidateRankingStore::new(Some(path.clone()));
        let keys = vec!["ㄕˋ".into()];
        s.record(&keys, "世", s.next_order());
        s.save();
        assert!(!path.exists());
        drop(lock);
        s.save();
        let reloaded = CandidateRankingStore::new(Some(path));
        assert_eq!(reloaded.count(&keys, "世"), 1);
    }
    #[test]
    fn personal_store_schema_validation_is_all_or_nothing_and_counts_saturate() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let group = json!({"keyArray":["ㄕˋ"],"latest":"世","counts":{"世":u64::MAX}});
        fs::write(
            &path,
            serde_json::to_vec(&json!({"version":1,"groups":[group.clone()]})).unwrap(),
        )
        .unwrap();
        let s = CandidateRankingStore::new(Some(path.clone()));
        s.record(&["ㄕˋ".into()], "世", s.next_order());
        s.save();
        assert_eq!(s.count(&["ㄕˋ".into()], "世"), u64::MAX);
        let bad = serde_json::to_vec(&json!({"version":1,"groups":[group.clone(),group]})).unwrap();
        fs::write(&path, &bad).unwrap();
        let broken = CandidateRankingStore::new(Some(path.clone()));
        assert_eq!(broken.count(&["ㄕˋ".into()], "世"), 0);
        broken.record(&["ㄕˋ".into()], "試", broken.next_order());
        broken.save();
        assert_eq!(fs::read(path).unwrap(), bad);
    }
}
