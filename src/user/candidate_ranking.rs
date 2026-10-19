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
    sync::{Arc, Mutex, OnceLock, mpsc},
    thread::{self, JoinHandle},
};

#[derive(Clone)]
struct FileIo {
    #[cfg(test)]
    probe: Option<Arc<test_io::IoProbe>>,
}
impl Default for FileIo {
    fn default() -> Self {
        Self {
            #[cfg(test)]
            probe: test_io::CURRENT.with(|p| p.borrow().clone()),
        }
    }
}
impl FileIo {
    fn read(&self, path: &std::path::Path) -> io::Result<Vec<u8>> {
        #[cfg(test)]
        if let Some(probe) = &self.probe {
            probe.open();
        }
        fs::read(path)
    }
    fn open(&self, options: &OpenOptions, path: &std::path::Path) -> io::Result<fs::File> {
        #[cfg(test)]
        if let Some(probe) = &self.probe {
            probe.open();
        }
        options.open(path)
    }
}

pub(crate) const FILE_NAME: &str = "candidate-ranking.json";
type Groups = BTreeMap<Vec<String>, Group>;
type Event = (Vec<String>, String, u64);
#[derive(Clone, Default)]
pub(crate) struct CandidateRankingStore {
    shared: Arc<Shared>,
}
#[derive(Default)]
struct Shared {
    state: Arc<Mutex<State>>,
    path: Option<PathBuf>,
    io: FileIo,
    worker: OnceLock<Option<Worker>>,
}
#[derive(Default)]
struct State {
    snapshot: Arc<Groups>,
    // Unsaved increments stay visible while the worker owns an in-flight batch.
    local: Groups,
    pending: Vec<Event>,
    order: u64,
    revision: u64,
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
        Self {
            shared: Arc::new(Shared {
                path,
                ..Default::default()
            }),
        }
    }
    /// Activation only spawns the worker. Loading never runs on the caller.
    pub(crate) fn activate(&self) {
        let Some(path) = &self.shared.path else {
            return;
        };
        self.shared.worker.get_or_init(|| {
            match Worker::spawn(
                path.clone(),
                self.shared.io.clone(),
                self.shared.state.clone(),
            ) {
                Ok(worker) => Some(worker),
                Err(_) => {
                    log::warn!("Candidate ranking worker could not start; memory retained");
                    None
                }
            }
        });
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
        let mut s = self.shared.state.lock().unwrap();
        s.order = s.order.saturating_add(1);
        s.order
    }
    fn apply(groups: &mut Groups, keys: &[String], word: &str, order: u64) {
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
    pub(crate) fn record(&self, keys: &[String], word: &str, order: u64) {
        if keys.is_empty() || keys.iter().any(String::is_empty) || word.is_empty() {
            return;
        }
        let mut s = self.shared.state.lock().unwrap();
        Self::apply(&mut s.local, keys, word, order);
        if self.shared.path.is_some() {
            s.pending.push((keys.to_vec(), word.into(), order));
        }
        s.revision = s.revision.wrapping_add(1);
    }
    fn latest_of<'a>(base: Option<&'a Group>, delta: Option<&'a Group>) -> Option<&'a str> {
        match (base, delta) {
            (Some(base), Some(delta)) if delta.order > base.order => Some(&delta.latest),
            (Some(base), _) => Some(&base.latest),
            (_, Some(delta)) => Some(&delta.latest),
            _ => None,
        }
    }
    fn count_of(base: Option<&Group>, delta: Option<&Group>, word: &str) -> u64 {
        let count = |g: Option<&Group>| g.and_then(|g| g.counts.get(word)).copied().unwrap_or(0);
        count(base).saturating_add(count(delta))
    }
    #[cfg(test)]
    pub(crate) fn count(&self, keys: &[String], word: &str) -> u64 {
        let s = self.shared.state.lock().unwrap();
        Self::count_of(s.snapshot.get(keys), s.local.get(keys), word)
    }
    #[cfg(feature = "d512")]
    pub(crate) fn revision(&self) -> u64 {
        self.shared.state.lock().unwrap().revision
    }
    pub(crate) fn latest(&self, keys: &[String]) -> Option<String> {
        let s = self.shared.state.lock().unwrap();
        Self::latest_of(s.snapshot.get(keys), s.local.get(keys)).map(str::to_owned)
    }
    pub(crate) fn ranked(
        &self,
        pairs: &[(Vec<String>, String)],
        including_single_characters: bool,
    ) -> Vec<(Vec<String>, String)> {
        let mut slots: BTreeMap<&[String], Vec<usize>> = BTreeMap::new();
        for (i, (keys, _)) in pairs.iter().enumerate() {
            slots.entry(keys).or_default().push(i);
        }
        let (snapshot, local) = {
            let s = self.shared.state.lock().unwrap();
            let local: Groups = slots
                .keys()
                .filter_map(|k| s.local.get(*k).map(|g| (k.to_vec(), g.clone())))
                .collect();
            (s.snapshot.clone(), local)
        };
        let mut out = pairs.to_vec();
        for (keys, indices) in slots {
            if !including_single_characters && keys.len() == 1 {
                continue;
            }
            let base = snapshot.get(keys);
            let delta = local.get(keys);
            let Some(latest) = Self::latest_of(base, delta) else {
                continue;
            };
            let mut sorted = indices.clone();
            sorted.sort_by(|a, b| {
                let a = &pairs[*a].1;
                let b = &pairs[*b].1;
                (b == latest).cmp(&(a == latest)).then_with(|| {
                    Self::count_of(base, delta, b).cmp(&Self::count_of(base, delta, a))
                })
            });
            for (slot, source) in indices.into_iter().zip(sorted) {
                out[slot] = pairs[source].clone();
            }
        }
        out
    }
    fn send(&self, command: Command) {
        if let Some(Some(worker)) = self.shared.worker.get() {
            let _ = worker.sender.send(command);
        }
    }
    pub(crate) fn refresh(&self) {
        self.send(Command::Refresh);
    }
    /// Queue persistence; never wait on the caller or touch the filesystem.
    pub(crate) fn save(&self) {
        self.send(Command::Save);
    }
    /// Lifecycle barrier. Never call from a key or receipt confirmation path.
    pub(crate) fn flush(&self) {
        if let Some(Some(worker)) = self.shared.worker.get() {
            let (done, wait) = mpsc::channel();
            if worker.sender.send(Command::Flush(done)).is_ok() {
                let _ = wait.recv();
            }
        }
    }
}

enum Command {
    Save,
    Refresh,
    Flush(mpsc::Sender<()>),
    Shutdown,
}
struct Worker {
    sender: mpsc::Sender<Command>,
    handle: Option<JoinHandle<()>>,
}
impl Worker {
    fn spawn(path: PathBuf, io: FileIo, state: Arc<Mutex<State>>) -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("chewing-personal-ranking".into())
            .spawn(move || {
                let mut disk = DiskState {
                    path,
                    io,
                    groups: Arc::default(),
                    published: Arc::default(),
                    pending: vec![],
                    readonly: false,
                };
                disk.refresh();
                disk.publish(&state, vec![]);
                while let Ok(first) = receiver.recv() {
                    let mut refresh = false;
                    let mut shutdown = false;
                    let mut waiters = vec![];
                    let mut commands = vec![first];
                    commands.extend(receiver.try_iter());
                    for command in commands {
                        match command {
                            Command::Save => {}
                            Command::Refresh => refresh = true,
                            Command::Flush(done) => waiters.push(done),
                            Command::Shutdown => shutdown = true,
                        }
                    }
                    disk.pending.extend({
                        let mut s = state.lock().unwrap();
                        std::mem::take(&mut s.pending)
                    });
                    let acknowledged = if disk.readonly {
                        disk.pending.clear();
                        vec![]
                    } else if !disk.pending.is_empty() {
                        disk.persist(shutdown || !waiters.is_empty())
                    } else {
                        if refresh {
                            disk.refresh();
                        }
                        vec![]
                    };
                    disk.publish(&state, acknowledged);
                    for done in waiters {
                        let _ = done.send(());
                    }
                    if shutdown {
                        break;
                    }
                }
            })?;
        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
struct DiskState {
    path: PathBuf,
    io: FileIo,
    groups: Arc<Groups>,
    published: Arc<Groups>,
    pending: Vec<Event>,
    readonly: bool,
}
impl DiskState {
    fn read(&mut self) -> Option<Groups> {
        if self.readonly {
            return None;
        }
        let result = match self.io.read(&self.path) {
            Ok(bytes) => CandidateRankingStore::decode(&bytes),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Some(Groups::new()),
            Err(_) => None,
        };
        let Some(mut groups) = result else {
            self.readonly = true;
            log::warn!("Candidate ranking data could not be loaded; original file retained");
            return None;
        };
        for (keys, g) in &mut groups {
            g.order = self.groups.get(keys).map(|old| old.order).unwrap_or(0);
        }
        Some(groups)
    }
    fn refresh(&mut self) {
        if let Some(groups) = self.read() {
            self.groups = Arc::new(groups);
        }
    }
    fn publish(&mut self, state: &Mutex<State>, acknowledged: Vec<Event>) {
        if acknowledged.is_empty() && self.groups == self.published {
            return;
        }
        // Build and compare whole snapshots outside the UI mutex. Only exchange
        // the Arc and retire acknowledged increments under that mutex.
        let mut retired: BTreeMap<Vec<String>, BTreeMap<String, u64>> = BTreeMap::new();
        for (keys, word, _) in acknowledged {
            *retired.entry(keys).or_default().entry(word).or_default() += 1;
        }
        let old = {
            let mut s = state.lock().unwrap();
            let old = std::mem::replace(&mut s.snapshot, self.groups.clone());
            for (keys, counts) in retired {
                if let Some(g) = s.local.get_mut(&keys) {
                    for (word, count) in counts {
                        if let Some(n) = g.counts.get_mut(&word) {
                            *n -= count;
                            if *n == 0 {
                                g.counts.remove(&word);
                            }
                        }
                    }
                    if g.counts.is_empty() {
                        s.local.remove(&keys);
                    }
                }
            }
            s.revision = s.revision.wrapping_add(1);
            old
        };
        drop(old);
        self.published = self.groups.clone();
    }
    fn persist(&mut self, blocking: bool) -> Vec<Event> {
        let Some(parent) = self.path.parent().map(std::path::Path::to_owned) else {
            return vec![];
        };
        let mut created = false;
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_usize(self.pending.len());
        let temp = parent.join(format!(
            ".candidate-ranking-{}-{:x}.tmp",
            std::process::id(),
            hasher.finish()
        ));
        let write = (|| -> io::Result<bool> {
            fs::create_dir_all(&parent)?;
            let mut options = OpenOptions::new();
            options.write(true).read(true).create(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let lock = self
                .io
                .open(&options, &self.path.with_extension("json.lock"))?;
            if blocking {
                lock.lock()?;
            } else {
                match lock.try_lock() {
                    Ok(()) => {}
                    Err(fs::TryLockError::WouldBlock) => return Ok(false),
                    Err(fs::TryLockError::Error(e)) => return Err(e),
                }
            }
            let Some(mut groups) = self.read() else {
                return Ok(false);
            };
            for (keys, word, order) in &self.pending {
                CandidateRankingStore::apply(&mut groups, keys, word, *order);
            }
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
            let mut file = self.io.open(&options, &temp)?;
            created = true;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, &self.path)?;
            self.groups = Arc::new(groups);
            Ok(true)
        })();
        match write {
            Ok(true) => std::mem::take(&mut self.pending),
            Ok(false) => vec![],
            Err(_) => {
                if created {
                    let _ = fs::remove_file(&temp);
                }
                log::warn!("Candidate ranking data could not be saved; original file retained");
                vec![]
            }
        }
    }
}

#[cfg(test)]
fn open_store(path: Option<PathBuf>) -> CandidateRankingStore {
    let store = CandidateRankingStore::new(path);
    store.activate();
    store.flush();
    store
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
        let s = open_store(Some(path.clone()));
        record(&s, "世事");
        s.save();
        s.flush();
        let loaded = open_store(Some(path.clone()));
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
            let s = open_store(Some(path.clone()));
            record(&s, "試試");
            s.save();
            s.flush();
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
        let s = open_store(Some(path.clone()));
        record(&s, "試試");
        s.save();
        s.flush();
        let old = fs::read(&path).unwrap();
        fs::rename(&dir, &saved).unwrap();
        fs::write(&dir, b"blocked").unwrap();
        record(&s, "逝世");
        s.save();
        s.flush();
        assert_eq!(fs::read(saved.join("ranking.json")).unwrap(), old);
        fs::remove_file(&dir).unwrap();
        fs::rename(&saved, &dir).unwrap();
        record(&s, "世事");
        s.save();
        s.flush();
        let r = open_store(Some(path));
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
        let a = open_store(Some(path.clone()));
        let b = open_store(Some(path.clone()));
        let keys = vec!["ㄕˋ".into()];
        a.record(&keys, "試", a.next_order());
        a.save();
        a.flush();
        b.record(&keys, "世", b.next_order());
        b.save();
        b.flush();
        a.save();
        a.flush();
        let loaded = open_store(Some(path));
        assert_eq!(loaded.count(&keys, "試"), 1);
        assert_eq!(loaded.count(&keys, "世"), 1);
    }
    #[test]
    fn personal_store_no_changes_does_not_create_a_file_and_later_corruption_is_readonly() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let s = open_store(Some(path.clone()));
        s.save();
        s.flush();
        assert!(!path.exists());
        s.record(&["ㄕˋ".into()], "試", s.next_order());
        s.save();
        s.flush();
        fs::write(&path, b"corrupted after loading").unwrap();
        s.record(&["ㄕˋ".into()], "世", s.next_order());
        s.save();
        s.flush();
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
        let s = open_store(Some(path.clone()));
        let keys = vec!["ㄕˋ".into()];
        s.record(&keys, "世", s.next_order());
        s.save();
        assert!(!path.exists());
        drop(lock);
        s.save();
        s.flush();
        let reloaded = open_store(Some(path));
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
        let s = open_store(Some(path.clone()));
        s.record(&["ㄕˋ".into()], "世", s.next_order());
        s.save();
        s.flush();
        assert_eq!(s.count(&["ㄕˋ".into()], "世"), u64::MAX);
        let bad = serde_json::to_vec(&json!({"version":1,"groups":[group.clone(),group]})).unwrap();
        fs::write(&path, &bad).unwrap();
        let broken = open_store(Some(path.clone()));
        assert_eq!(broken.count(&["ㄕˋ".into()], "世"), 0);
        broken.record(&["ㄕˋ".into()], "試", broken.next_order());
        broken.save();
        broken.flush();
        assert_eq!(fs::read(path).unwrap(), bad);
    }
}

#[cfg(test)]
pub(crate) mod test_io {
    use std::{
        cell::RefCell,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
    };
    thread_local! {pub(super) static CURRENT:RefCell<Option<Arc<IoProbe>>>=const {RefCell::new(None)};}
    #[derive(Default)]
    pub(crate) struct IoProbe {
        pub(crate) opens: AtomicUsize,
        gate: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
    }
    impl IoProbe {
        pub(super) fn open(&self) {
            self.opens.fetch_add(1, Ordering::SeqCst);
            let gate = self.gate.lock().unwrap().take();
            if let Some((entered, release)) = gate {
                entered.send(()).unwrap();
                release.recv().unwrap();
            }
        }
        pub(crate) fn block_next_open(&self) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
            let (tx, rx) = mpsc::channel();
            let (release, wait) = mpsc::channel();
            *self.gate.lock().unwrap() = Some((tx, wait));
            (rx, release)
        }
    }
    pub(crate) struct Observation(Option<Arc<IoProbe>>);
    pub(crate) fn observe(probe: Arc<IoProbe>) -> Observation {
        Observation(CURRENT.with(|p| p.replace(Some(probe))))
    }
    impl Drop for Observation {
        fn drop(&mut self) {
            CURRENT.with(|p| {
                p.replace(self.0.take());
            });
        }
    }
}

#[cfg(test)]
mod async_storage_tests {
    use super::*;
    #[test]
    fn personal_fix1_initial_load_merges_choices_confirmed_while_loading() {
        use super::test_io::{IoProbe, observe};
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let keys = vec!["ㄕˋ".to_owned()];
        fs::write(
            &path,
            serde_json::to_vec(
                &json!({"version":1,"groups":[{"keyArray":keys,"latest":"世","counts":{"世":5}}]}),
            )
            .unwrap(),
        )
        .unwrap();
        let probe = Arc::new(IoProbe::default());
        let _observe = observe(probe.clone());
        let s = CandidateRankingStore::new(Some(path));
        let (entered, release) = probe.block_next_open();
        s.activate();
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        s.record(&keys, "世", s.next_order());
        s.save();
        let immediate = s.count(&keys, "世");
        release.send(()).unwrap();
        s.flush();
        assert_eq!(immediate, 1);
        assert_eq!(s.count(&keys, "世"), 6);
        assert_eq!(s.latest(&keys).as_deref(), Some("世"));
    }
    #[test]
    fn personal_fix1_child_writer() {
        let Some(path) = std::env::var_os("C1_RANKING_CHILD_FILE") else {
            return;
        };
        let word = std::env::var("C1_RANKING_CHILD_WORD").unwrap();
        let s = open_store(Some(path.into()));
        let keys = vec!["ㄕˋ".into()];
        for _ in 0..100 {
            s.record(&keys, &word, s.next_order());
            s.save();
        }
        s.flush();
        assert_eq!(s.count(&keys, &word), 100);
    }
    #[test]
    fn personal_fix1_four_processes_keep_all_confirmed_counts() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("ranking.json");
        let exe = std::env::current_exe().unwrap();
        let jobs: Vec<_> = ["試", "世", "事", "市"]
            .into_iter()
            .map(|word| {
                let exe = exe.clone();
                let path = path.clone();
                thread::spawn(move || {
                    std::process::Command::new(exe)
                        .args([
                            "--exact",
                            "user::candidate_ranking::async_storage_tests::personal_fix1_child_writer",
                            "--nocapture",
                        ])
                        .env("C1_RANKING_CHILD_FILE", path)
                        .env("C1_RANKING_CHILD_WORD", word)
                        .output()
                        .unwrap()
                })
            })
            .collect();
        for job in jobs {
            let output = job.join().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        let s = open_store(Some(path));
        for word in ["試", "世", "事", "市"] {
            assert_eq!(s.count(&["ㄕˋ".into()], word), 100);
        }
    }
}
