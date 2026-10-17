// SPDX-License-Identifier: LGPL-2.1-or-later
//! In-process ONNX implementation, for the scoring host and offline tools.
use super::SentenceScorer;
use super::{D512Error, SentenceRequest};
use ort::{
    environment::Environment,
    session::{
        RunOptions, Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::Tensor,
};
use sha2::{Digest, Sha256};
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
};

static RUNTIME: Mutex<Option<(PathBuf, Environment)>> = Mutex::new(None);

fn session_builder(env: &Environment) -> Result<SessionBuilder, D512Error> {
    Session::builder(env)
        .map_err(|_| D512Error::InvalidModel)?
        // ort rc.14 defaults to MaxEfficiency, which selects CoreML on macOS.
        // Its CPU provider only toggles the arena; reset device policy as well.
        .with_auto_device(ort::session::builder::AutoDevicePolicy::Default)
        .map_err(|_| D512Error::InvalidModel)?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(true).build()])
        .map_err(|_| D512Error::InvalidModel)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|_| D512Error::InvalidModel)?
        .with_intra_threads(2)
        .map_err(|_| D512Error::InvalidModel)?
        .with_inter_threads(1)
        .map_err(|_| D512Error::InvalidModel)
}

fn environment(path: &Path) -> Result<Environment, D512Error> {
    // Rust 2024 forbids mutating a Unix process environment while other threads
    // may run. Require the launcher to disable telemetry before starting us.
    if std::env::var("ORT_DISABLE_TELEMETRY").as_deref() != Ok("1") {
        return Err(D512Error::TelemetryNotDisabled);
    }
    if !path.is_absolute() || !path.is_file() {
        return Err(D512Error::RuntimeUnavailable);
    }
    let path = path
        .canonicalize()
        .map_err(|_| D512Error::RuntimeUnavailable)?;
    let mut runtime = RUNTIME.lock().map_err(|_| D512Error::Busy)?;
    if let Some((loaded, env)) = runtime.as_ref() {
        return if *loaded == path {
            Ok(env.clone())
        } else {
            Err(D512Error::RuntimeUnavailable)
        };
    }
    let env = ort::init_from(&path)
        .map_err(|_| D512Error::RuntimeUnavailable)?
        .with_name("chewing-d512")
        .with_telemetry(false)
        .build()
        .map_err(|_| D512Error::RuntimeUnavailable)?;
    *runtime = Some((path, env.clone()));
    Ok(env)
}

pub struct OrtSentenceScorer {
    vocab: Vocab,
    session: Session,
    _environment: Environment,
}
impl std::fmt::Debug for OrtSentenceScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtSentenceScorer").finish_non_exhaustive()
    }
}
impl OrtSentenceScorer {
    /// Load the immutable pinned release and an explicit ONNX Runtime library.
    /// Call this in the scoring host, never in a TIP process.
    pub fn load(model_directory: &Path, runtime_library: &Path) -> Result<Self, D512Error> {
        verify_model(model_directory)?;
        let vocab = Vocab::load(model_directory)?;
        let env = environment(runtime_library)?;
        let session = session_builder(&env)?
            .commit_from_file(model_directory.join("model.onnx"))
            .map_err(|_| D512Error::InvalidModel)?;
        Ok(Self {
            vocab,
            session,
            _environment: env,
        })
    }
    fn run(&mut self, x: &Input, options: &RunOptions) -> Result<f64, D512Error> {
        let l = x.ids.len();
        let n = x.target_ids.len();
        let int = |data: &[i64], shape: Vec<usize>| {
            Tensor::from_array((shape, data.to_vec().into_boxed_slice()))
                .map_err(|_| D512Error::Inference)
        };
        let feeds = ort::inputs! {
            "ids"=>int(&x.ids,vec![1,l])?,"bases"=>int(&x.bases,vec![1,l])?,"tones"=>int(&x.tones,vec![1,l])?,"types"=>int(&x.types,vec![1,l])?,"positions"=>int(&x.positions,vec![1,l])?,
            "mask"=>Tensor::from_array(([1,1,l,l],x.mask.clone().into_boxed_slice())).map_err(|_|D512Error::Inference)?,"proj_pos"=>int(&x.proj_pos,vec![n])?,"target_node"=>int(&x.target_node,vec![1,n])?,"target_ids"=>int(&x.target_ids,vec![1,n])?,
        };
        let outputs = self
            .session
            .run_with_options(feeds, options)
            .map_err(|_| D512Error::Inference)?;
        let (_, values) = outputs
            .get("scores")
            .ok_or(D512Error::InvalidModel)?
            .try_extract_tensor::<f32>()
            .map_err(|_| D512Error::InvalidModel)?;
        if values.len() != 1 {
            return Err(D512Error::InvalidModel);
        }
        let score = f64::from(values[0]);
        if !score.is_finite() {
            return Err(D512Error::NonFinite);
        }
        Ok(score)
    }
}
impl SentenceScorer for OrtSentenceScorer {
    fn score(
        &mut self,
        request: &SentenceRequest,
        deadline: Instant,
    ) -> Result<Vec<f64>, D512Error> {
        if Instant::now() >= deadline {
            return Err(D512Error::Timeout);
        }
        let prepared = self.vocab.prepare(request)?;
        let options = Arc::new(RunOptions::new().map_err(|_| D512Error::Inference)?);
        let (done, wait) = mpsc::sync_channel::<()>(1);
        let cancel = options.clone();
        let timer = thread::Builder::new()
            .name("d512-deadline".into())
            .spawn(move || {
                if matches!(
                    wait.recv_timeout(deadline.saturating_duration_since(Instant::now())),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let _ = cancel.terminate();
                }
            })
            .map_err(|_| D512Error::Busy)?;
        let result = prepared
            .iter()
            .map(|inputs| {
                let mut best = f64::NEG_INFINITY;
                for x in inputs {
                    if Instant::now() >= deadline {
                        return Err(D512Error::Timeout);
                    }
                    best = best.max(self.run(x, &options)?);
                }
                Ok(best)
            })
            .collect();
        drop(done);
        let _ = timer.join();
        if Instant::now() >= deadline {
            Err(D512Error::Timeout)
        } else {
            result
        }
    }
}

#[derive(Debug)]
struct Vocab {
    chars: HashMap<char, i64>,
    bases: HashMap<String, i64>,
    neutral: HashSet<String>,
}
impl Vocab {
    fn load(dir: &Path) -> Result<Self, D512Error> {
        let v: serde_json::Value = serde_json::from_reader(
            File::open(dir.join("vocab.json")).map_err(|_| D512Error::InvalidModel)?,
        )
        .map_err(|_| D512Error::InvalidModel)?;
        let m: serde_json::Value = serde_json::from_reader(
            File::open(dir.join("manifest.json")).map_err(|_| D512Error::InvalidModel)?,
        )
        .map_err(|_| D512Error::InvalidModel)?;
        if m["schema"] != "zuin-sequence-runtime-v1" {
            return Err(D512Error::InvalidModel);
        }
        let array = |value: &serde_json::Value| -> Result<Vec<String>, D512Error> {
            value
                .as_array()
                .ok_or(D512Error::InvalidModel)?
                .iter()
                .map(|s| s.as_str().map(String::from).ok_or(D512Error::InvalidModel))
                .collect()
        };
        let mut chars = HashMap::new();
        for (i, s) in array(&v["characters"])?.iter().enumerate() {
            let mut iter = s.chars();
            let c = iter.next().ok_or(D512Error::InvalidModel)?;
            if iter.next().is_some() {
                return Err(D512Error::InvalidModel);
            }
            chars.insert(c, i as i64 + 8);
        }
        let bases = array(&v["bases"])?
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as i64 + 2))
            .collect();
        Ok(Self {
            chars,
            bases,
            neutral: array(&m["neutral_tolerant_last"])?.into_iter().collect(),
        })
    }
    fn prepare(&self, r: &SentenceRequest) -> Result<Vec<Vec<Input>>, D512Error> {
        if !(1..=32).contains(&r.reading.len())
            || !(1..=16).contains(&r.candidates.len())
            || r.candidates
                .iter()
                .any(|c| c.chars().count() != r.reading.len())
        {
            return Err(D512Error::OutOfLimits);
        }
        let text = r.reading.len() == 1 && !r.context.is_empty();
        let mut sounds = vec![];
        let mut last_base = "";
        for reading in &r.reading {
            let (base, tone) = match reading.chars().last() {
                Some('ˊ') => (&reading[..reading.len() - 'ˊ'.len_utf8()], 2),
                Some('ˇ') => (&reading[..reading.len() - 'ˇ'.len_utf8()], 3),
                Some('ˋ') => (&reading[..reading.len() - 'ˋ'.len_utf8()], 4),
                Some('˙') => (&reading[..reading.len() - '˙'.len_utf8()], 5),
                Some(_) => (reading.as_str(), 1),
                None => return Err(D512Error::BadReading),
            };
            sounds.push((*self.bases.get(base).ok_or(D512Error::BadReading)?, tone));
            last_base = base;
        }
        let context: Vec<_> = r
            .context
            .chars()
            .rev()
            .take(64)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|c| self.chars.get(&c).copied().unwrap_or(4))
            .collect();
        let neutral =
            !text && self.neutral.contains(last_base) && sounds.last().is_some_and(|s| s.1 != 5);
        r.candidates
            .iter()
            .map(|c| {
                let ids: Vec<_> = c
                    .chars()
                    .map(|c| self.chars.get(&c).copied().ok_or(D512Error::Oov))
                    .collect::<Result<_, _>>()?;
                let mut inputs = vec![encode(&context, if text { &[] } else { &sounds }, &ids)];
                if neutral {
                    let mut alternative = sounds.clone();
                    if let Some(last) = alternative.last_mut() {
                        last.1 = 5;
                    }
                    inputs.push(encode(&context, &alternative, &ids));
                }
                Ok(inputs)
            })
            .collect()
    }
}
fn verify_model(dir: &Path) -> Result<(), D512Error> {
    const PINS: [(&str, &str); 3] = [
        (
            "model.onnx",
            "71af1f0e77b2752cc71cbc04bf19c8e8f2e765662246f766e65a2822bab4e66e",
        ),
        (
            "vocab.json",
            "556dffaffbf4c05211a846dcee55ef735cea68c5754f9bffcc6f9588124172b3",
        ),
        (
            "manifest.json",
            "f8d737cf44e9ff8f90e7108f1e9df4fece0cdcae254e050968fc6ef723ee1976",
        ),
    ];
    if PINS.iter().any(|(name, _)| !dir.join(name).is_file()) {
        return Err(D512Error::MissingFiles);
    }
    for (name, pin) in PINS {
        let mut file = File::open(dir.join(name)).map_err(|_| D512Error::InvalidModel)?;
        let mut hash = Sha256::new();
        let mut buf = [0; 65536];
        loop {
            let n = file.read(&mut buf).map_err(|_| D512Error::InvalidModel)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        if format!("{:x}", hash.finalize()) != pin {
            return Err(D512Error::InvalidModel);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct Input {
    ids: Vec<i64>,
    bases: Vec<i64>,
    tones: Vec<i64>,
    types: Vec<i64>,
    positions: Vec<i64>,
    mask: Vec<bool>,
    proj_pos: Vec<i64>,
    target_node: Vec<i64>,
    target_ids: Vec<i64>,
}

fn encode(context: &[i64], sounds: &[(i64, i64)], candidate: &[i64]) -> Input {
    let context = &context[context.len().saturating_sub(64)..];
    let h = context.len() + 2;
    let k = sounds.len();
    let n = candidate.len();
    let out = h + k;
    let len = out + n;
    let mut ids = vec![1];
    ids.extend_from_slice(context);
    ids.push(2);
    ids.extend(vec![0; k]);
    ids.push(3);
    ids.extend_from_slice(&candidate[..n - 1]);
    let mut bases = vec![0; h];
    bases.extend(sounds.iter().map(|s| s.0));
    bases.extend(vec![0; n]);
    let mut tones = vec![0; h];
    tones.extend(sounds.iter().map(|s| s.1));
    tones.extend(vec![0; n]);
    let mut types = vec![0; h];
    types.extend(vec![1; k]);
    types.push(0);
    types.extend(vec![2; n - 1]);
    Input {
        ids,
        bases,
        tones,
        types,
        positions: (0..len as i64).collect(),
        mask: (0..len)
            .flat_map(|r| (0..len).map(move |c| c <= r))
            .collect(),
        proj_pos: (out as i64..len as i64).collect(),
        target_node: (0..n as i64).collect(),
        target_ids: candidate.iter().map(|c| c - 6).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires D512_MODEL_DIR, D512_ORT_LIBRARY and ORT_DISABLE_TELEMETRY=1"]
    fn session_uses_only_cpu_kernels() {
        let model = std::env::var_os("D512_MODEL_DIR").unwrap();
        let runtime = std::env::var_os("D512_ORT_LIBRARY").unwrap();
        let env = environment(Path::new(&runtime)).unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let session = session_builder(&env)
            .unwrap()
            .with_profiling(scratch.path().join("cpu-regression"))
            .unwrap()
            .commit_from_file(Path::new(&model).join("model.onnx"))
            .unwrap();
        let mut scorer = OrtSentenceScorer {
            vocab: Vocab::load(Path::new(&model)).unwrap(),
            session,
            _environment: env,
        };
        let req = SentenceRequest {
            context: "今天去".into(),
            reading: vec!["ㄍㄨㄥ".into(), "ㄩㄢˊ".into()],
            candidates: vec!["公園".into(), "公元".into()],
        };
        scorer
            .score(&req, Instant::now() + std::time::Duration::from_secs(5))
            .unwrap();
        let profile = scorer.session.end_profiling().unwrap();
        let events: Vec<serde_json::Value> =
            serde_json::from_reader(File::open(profile).unwrap()).unwrap();
        let providers: HashSet<_> = events
            .iter()
            .filter(|e| e["cat"] == "Node")
            .filter_map(|e| e["args"]["provider"].as_str())
            .collect();
        assert!(
            !providers.is_empty(),
            "profile must contain executed kernels"
        );
        assert_eq!(providers, HashSet::from(["CPUExecutionProvider"]));
    }

    #[test]
    #[ignore = "requires D512_MODEL_DIR, D512_ORT_LIBRARY, D512_FIXTURE and ORT_DISABLE_TELEMETRY=1"]
    fn a1_tensor_and_score_fixture() {
        use serde_json::json;
        let model = std::env::var_os("D512_MODEL_DIR").unwrap();
        let runtime = std::env::var_os("D512_ORT_LIBRARY").unwrap();
        let fixture = std::env::var_os("D512_FIXTURE").unwrap();
        let mut scorer = OrtSentenceScorer::load(Path::new(&model), Path::new(&runtime)).unwrap();
        println!("native ORT build: {}", ort::info());
        assert_eq!(
            environment(Path::new("onnxruntime.dll")).err(),
            Some(D512Error::RuntimeUnavailable)
        );
        let v = Vocab::load(Path::new(&model)).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_reader(File::open(fixture).unwrap()).unwrap();
        let mut max_diff = 0.0_f64;
        let mut count = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let r = &case["request"];
            let req = SentenceRequest {
                context: r["context"].as_str().unwrap().into(),
                reading: r["reading"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().into())
                    .collect(),
                candidates: r["candidates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().into())
                    .collect(),
            };
            let prepared = v.prepare(&req).unwrap();
            let feeds:Vec<_>=prepared.iter().flatten().map(|x| {
                let l=x.ids.len();let n=x.target_ids.len();let int=|shape:Vec<usize>,values:&Vec<i64>|json!({"dtype":"int64","shape":shape,"values":values});
                json!({"ids":int(vec![1,l],&x.ids),"bases":int(vec![1,l],&x.bases),"tones":int(vec![1,l],&x.tones),"types":int(vec![1,l],&x.types),"positions":int(vec![1,l],&x.positions),"mask":{"dtype":"bool","shape":[1,1,l,l],"values":x.mask},"proj_pos":int(vec![n],&x.proj_pos),"target_node":int(vec![1,n],&x.target_node),"target_ids":int(vec![1,n],&x.target_ids)})
            }).collect();
            assert_eq!(json!(feeds), case["inputs"]);
            let scores = scorer
                .score(&req, Instant::now() + std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(scores.len(), case["scores"].as_array().unwrap().len());
            for (actual, expected) in scores.iter().zip(case["scores"].as_array().unwrap()) {
                let diff = (actual - expected.as_f64().unwrap()).abs();
                max_diff = max_diff.max(diff);
                count += 1;
                assert!(diff <= 1e-4, "score error {diff}");
            }
        }
        println!("A1 fixture: {count} scores; max_abs_diff={max_diff}");
        let req = SentenceRequest {
            context: "你".repeat(64),
            reading: vec!["ㄍㄨㄥ".into(), "ㄩㄢˊ".into()],
            candidates: vec!["公園".into(), "公元".into()],
        };
        let start = Instant::now();
        assert_eq!(
            scorer.score(&req, start + std::time::Duration::from_millis(1)),
            Err(D512Error::Timeout)
        );
        assert!(start.elapsed() < std::time::Duration::from_millis(200));
        let recovered = scorer
            .score(&req, Instant::now() + std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(recovered.iter().all(|s| s.is_finite()));
        println!("native 1ms deadline canceled; subsequent request recovered");
    }
    fn vocab() -> Vocab {
        Vocab {
            chars: [('吧', 8)].into(),
            bases: [("ㄅㄚ".into(), 2)].into(),
            neutral: ["ㄅㄚ".into()].into(),
        }
    }
    fn req(context: &str, reading: &str, candidate: &str) -> SentenceRequest {
        SentenceRequest {
            context: context.into(),
            reading: vec![reading.into()],
            candidates: vec![candidate.into()],
        }
    }
    #[test]
    fn neutral_fallback_covers_marked_tones_and_text_mode_uses_confirmed_context() {
        let v = vocab();
        let x = v.prepare(&req("", "ㄅㄚˊ", "吧")).unwrap();
        assert_eq!(x[0].len(), 2);
        assert_eq!(x[0][0].tones, [0, 0, 2, 0]);
        assert_eq!(x[0][1].tones, [0, 0, 5, 0]);
        let x = v.prepare(&req("外", "ㄅㄚˊ", "吧")).unwrap();
        assert_eq!(x[0].len(), 1);
        assert_eq!(x[0][0].ids, [1, 4, 2, 3]);
        assert_eq!(x[0][0].types, [0, 0, 0, 0]);
        let x = v.prepare(&req("", "ㄅㄚ˙", "吧")).unwrap();
        assert_eq!(x[0].len(), 1);
    }
    #[test]
    fn candidate_oov_bad_reading_and_empty_input_fall_back_before_inference() {
        let v = vocab();
        assert_eq!(v.prepare(&req("", "ㄅㄚ", "外")), Err(D512Error::Oov));
        assert_eq!(v.prepare(&req("", "bad", "吧")), Err(D512Error::BadReading));
        let mut r = req("", "ㄅㄚ", "吧");
        r.candidates.clear();
        assert_eq!(v.prepare(&r), Err(D512Error::OutOfLimits));
    }
    #[test]
    fn missing_and_altered_model_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(verify_model(dir.path()), Err(D512Error::MissingFiles));
        for name in ["model.onnx", "vocab.json", "manifest.json"] {
            std::fs::write(dir.path().join(name), b"{}").unwrap();
        }
        assert_eq!(verify_model(dir.path()), Err(D512Error::InvalidModel));
    }
    #[test]
    fn phonetic_tensor_protocol_and_context_tail() {
        let x = encode(&[8, 9], &[(2, 1), (3, 2)], &[10, 11]);
        assert_eq!(x.ids, [1, 8, 9, 2, 0, 0, 3, 10]);
        assert_eq!(x.bases, [0, 0, 0, 0, 2, 3, 0, 0]);
        assert_eq!(x.tones, [0, 0, 0, 0, 1, 2, 0, 0]);
        assert_eq!(x.types, [0, 0, 0, 0, 1, 1, 0, 2]);
        assert_eq!(x.positions, (0..8).collect::<Vec<_>>());
        assert_eq!(x.proj_pos, [6, 7]);
        assert_eq!(x.target_node, [0, 1]);
        assert_eq!(x.target_ids, [4, 5]);
        assert_eq!(
            x.mask,
            (0..8)
                .flat_map(|r| (0..8).map(move |c| c <= r))
                .collect::<Vec<_>>()
        );
        let long = encode(&vec![8; 70], &[(2, 1)], &[9]);
        assert_eq!(long.ids.len(), 68);
    }

    #[test]
    fn text_mode_has_no_reading_slots() {
        let x = encode(&[8], &[], &[9]);
        assert_eq!(x.ids, [1, 8, 2, 3]);
        assert_eq!(x.bases, [0, 0, 0, 0]);
        assert_eq!(x.tones, [0, 0, 0, 0]);
        assert_eq!(x.types, [0, 0, 0, 0]);
        assert_eq!(x.proj_pos, [3]);
    }
}
