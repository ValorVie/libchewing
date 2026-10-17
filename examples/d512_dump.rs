//! Offline A1 JSONL adapter. Only explicit system data; no editor or user I/O.
use chewing::{
    conversion::{
        ChewingEngine, Composition, ConversionEngine, Decoder, LatticeBuilder, Outcome, Symbol,
    },
    dictionary::{CompositeDict, LookupStrategy, StringTable},
    lm::{LoadMode, StaticDict, StaticLm},
    user::{HistoryDict, UserDict},
    zhuyin::Syllable,
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

fn parse_reading(reading: &[Value]) -> Result<Vec<Syllable>, String> {
    if reading.is_empty() || reading.len() > 32 {
        return Err("expected 1..32 syllables".into());
    }
    reading
        .iter()
        .map(|v| {
            let s = v.as_str().ok_or("syllable must be a string")?;
            if s.is_empty() || s.chars().all(|c| "ˉˊˇˋ˙".contains(c)) {
                return Err("empty syllable".into());
            }
            let normalized = match s.strip_prefix('˙') {
                Some(base) => format!("{base}˙"),
                None => s.trim_end_matches('ˉ').to_owned(),
            };
            normalized.parse().map_err(|e| format!("{s}: {e}"))
        })
        .collect()
}

fn candidates(outcomes: Vec<Outcome>, length: usize) -> Result<Vec<Value>, String> {
    if outcomes.is_empty() {
        return Err("engine returned no outcomes".into());
    }
    let mut last_cost = f64::NEG_INFINITY;
    let mut seen = HashSet::new();
    let mut result = vec![];
    for out in outcomes {
        if !out.cost.is_finite() || out.cost < last_cost {
            return Err("nonfinite or unsorted engine costs".into());
        }
        last_cost = out.cost;
        let text: String = out.intervals.iter().map(|i| i.text.as_ref()).collect();
        if text.chars().count() != length {
            return Err("engine text does not cover syllables".into());
        }
        if !seen.insert(text.clone()) {
            continue;
        }
        let segments: Vec<_> = out.intervals.iter().map(|i| json!({"Start":i.start,"End":i.end,"Word":i.text,"Partial":false,"Correction":false})).collect();
        // Decoder::cost_fun sums negative log10 probabilities, including its
        // default rare-word preference. Preserve this native scale exactly.
        result.push(json!({"text":text,"log_score":-out.cost,"segments":segments}));
    }
    Ok(result)
}

fn system_engine(dir: &Path) -> Result<ChewingEngine, Box<dyn Error>> {
    let strings = StringTable::open_bin(dir.join("static_words.bin"))?;
    let hist = HistoryDict::new(strings.clone());
    let dict = CompositeDict::new(
        StaticDict::open(dir.join("static_dict.bin"))?,
        StaticDict::open(dir.join("rare_dict.bin"))?,
        hist.clone(),
        UserDict::new(strings.clone()),
    );
    Ok(ChewingEngine {
        word_lattice_builder: LatticeBuilder {
            dict,
            lookup_strategy: LookupStrategy::Standard,
        },
        decoder: Decoder {
            lm: StaticLm::open(dir.join("static_lm.bin"), LoadMode::Lazy)?,
            hist,
            lambda: Decoder::LAMBDA,
        },
        string_table: strings,
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: d512_dump SYSTEM_DATA_DIR INPUT_JSONL > requests.jsonl".into());
    }
    let engine = system_engine(Path::new(&args[0]))?;
    let mut rows = vec![];
    let mut ids = HashSet::new();
    for (i, line) in BufReader::new(File::open(&args[1])?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let id = row["id"]
            .as_str()
            .ok_or(format!("line {}: missing id", i + 1))?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate id {id}").into());
        }
        if !row["context"].is_string() || !row["gold"].is_string() || !row["reading"].is_array() {
            return Err(format!("line {}: invalid case schema", i + 1).into());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("no cases".into());
    }
    let total = rows.len();
    let (mut skipped, mut emitted, mut warmed) = (0, 0, false);
    let mut writer = BufWriter::new(std::io::stdout().lock());
    for row in rows {
        let result = (|| -> Result<Value, String> {
            let reading = parse_reading(row["reading"].as_array().unwrap())?;
            let mut comp = Composition::new();
            for syl in reading {
                comp.push(Symbol::from(syl));
            }
            if !warmed {
                for _ in 0..3 {
                    let _ = engine.convert(&comp);
                }
                warmed = true;
            }
            let start = Instant::now();
            let outcomes = engine.convert(&comp);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let raw_count = outcomes.len();
            let cs = candidates(outcomes, row["reading"].as_array().unwrap().len())?;
            let oracle = cs.iter().any(|c| c["text"] == row["gold"]);
            Ok(
                json!({"id":row["id"],"mode":"nbest","context":row["context"],"reading":row["reading"],"gold":row["gold"],"source":row["source"],"candidates":cs,"engine_default":0,"gold_in_candidates":oracle,"score_scale":"libchewing negative cost (log10)","engine_ms":ms,"raw_outcomes":raw_count}),
            )
        })();
        match result {
            Ok(req) => {
                serde_json::to_writer(&mut writer, &req)?;
                writeln!(writer)?;
                emitted += 1;
            }
            Err(reason) => {
                skipped += 1;
                eprintln!("{}", json!({"skip":row["id"],"reason":reason}));
            }
        }
    }
    writer.flush()?;
    eprintln!(
        "{}",
        json!({"total":total,"emitted":emitted,"skipped":skipped,"skip_fraction":skipped as f64/total as f64,"engine_warmup":3})
    );
    if emitted == 0 || skipped * 100 > total {
        return Err("conversion/output loss exceeds 1% (or no successes)".into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chewing::conversion::Interval;

    #[test]
    fn parses_tones_without_changing_readings() {
        let reading = vec![json!("ㄓㄨㄥ"), json!("ㄨㄣˊ"), json!("˙ㄉㄜ")];
        let result = parse_reading(&reading).unwrap();
        assert_eq!(
            result,
            vec![
                "ㄓㄨㄥ".parse().unwrap(),
                "ㄨㄣˊ".parse().unwrap(),
                "ㄉㄜ˙".parse().unwrap()
            ]
        );
    }

    #[test]
    fn rejects_invalid_and_empty_syllables() {
        for r in [vec![], vec![json!("")], vec![json!("abc")], vec![json!(12)]] {
            assert!(parse_reading(&r).is_err());
        }
    }

    fn out(text: &str, cost: f64) -> Outcome {
        Outcome {
            cost,
            intervals: vec![Interval {
                start: 0,
                end: text.chars().count(),
                is_phrase: true,
                text: text.into(),
            }],
        }
    }

    #[test]
    fn negates_log10_cost_and_keeps_best_unique_surface() {
        let cs = candidates(
            vec![out("公園", 2.0), out("公園", 3.0), out("公元", 4.0)],
            2,
        )
        .unwrap();
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0]["text"], "公園");
        assert_eq!(cs[0]["log_score"], -2.0);
        assert_eq!(cs[1]["log_score"], -4.0);
    }

    #[test]
    fn rejects_nonfinite_unsorted_and_incomplete_outcomes() {
        for os in [
            vec![],
            vec![out("字", 1.0)],
            vec![out("公園", f64::NAN)],
            vec![out("公園", 4.0), out("公元", 2.0)],
        ] {
            assert!(candidates(os, 2).is_err());
        }
    }
}
