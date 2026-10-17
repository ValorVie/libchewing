// SPDX-License-Identifier: LGPL-2.1-or-later
//! Replay the synthetic A1 tensor oracle, timing tensor creation, ORT, and extraction.
//! Usage: d512_perf MODEL_DIR ABSOLUTE_ORT_LIBRARY ORACLE_JSON OUTPUT_JSON [PROFILE_PREFIX]
//! Run sequentially with the Python probe; profiling changes timings.
//! D512_PERF_REPRO_AUTODEVICE=1 reproduces ort rc.14's implicit MaxEfficiency.

#[cfg(not(feature = "d512-ort"))]
fn main() {
    eprintln!("enable d512-ort");
    std::process::exit(2);
}

#[cfg(feature = "d512-ort")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ort::{
        logging::LogLevel,
        session::{RunOptions, Session, builder::GraphOptimizationLevel},
        value::Tensor,
    };
    use serde_json::{Value, json};
    use std::{path::Path, time::Instant};
    let a: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        (4..=5).contains(&a.len()),
        "MODEL_DIR ABSOLUTE_ORT_LIBRARY ORACLE_JSON OUTPUT_JSON [PROFILE_PREFIX]"
    );
    assert_eq!(std::env::var("ORT_DISABLE_TELEMETRY").as_deref(), Ok("1"));
    assert!(Path::new(&a[1]).is_absolute());
    let fixture: Value = serde_json::from_reader(std::fs::File::open(&a[2])?)?;
    let env = ort::init_from(&a[1])?.with_telemetry(false).build()?;
    env.set_log_level(LogLevel::Info);
    let mut builder = Session::builder(&env)?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(true).build()])?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(2)?
        .with_inter_threads(1)?
        .with_log_level(LogLevel::Info)?;
    if let Some(prefix) = a.get(4) {
        builder = builder.with_profiling(prefix)?;
    }
    if std::env::var("D512_PERF_REPRO_AUTODEVICE").as_deref() == Ok("1") {
        builder =
            builder.with_auto_device(ort::session::builder::AutoDevicePolicy::MaxEfficiency)?;
    } else {
        let devices: Vec<_> = env
            .devices()
            .filter(|d| d.ep().ok() == Some("CPUExecutionProvider"))
            .collect();
        assert!(!devices.is_empty());
        builder = builder.with_devices(devices, None)?;
    }
    let mut session = builder.commit_from_file(Path::new(&a[0]).join("model.onnx"))?;
    let options = RunOptions::new()?;
    let mut rows = vec![];
    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        for iteration in 0..13 {
            let mut phases = [0.0_f64; 3];
            let mut scores = vec![];
            for x in case["inputs"].as_array().unwrap() {
                let start = Instant::now();
                let int = |name: &str| -> Result<Tensor<i64>, ort::Error> {
                    let t = &x[name];
                    let shape: Vec<usize> = t["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let data: Vec<i64> = t["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect();
                    Tensor::from_array((shape, data.into_boxed_slice()))
                };
                let mask = &x["mask"];
                let shape: Vec<usize> = mask["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                let l = shape[2];
                let values: Vec<bool> = (0..l).flat_map(|r| (0..l).map(move |c| c <= r)).collect();
                assert!(
                    values
                        .iter()
                        .zip(mask["values"].as_array().unwrap())
                        .all(|(a, b)| *a == b.as_bool().unwrap())
                );
                let feeds = ort::inputs! {
                    "ids"=>int("ids")?, "bases"=>int("bases")?, "tones"=>int("tones")?,
                    "types"=>int("types")?, "positions"=>int("positions")?,
                    "mask"=>Tensor::from_array((shape,values.into_boxed_slice()))?,
                    "proj_pos"=>int("proj_pos")?, "target_node"=>int("target_node")?, "target_ids"=>int("target_ids")?
                };
                phases[0] += start.elapsed().as_secs_f64() * 1000.0;
                let start = Instant::now();
                let out = session.run_with_options(feeds, &options)?;
                phases[1] += start.elapsed().as_secs_f64() * 1000.0;
                let start = Instant::now();
                let (_, score) = out["scores"].try_extract_tensor::<f32>()?;
                scores.push(f64::from(score[0]));
                drop(out);
                phases[2] += start.elapsed().as_secs_f64() * 1000.0;
            }
            if iteration >= 3 {
                rows.push(json!({"case":case_index,"iteration":iteration-3,"tensor_ms":phases[0],"run_ms":phases[1],"extract_ms":phases[2],"scores":scores}));
            }
        }
    }
    let profile = if a.len() == 5 {
        Some(session.end_profiling()?)
    } else {
        None
    };
    std::fs::write(
        &a[3],
        serde_json::to_vec_pretty(&json!({"build":ort::info(),"profile":profile,"rows":rows}))?,
    )?;
    Ok(())
}
