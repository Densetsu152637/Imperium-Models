//! `ncm-bench --device ort-cpu --onnx-dir <export>`: batch-1 prefill and greedy decode speed of
//! an `ncm-export` directory through the `ncm-ort` backend (ONNX Runtime's CPU execution
//! provider). The weights are the export's, not random ones: export a model first.

use std::time::Instant;

use ncm_exec::{ExecBackend, ExecSession, PendingOutput, PickSpec, StepOut};
use ncm_ort::{OrtBackend, OrtOptions};
use serde_json::{Value, json};

use crate::{Cli, median};

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

pub fn run(cli: &Cli) -> Result<Value, String> {
    let dir =
        cli.onnx_dir.clone().ok_or("--device ort-cpu needs --onnx-dir <ncm-export directory>")?;
    if cli.repeats == 0 || cli.context.is_empty() {
        return Err("--repeats and --context need at least one value".into());
    }
    let backend = OrtBackend::load(&OrtOptions {
        dir,
        dylib: cli.ort_lib.clone(),
        threads: None,
        allow_version_skew: false,
    })
    .map_err(|e| e.to_string())?;
    let manifest = backend.manifest().clone();
    let mut results = Vec::new();
    for &context in &cli.context {
        if context + cli.new_tokens > manifest.ctx {
            return Err(format!(
                "context {context} plus {} new tokens exceeds the export's context {}",
                cli.new_tokens, manifest.ctx
            ));
        }
        let prompt: Vec<i32> = (0..context as i32).map(|i| 16 + (i * 7) % 200).collect();
        let (mut prefill, mut decode) = (Vec::new(), Vec::new());
        for _ in 0..cli.repeats {
            let mut session = backend.new_session(manifest.ctx).map_err(|e| e.to_string())?;
            let started = Instant::now();
            let out = session.prefill(&prompt, 512).map_err(|e| e.to_string())?;
            debug_assert_eq!(out.output, PendingOutput::Logits);
            prefill.push(context as f64 / started.elapsed().as_secs_f64());
            let mut token = argmax(&session.pending_logits().ok_or("no logits after the prompt")?);
            let started = Instant::now();
            for _ in 0..cli.new_tokens {
                let StepOut::Picked(next) = session
                    .decode_step(&[token as i32], &PickSpec::greedy())
                    .map_err(|e| e.to_string())?
                else {
                    return Err("a greedy step returned logits".into());
                };
                token = next[0] as usize;
            }
            decode.push(cli.new_tokens as f64 / started.elapsed().as_secs_f64());
        }
        let (prefill_tok_s, decode_tok_s) = (median(&mut prefill), median(&mut decode));
        println!(
            "ort-cpu tok ({}) @ ctx {context}: prefill {prefill_tok_s:.0} tok/s, decode \
             {decode_tok_s:.1} tok/s",
            manifest.quant
        );
        results.push(json!({
            "model": "tok",
            "context": context,
            "new_tokens": cli.new_tokens,
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
        }));
    }
    Ok(json!({
        "device": "ort-cpu",
        "backend": backend.describe(),
        "quant": manifest.quant,
        "target": manifest.target,
        "export_ctx": manifest.ctx,
        "repeats": cli.repeats,
        "results": results,
    }))
}
