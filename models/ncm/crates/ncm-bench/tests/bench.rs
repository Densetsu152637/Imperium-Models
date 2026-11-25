//! Benchmark CLI: help text, thresholds and a tiny CPU run.

use clap::{CommandFactory, Parser};
use ncm_bench::{Cli, run, verdict};

#[test]
fn help_lists_the_documented_flags() {
    let help = Cli::command().render_long_help().to_string();
    for flag in [
        "--model",
        "--context",
        "--new-tokens",
        "--device",
        "--dtype",
        "--repeats",
        "--json",
        "--tiny",
    ] {
        assert!(help.contains(flag), "missing {flag}");
    }
}

#[test]
fn verdict_thresholds() {
    assert_eq!(verdict(1.5), "PASS");
    assert_eq!(verdict(1.3), "WEAK");
    assert_eq!(verdict(1.19), "FAIL");
}

#[test]
fn cuda_is_refused_without_the_feature() {
    if cfg!(feature = "cuda") {
        return;
    }
    let cli = Cli::parse_from(["ncm-bench", "--device", "cuda", "--tiny"]);
    assert!(run(&cli).unwrap_err().contains("--features cuda"));
}

#[test]
fn tiny_cpu_run_reports_both_models() {
    let cli = Cli::parse_from([
        "ncm-bench",
        "--tiny",
        "--device",
        "cpu",
        "--dtype",
        "f32",
        "--context",
        "64",
        "--new-tokens",
        "8",
        "--repeats",
        "1",
    ]);
    let payload = run(&cli).expect("tiny run");
    let names: Vec<&str> = payload["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["tok", "ncm"]);
    assert_eq!(payload["ratios"].as_array().unwrap().len(), 1);
    assert_eq!(payload["ratios"][0]["context"], 64);
    let tok = &payload["results"][0]["contexts"]["64"];
    assert!(tok["decode_tok_s"].as_f64().unwrap() > 0.0);
    assert!(tok["cache_bytes"].as_u64().unwrap() > 0);
}
