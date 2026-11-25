//! Benchmark CLI: help text, thresholds and a tiny CPU run.

use clap::{CommandFactory, Parser};
use ncm_bench::{Cli, fallback_verdict, run, verdict};

#[test]
fn help_lists_the_documented_flags() {
    let help = Cli::command().render_long_help().to_string();
    for flag in [
        "--model",
        "--design",
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
fn fallback_floor() {
    assert_eq!(fallback_verdict(0.75), "PASS");
    assert_eq!(fallback_verdict(0.74), "FAIL");
}

#[test]
fn tiny_cpu_run_reports_both_models_in_both_designs() {
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
    let names: Vec<String> = payload["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| format!("{} {}", r["name"].as_str().unwrap(), r["design"].as_str().unwrap()))
        .collect();
    assert_eq!(names, ["tok fallback", "ncm fallback", "tok v2", "ncm v2"]);
    let ratios = payload["ratios"].as_array().unwrap();
    assert_eq!(ratios.len(), 2);
    assert_eq!(
        (ratios[0]["design"].as_str(), ratios[1]["design"].as_str()),
        (Some("fallback"), Some("v2"))
    );
    assert_eq!(ratios[0]["context"], 64);
    assert_eq!(payload["ncm_v2_over_fallback"].as_array().unwrap().len(), 1);
    for result in payload["results"].as_array().unwrap() {
        let ctx = &result["contexts"]["64"];
        assert!(ctx["decode_tok_s"].as_f64().unwrap() > 0.0);
        assert!(ctx["cache_bytes"].as_u64().unwrap() > 0);
    }
}

#[test]
fn design_flag_picks_one_design() {
    let cli = Cli::parse_from([
        "ncm-bench",
        "--tiny",
        "--device",
        "cpu",
        "--design",
        "v2",
        "--model",
        "ncm",
        "--context",
        "32",
        "--new-tokens",
        "4",
        "--repeats",
        "1",
    ]);
    let payload = run(&cli).expect("tiny run");
    let results = payload["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["design"], "v2");
    assert!(payload["ratios"].as_array().unwrap().is_empty());
}
