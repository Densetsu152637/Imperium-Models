//! The real server, in process: command line to model to HTTP answers, with random and saved
//! weights. CPU backend, tiny models.

mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use burn::tensor::Device;
use clap::Parser;
use common::{tiny_ncm_config, tiny_tok_config};
use ncm_model::weights::save_weights;
use ncm_model::{Config, NcmLm, TokenLm};
use ncm_server::app::{Running, start};
use ncm_server::cli::{Cli, normalize_args};
use ncm_server::testing::byte_tokenizer_json;
use serde_json::{Value, json};

/// Model construction seeds a process-wide generator, so tests that build models take turns.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn cli(args: &[&str]) -> Cli {
    let args = ["ncm-server"].into_iter().chain(args.iter().copied()).map(str::to_owned);
    Cli::try_parse_from(normalize_args(args)).unwrap()
}

async fn wait_ready(running: &Running) {
    let url = format!("http://{}/health", running.addr);
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if reqwest::get(&url).await.is_ok_and(|r| r.status() == 200) {
            return;
        }
        assert!(running.state.failure().is_none(), "{:?}", running.state.failure());
        assert!(Instant::now() < deadline, "the model did not load");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn chat(running: &Running, body: Value) -> Value {
    reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", running.addr))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

fn greedy(content: &str) -> Value {
    json!({"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 12})
}

fn write_tiny(dir: &Path, config: &Config) -> (String, String) {
    let config_path = dir.join("config.toml");
    std::fs::write(&config_path, config.to_toml()).unwrap();
    let tokenizer = dir.join("tokenizer.json");
    std::fs::write(&tokenizer, byte_tokenizer_json().to_string()).unwrap();
    (config_path.display().to_string(), tokenizer.display().to_string())
}

/// The device the tests serve on: `NCM_E2E_DEVICE` (default: the Burn CPU backend). `ort-cpu`
/// needs `--features ort` and the ONNX Runtime library (`ORT_DYLIB_PATH`); it serves the token
/// baseline from an `ncm-export` of the same weights.
fn on_ort() -> bool {
    match std::env::var("NCM_E2E_DEVICE").as_deref() {
        Ok("ort-cpu") => true,
        Ok("") | Ok("cpu") | Err(_) => false,
        Ok(other) => panic!("NCM_E2E_DEVICE={other}: expected cpu or ort-cpu"),
    }
}

/// Saves `config`'s model (weights seeded `seed`, as the server seeds before it builds) in
/// `dir` and exports it for ONNX Runtime into `dir/onnx`: the model directory to serve.
fn export_tiny(dir: &Path, config: &Config, seed: u64) -> String {
    let device = Device::flex();
    device.seed(seed);
    let weights = dir.join("export-source.safetensors");
    match config {
        Config::Tok(c) => save_weights(&TokenLm::new(c, &device), config, &weights).unwrap(),
        Config::Ncm(_) => unreachable!("the k = 4 NCM has no ONNX path"),
    }
    let out = dir.join("onnx");
    let opts = ncm_export::ExportOptions {
        ctx: 512,
        buckets: vec![8],
        form: ncm_export::WeightForm::F32,
        target: ncm_export::Target::Cpu,
    };
    let source = ncm_export::weights::Weights::open(&weights).unwrap();
    ncm_export::export(config, &source, &out, &opts).unwrap();
    out.display().to_string()
}

#[tokio::test]
async fn random_weights_serve_a_chat_completion_for_both_architectures() {
    let _serial = SERIAL.lock().await;
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        if on_ort() && matches!(config, Config::Ncm(_)) {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let (config_path, tokenizer) = write_tiny(dir.path(), &config);
        // ONNX Runtime serves one request at a time; the Burn backends two slots.
        let (slots, n_ctx) = if on_ort() { (1, 512) } else { (2, 256) };
        let slots_arg = slots.to_string();
        let args: Vec<String> = if on_ort() {
            let model = export_tiny(dir.path(), &config, 0);
            ["--device", "ort-cpu", "--model", &model, "--tokenizer", &tokenizer]
                .map(String::from)
                .to_vec()
        } else {
            ["--random-weights", "--config", &config_path].map(String::from).to_vec()
        };
        let mut all: Vec<&str> = args.iter().map(String::as_str).collect();
        all.extend(["--port", "0", "-c", "512", "--parallel", &slots_arg]);
        let running = start(cli(&all)).await.unwrap();
        // Listening at once; ready once the model has loaded.
        wait_ready(&running).await;
        let base = format!("http://{}", running.addr);

        let props: Value =
            reqwest::get(format!("{base}/props")).await.unwrap().json().await.unwrap();
        assert_eq!(props["default_generation_settings"]["n_ctx"], n_ctx, "512 split over slots");
        assert_eq!(props["total_slots"], slots);

        let body = chat(&running, greedy("hello")).await;
        assert_eq!(body["object"], "chat.completion");
        assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);
        assert!(body["timings"]["prompt_per_second"].as_f64().unwrap() > 0.0);
        let again = chat(&running, greedy("hello")).await;
        assert_eq!(body["choices"], again["choices"], "greedy replies repeat");
        running.stop().await.unwrap();
    }
}

#[tokio::test]
async fn saved_weights_load_and_give_the_same_replies_as_the_random_weights_they_came_from() {
    let _serial = SERIAL.lock().await;
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        if on_ort() && matches!(config, Config::Ncm(_)) {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let (config_path, tokenizer) = write_tiny(dir.path(), &config);
        let weights = dir.path().join("tiny.safetensors");
        // The server seeds the device and then builds the model; do the same to save it.
        let device = Device::flex();
        device.seed(5);
        match &config {
            Config::Tok(c) => save_weights(&TokenLm::new(c, &device), &config, &weights).unwrap(),
            Config::Ncm(c) => save_weights(&NcmLm::new(c, &device), &config, &weights).unwrap(),
        }
        let weights = weights.display().to_string();

        // The config comes from the weight file's header: no --config, no --arch. On ONNX
        // Runtime the loaded model is an export of the same weights, checked against the Burn
        // random-weights server (the reference).
        let loaded = if on_ort() {
            let model = export_tiny(dir.path(), &config, 5);
            start(cli(&[
                "--device",
                "ort-cpu",
                "--model",
                &model,
                "--tokenizer",
                &tokenizer,
                "--alias",
                "tiny",
                "--port",
                "0",
                "-c",
                "256",
            ]))
            .await
            .unwrap()
        } else {
            start(cli(&[
                "--model",
                &weights,
                "--tokenizer",
                &tokenizer,
                "--port",
                "0",
                "-c",
                "256",
            ]))
            .await
            .unwrap()
        };
        wait_ready(&loaded).await;
        // Built one after the other: model construction seeds a process-wide generator.
        let random = start(cli(&[
            "--random-weights",
            "--weights-seed",
            "5",
            "--config",
            &config_path,
            "--port",
            "0",
            "-c",
            "256",
        ]))
        .await
        .unwrap();
        wait_ready(&random).await;
        let b = chat(&loaded, greedy("compare these")).await;
        let a = chat(&random, greedy("compare these")).await;
        assert_eq!(a["choices"][0]["message"], b["choices"][0]["message"]);
        assert_eq!(b["model"], "tiny", "the alias defaults to the weight file's name");
        random.stop().await.unwrap();
        loaded.stop().await.unwrap();
    }
}

#[tokio::test]
async fn a_bad_weight_file_fails_the_launch() {
    let _serial = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let (config_path, tokenizer) = write_tiny(dir.path(), &tiny_tok_config());
    if on_ort() {
        // A directory whose manifest is not one: refused before anything starts.
        let model = dir.path().join("onnx");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("manifest.json"), b"not a manifest").unwrap();
        let error = start(cli(&[
            "--device",
            "ort-cpu",
            "--model",
            &model.display().to_string(),
            "--tokenizer",
            &tokenizer,
            "--port",
            "0",
        ]))
        .await
        .err()
        .expect("a refused launch");
        assert!(error.contains("manifest.json"), "{error}");
        return;
    }
    let weights = dir.path().join("bad.safetensors");
    std::fs::write(&weights, b"not a safetensors file at all").unwrap();
    let running = start(cli(&[
        "--model",
        &weights.display().to_string(),
        "--tokenizer",
        &tokenizer,
        "--config",
        &config_path,
        "--port",
        "0",
    ]))
    .await
    .unwrap();
    // The server answers 503 while loading and then exits with the reason, so a supervisor that
    // polls /health sees the process die instead of waiting forever.
    let error = running.wait().await.unwrap_err();
    assert!(error.contains("bad.safetensors"), "{error}");
}

#[tokio::test]
async fn launch_mistakes_are_explained_before_anything_starts() {
    let message = |args: &[&str]| {
        let cli = cli(args);
        async move { start(cli).await.err().expect("a refused launch") }
    };
    assert!(message(&["--random-weights"]).await.contains("--arch"));
    assert!(message(&["--arch", "tok"]).await.contains("--model is required"));
    assert!(
        message(&["--arch", "tok", "--model", "/nope/m.safetensors"]).await.contains("not found")
    );
    let dir = tempfile::tempdir().unwrap();
    let (config_path, _) = write_tiny(dir.path(), &tiny_tok_config());
    assert!(
        message(&["--random-weights", "--config", &config_path, "--arch", "ncm"])
            .await
            .contains("--arch")
    );
}

#[test]
fn imperiums_llama_server_launch_line_parses() {
    // What imperium-local's LlamaCpp backend passes, then the port ProcessLauncher adds.
    let parsed = cli(&[
        "--model",
        "m.safetensors",
        "--host",
        "127.0.0.1",
        "--ctx-size",
        "8192",
        "--parallel",
        "2",
        "--n-gpu-layers",
        "999",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--jinja",
        "--port",
        "40123",
    ]);
    assert_eq!(parsed.ctx_size, 8192);
    assert_eq!(parsed.parallel, 2);
    assert_eq!(parsed.port, 40123);
    assert_eq!(parsed.host, "127.0.0.1");
    assert!(parsed.jinja);
    assert!(parsed.warnings().iter().all(|w| !w.contains("cache-type")));
    // llama.cpp's single-dash long flags (as users write them in extra_args).
    let short = cli(&[
        "-m",
        "m.safetensors",
        "-c",
        "1000000",
        "-ngl",
        "0",
        "-np",
        "3",
        "-fa",
        "-ctk",
        "q8_0",
    ]);
    assert_eq!(short.ctx_size, 1_000_000);
    assert_eq!(short.n_gpu_layers.as_deref(), Some("0"));
    assert_eq!(short.parallel, 3);
    assert!(short.warnings().iter().any(|w| w.contains("--cache-type-k q8_0")));
    // --parallel is real batching now: no note.
    assert!(!short.warnings().iter().any(|w| w.contains("--parallel")));
}
