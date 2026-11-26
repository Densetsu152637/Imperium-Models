//! Starting the server: resolving the model, tokenizer and device from the command line,
//! listening at once, and loading the model in the background so `/health` can answer 503 while
//! it loads (Imperium polls it).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::tensor::FloatDType;
use ncm_model::weights;
use ncm_model::{Config, ncm_250m, tok_250m};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::cli::{ArchArg, Cli, DeviceArg, DtypeArg};
#[cfg(feature = "ort")]
use crate::engine::OrtEngineOptions;
use crate::engine::{DeviceChoice, Engine, EngineOptions, make_device};
use crate::sampling::SamplingParams;
use crate::server::{AppState, serve};
use crate::testing::byte_tokenizer;
use crate::vocab::Vocab;

/// Everything `load_engine` needs, resolved from the command line.
#[derive(Debug)]
pub struct Plan {
    pub config: Config,
    pub weights: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub device: DeviceChoice,
    pub dtype: FloatDType,
    /// Context tokens per request.
    pub ctx: usize,
    pub alias: String,
}

fn arch_of(config: &Config) -> ArchArg {
    match config {
        Config::Tok(_) => ArchArg::Tok,
        Config::Ncm(_) => ArchArg::Ncm,
    }
}

fn check_arch(config: Config, wanted: Option<ArchArg>, source: &str) -> Result<Config, String> {
    match wanted {
        Some(arch) if arch != arch_of(&config) => {
            Err(format!("{source} is a {:?} model but --arch is {arch:?}", arch_of(&config)))
        }
        _ => Ok(config),
    }
}

/// The model config: `--config`, else the weight file's header, else `config.toml` beside the
/// weights, else the 250M config of `--arch`.
fn resolve_config(cli: &Cli) -> Result<Config, String> {
    if let Some(path) = &cli.config {
        let config = Config::load(path).map_err(|e| e.to_string())?;
        return check_arch(config, cli.arch, &path.display().to_string());
    }
    if let Some(model) = cli.model.as_deref().filter(|_| !cli.random_weights) {
        if let Some(config) = weights::read_config(model).map_err(|e| e.to_string())? {
            return check_arch(config, cli.arch, &model.display().to_string());
        }
        let beside = model.with_file_name("config.toml");
        if beside.is_file() {
            let config = Config::load(&beside).map_err(|e| e.to_string())?;
            return check_arch(config, cli.arch, &beside.display().to_string());
        }
    }
    match cli.arch {
        Some(ArchArg::Tok) => Ok(Config::Tok(tok_250m())),
        Some(ArchArg::Ncm) => Ok(Config::Ncm(ncm_250m())),
        None => Err("cannot tell the architecture: pass --arch tok|ncm or --config, or a weight \
                     file with a config in its header"
            .into()),
    }
}

fn first_existing(paths: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    paths.into_iter().find(|p| p.is_file())
}

/// The plan of `--device ort-cpu`: `--model` names an `ncm-export` directory, whose manifest
/// carries the config.
fn plan_ort(cli: &Cli) -> Result<Plan, String> {
    if !cfg!(feature = "ort") {
        return Err("this build has no ONNX Runtime support; rebuild with `--features ort`".into());
    }
    let dir = cli
        .model
        .clone()
        .ok_or_else(|| "--device ort-cpu needs --model <ncm-export directory>".to_owned())?;
    let manifest_path = dir.join(ncm_exec::manifest::FILE_NAME);
    if !manifest_path.is_file() {
        return Err(format!(
            "{} has no {} (--device ort-cpu takes an `ncm-export` directory)",
            dir.display(),
            ncm_exec::manifest::FILE_NAME
        ));
    }
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest = ncm_exec::manifest::Manifest::from_json(&text)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let config = Config::parse(&manifest.config_toml)
        .map_err(|e| format!("{}: its config: {e}", manifest_path.display()))?;
    let config = check_arch(config, cli.arch, &manifest_path.display().to_string())?;
    if cli.parallel != 1 {
        return Err("--device ort-cpu serves one request at a time: --parallel must be 1".into());
    }
    let tokenizer = cli.tokenizer.clone().or_else(|| first_existing([dir.join("tokenizer.json")]));
    if tokenizer.is_none() {
        return Err(
            "no tokenizer: pass --tokenizer or put tokenizer.json in the export directory".into()
        );
    }
    let alias = cli.alias.clone().unwrap_or_else(|| {
        dir.file_name().map_or_else(|| "ncm-onnx".to_owned(), |s| s.to_string_lossy().into_owned())
    });
    Ok(Plan {
        config,
        weights: Some(dir),
        tokenizer,
        device: DeviceChoice::OrtCpu,
        dtype: FloatDType::F32,
        ctx: cli.ctx_size,
        alias,
    })
}

/// Turns the command line into a plan, checking the files it names.
pub fn plan(cli: &Cli) -> Result<Plan, String> {
    if cli.device == Some(DeviceArg::OrtCpu) {
        return plan_ort(cli);
    }
    if let Some(path) = cli.model.as_ref().filter(|p| !cli.random_weights && !p.is_file()) {
        return Err(format!("model file {} not found", path.display()));
    }
    let config = resolve_config(cli)?;
    let weights_path = match (&cli.model, cli.random_weights) {
        (_, true) => None,
        (Some(path), false) => Some(path.clone()),
        (None, false) => return Err("--model is required (or --random-weights)".into()),
    };
    let tokenizer = match (&cli.tokenizer, &weights_path) {
        (Some(path), _) => Some(path.clone()),
        (None, Some(model)) => first_existing([model.with_file_name("tokenizer.json")]),
        (None, None) => None,
    };
    if tokenizer.is_none() && weights_path.is_some() {
        return Err("no tokenizer: pass --tokenizer or put tokenizer.json beside the model".into());
    }
    if cli.parallel == 0 {
        return Err("--parallel must be at least 1".into());
    }
    let ctx = cli.ctx_size / cli.parallel;
    if ctx == 0 {
        return Err("--ctx-size is smaller than --parallel".into());
    }
    let wants_gpu = cli.n_gpu_layers.as_deref().is_none_or(|n| n.trim() != "0");
    let device = match cli.device {
        Some(DeviceArg::Cpu) => DeviceChoice::Cpu,
        Some(DeviceArg::Cuda) => DeviceChoice::Cuda,
        Some(DeviceArg::OrtCpu) => unreachable!("planned by plan_ort"),
        None if cfg!(feature = "cuda") && wants_gpu => DeviceChoice::Cuda,
        None => DeviceChoice::Cpu,
    };
    let dtype = match (cli.dtype, device) {
        (Some(DtypeArg::Bf16), _) | (None, DeviceChoice::Cuda) => FloatDType::BF16,
        _ => FloatDType::F32,
    };
    let alias = cli.alias.clone().unwrap_or_else(|| {
        weights_path
            .as_deref()
            .and_then(Path::file_stem)
            .map_or_else(|| "ncm-random".to_owned(), |s| s.to_string_lossy().into_owned())
    });
    Ok(Plan { config, weights: weights_path, tokenizer, device, dtype, ctx, alias })
}

/// Loads the tokenizer, builds the model and allocates its caches. Slow for big models.
pub fn load_engine(plan: &Plan, cli: &Cli) -> Result<Engine, String> {
    let vocab_size = plan.config.vocab_size();
    let vocab = match &plan.tokenizer {
        Some(path) => Vocab::from_file(path, vocab_size),
        None => Vocab::new(byte_tokenizer(), vocab_size),
    }
    .map_err(|e| e.to_string())?;
    if plan.device == DeviceChoice::OrtCpu {
        return load_ort(plan, cli, vocab);
    }
    // Before the device is first used: CubeCL reads its configuration once per process.
    #[cfg(feature = "cuda")]
    if plan.device == DeviceChoice::Cuda && !cli.no_kernel_cache {
        match ncm_model::kernels::enable_kernel_cache(cli.kernel_cache_dir.clone()) {
            Some(dir) => eprintln!("ncm-server: kernel cache in {}", dir.display()),
            None => eprintln!("ncm-server: kernel cache not enabled (no cache directory)"),
        }
    }
    let device = make_device(plan.device, plan.dtype).map_err(|e| e.to_string())?;
    let options = EngineOptions {
        config: plan.config.clone(),
        ctx: plan.ctx,
        device,
        device_name: format!("{:?}", plan.device).to_lowercase(),
        dtype: plan.dtype,
        weights: plan.weights.clone(),
        seed: cli.weights_seed,
        prefill_chunk: cli.prefill_chunk.unwrap_or(match plan.config.design() {
            ncm_model::Design::V2 => crate::engine::DEFAULT_PREFILL_CHUNK_V2,
            ncm_model::Design::Fallback => crate::engine::DEFAULT_PREFILL_CHUNK,
        }),
        quant: cli.quant_spec()?,
        speculate: cli.speculate,
        prefix_reuse: !cli.no_prefix_reuse,
        warmup: !cli.no_warmup,
        slots: cli.parallel,
    };
    Engine::new(options, Arc::new(vocab)).map_err(|e| e.to_string())
}

/// The engine over ONNX Runtime's CPU provider.
#[cfg(feature = "ort")]
fn load_ort(plan: &Plan, cli: &Cli, vocab: Vocab) -> Result<Engine, String> {
    let dir = plan.weights.clone().expect("an ort plan names its export directory");
    let options = OrtEngineOptions {
        config: plan.config.clone(),
        dir,
        ctx: plan.ctx,
        dylib: cli.ort_lib.clone(),
        threads: cli.threads,
        allow_version_skew: cli.ort_allow_version_skew,
        prefill_chunk: cli.prefill_chunk.unwrap_or(crate::engine::DEFAULT_PREFILL_CHUNK),
        prefix_reuse: !cli.no_prefix_reuse,
        warmup: !cli.no_warmup,
    };
    Engine::new_ort(options, Arc::new(vocab)).map_err(|e| e.to_string())
}

#[cfg(not(feature = "ort"))]
fn load_ort(_plan: &Plan, _cli: &Cli, _vocab: Vocab) -> Result<Engine, String> {
    Err("this build has no ONNX Runtime support; rebuild with `--features ort`".into())
}

/// A server that is listening; the model loads in the background.
pub struct Running {
    pub addr: SocketAddr,
    pub state: Arc<AppState>,
    stop: Arc<Notify>,
    server: JoinHandle<std::io::Result<()>>,
    loader: JoinHandle<()>,
}

impl Running {
    /// Asks the server to stop.
    pub fn stop_signal(&self) -> Arc<Notify> {
        self.stop.clone()
    }

    /// Waits until the server stops: a stop signal, or a model that failed to load (an error).
    pub async fn wait(self) -> Result<(), String> {
        let served = self.server.await.map_err(|e| e.to_string())?;
        self.loader.abort();
        served.map_err(|e| e.to_string())?;
        match self.state.failure() {
            Some(message) => Err(message),
            None => Ok(()),
        }
    }

    pub async fn stop(self) -> Result<(), String> {
        self.stop.notify_one();
        self.wait().await
    }
}

/// Binds the listener at once and loads the model in the background.
pub async fn start(cli: Cli) -> Result<Running, String> {
    let plan = plan(&cli)?;
    for note in cli.warnings() {
        eprintln!("ncm-server: {note}");
    }
    let listener = tokio::net::TcpListener::bind((cli.host.as_str(), cli.port))
        .await
        .map_err(|e| format!("cannot listen on {}:{}: {e}", cli.host, cli.port))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let defaults =
        SamplingParams { temperature: cli.temp, top_k: cli.top_k, top_p: cli.top_p, seed: None };
    let model_path = plan.weights.as_ref().map_or_else(String::new, |p| p.display().to_string());
    let state = Arc::new(AppState::loading(&plan.alias, &model_path, cli.parallel, defaults));
    let stop = Arc::new(Notify::new());
    eprintln!("ncm-server: listening on http://{addr}, loading the model");

    let shutdown = {
        let stop = stop.clone();
        async move { stop.notified().await }
    };
    let server = tokio::spawn(serve(listener, state.clone(), shutdown));

    let loader = {
        let (state, stop) = (state.clone(), stop.clone());
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let loaded = tokio::task::spawn_blocking(move || load_engine(&plan, &cli)).await;
            match loaded {
                Ok(Ok(engine)) => {
                    let info = crate::engine::Backend::info(&engine).clone();
                    eprintln!(
                        "ncm-server: ready in {:.1}s: {} model, {:.1}M parameters, {} on {}, \
                         context {} tokens, cache {:.1} MiB",
                        started.elapsed().as_secs_f64(),
                        info.arch,
                        info.params as f64 / 1e6,
                        info.dtype,
                        info.device,
                        info.ctx,
                        info.cache_bytes as f64 / (1u64 << 20) as f64,
                    );
                    state.set_backend(Arc::new(engine));
                }
                Ok(Err(message)) => {
                    eprintln!("ncm-server: cannot load the model: {message}");
                    state.set_failure(message);
                    stop.notify_one();
                }
                Err(join) => {
                    let message = format!("loading the model panicked: {join}");
                    eprintln!("ncm-server: {message}");
                    state.set_failure(message);
                    stop.notify_one();
                }
            }
        })
    };
    Ok(Running { addr, state, stop, server, loader })
}
