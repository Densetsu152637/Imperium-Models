//! Command line: the llama-server flags Imperium passes, plus `--arch`, `--random-weights` and
//! a few NCM options.
//!
//! Imperium starts `llama-server` as
//! `--model <path> --host 127.0.0.1 --ctx-size <n> --parallel <slots> --n-gpu-layers <n>
//! --cache-type-k <t> --cache-type-v <t> --jinja [--mmproj <path>] <extra args> --port <port>`.
//! `ncm-server` takes all of those. Flags that only make sense for llama.cpp (cache types, thread
//! counts, `--jinja`) are accepted and ignored, with a note on stderr when they ask for
//! something this runtime does not do.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use ncm_model::quant::QuantSpec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ArchArg {
    /// The token baseline.
    Tok,
    /// The Next Concept Model.
    Ncm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceArg {
    Cpu,
    /// Needs a build with `--features cuda`.
    Cuda,
    /// ONNX Runtime's CPU execution provider over an `ncm-export` directory (`--model <dir>`);
    /// needs a build with `--features ort` and the ONNX Runtime shared library
    /// (`--ort-lib` or `ORT_DYLIB_PATH`). The token baseline only.
    #[value(name = "ort-cpu")]
    OrtCpu,
}

/// Weight-only quantisation of the linear layers (see `ncm_model::quant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum QuantArg {
    /// Dense weights at the run dtype.
    None,
    /// 8-bit codes, one scale per output channel (W8A16).
    Q8,
    /// 4-bit codes, one scale per `--quant-group` inputs (W4A16).
    Q4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    Bf16,
    #[value(alias = "fp32")]
    F32,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ncm-server",
    version,
    about = "OpenAI-compatible server for the NCM and the token baseline, started like llama-server.",
    after_help = "Weights: a safetensors file written by ncm_model::weights (the config is in its header)."
)]
pub struct Cli {
    /// Weight file (safetensors, as `ncm_model::weights::save_weights` writes).
    #[arg(short = 'm', long, value_name = "PATH")]
    pub model: Option<PathBuf>,
    /// Hugging Face tokenizer.json. Defaults to `tokenizer.json` next to the model; with
    /// `--random-weights` a built-in byte-level tokenizer is used when none is given.
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,
    /// Model config TOML. Defaults to the one in the weight file's header, then `config.toml`
    /// beside it, then the 250M config of `--arch`.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Architecture; read from the config when there is one.
    #[arg(long, value_enum)]
    pub arch: Option<ArchArg>,
    /// Serve randomly initialised weights (for tests and speed runs); `--model` is not needed.
    #[arg(long)]
    pub random_weights: bool,
    /// Seed for `--random-weights`.
    #[arg(long, default_value_t = 0)]
    pub weights_seed: u64,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to listen on; 0 picks a free one.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Context tokens in total, allocated at launch (the key/value cache is sized for it).
    /// With `--parallel N`, each request gets `ctx-size / N`.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,
    /// Slots: up to this many requests decode together (continuous batching), each with
    /// `ctx-size / N` context tokens and caches of its own, as in llama-server. 1 serves
    /// requests one at a time.
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    /// `cuda` when built with it and `--n-gpu-layers` is not 0, else `cpu`; `ort-cpu` runs an
    /// ONNX export on ONNX Runtime (`-ngl` is ignored there).
    #[arg(long, value_enum)]
    pub device: Option<DeviceArg>,
    /// The ONNX Runtime shared library for `--device ort-cpu` (default: `ORT_DYLIB_PATH`).
    #[arg(long, value_name = "PATH")]
    pub ort_lib: Option<PathBuf>,
    /// Accept an ONNX Runtime whose version differs from the one this build is pinned to.
    #[arg(long)]
    pub ort_allow_version_skew: bool,
    /// `bf16` on CUDA, `f32` on CPU.
    #[arg(long, value_enum)]
    pub dtype: Option<DtypeArg>,
    /// Model name in responses; defaults to the weight file's name.
    #[arg(short = 'a', long)]
    pub alias: Option<String>,
    /// Default temperature (0 is greedy); a request may override it.
    #[arg(long, default_value_t = 0.8)]
    pub temp: f32,
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Prompt tokens fed to the model per step (default 8192 for the v2 design, 512 for the
    /// fallback).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    /// Decode greedy, unconstrained NCM requests by self-speculation (draft and verify passes)
    /// when the model has a draft head; CUDA only. Off by default: in bf16 the parallel verify
    /// pass may resolve a near tie differently from step-by-step greedy decoding.
    #[arg(long)]
    pub speculate: bool,
    /// Prefill every prompt from scratch instead of reusing the previous request's cache when
    /// the prompt extends the previous prompt.
    #[arg(long)]
    pub no_prefix_reuse: bool,
    /// Skip the warm-up requests at launch (the first request then compiles the kernels and
    /// captures the decode graphs).
    #[arg(long)]
    pub no_warmup: bool,
    /// Directory of the on-disk CUDA kernel cache (default: the user cache directory's
    /// `ncm/kernels`): a restarted server skips compiling the kernels a previous run compiled.
    #[arg(long, value_name = "DIR")]
    pub kernel_cache_dir: Option<PathBuf>,
    /// Compile every kernel anew instead of keeping them on disk.
    #[arg(long)]
    pub no_kernel_cache: bool,
    /// Quantise the linear layers' weights at load. A quantised weight file loads as it is.
    #[arg(long, value_enum, default_value = "none")]
    pub quant: QuantArg,
    /// Inputs per scale for `--quant q4` (a multiple of 32).
    #[arg(long, default_value_t = 64)]
    pub quant_group: usize,
    /// Keep the tied output head dense when quantising.
    #[arg(long)]
    pub dense_head: bool,

    // llama-server flags: accepted so Imperium's launch line works unchanged.
    /// Layers on the GPU; 0 means CPU. Anything else uses the GPU when the build has one.
    #[arg(long = "n-gpu-layers", alias = "gpu-layers", value_name = "N")]
    pub n_gpu_layers: Option<String>,
    /// Key/value cache dtype; this runtime keeps the model's dtype (quantised caches are TODO).
    #[arg(long = "cache-type-k", value_name = "TYPE")]
    pub cache_type_k: Option<String>,
    #[arg(long = "cache-type-v", value_name = "TYPE")]
    pub cache_type_v: Option<String>,
    /// Ignored: the chat format is fixed.
    #[arg(long)]
    pub jinja: bool,
    #[arg(long, value_name = "PATH")]
    pub mmproj: Option<PathBuf>,
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    #[arg(long = "threads-batch")]
    pub threads_batch: Option<usize>,
    #[arg(short = 'b', long = "batch-size")]
    pub batch_size: Option<usize>,
    #[arg(long = "ubatch-size")]
    pub ubatch_size: Option<usize>,
    #[arg(long = "flash-attn", num_args = 0..=1, default_missing_value = "on")]
    pub flash_attn: Option<String>,
    #[arg(long = "no-mmap")]
    pub no_mmap: bool,
    #[arg(long)]
    pub mlock: bool,
    #[arg(long = "no-webui")]
    pub no_webui: bool,
    #[arg(long = "cont-batching")]
    pub cont_batching: bool,
    #[arg(long = "no-context-shift")]
    pub no_context_shift: bool,
    #[arg(long)]
    pub metrics: bool,
}

/// llama.cpp's single-dash long flags, which clap cannot express as short options.
pub fn normalize_args<I: IntoIterator<Item = String>>(args: I) -> Vec<String> {
    args.into_iter()
        .map(|arg| match arg.as_str() {
            "-ngl" => "--n-gpu-layers".into(),
            "-ctk" => "--cache-type-k".into(),
            "-ctv" => "--cache-type-v".into(),
            "-np" => "--parallel".into(),
            "-fa" => "--flash-attn".into(),
            "-ub" => "--ubatch-size".into(),
            "-tb" => "--threads-batch".into(),
            "-cb" => "--cont-batching".into(),
            _ => arg,
        })
        .collect()
}

impl Cli {
    /// The quantisation `--quant`, `--quant-group` and `--dense-head` ask for.
    pub fn quant_spec(&self) -> Result<Option<QuantSpec>, String> {
        let mode = match self.quant {
            QuantArg::None => "none",
            QuantArg::Q8 => "q8",
            QuantArg::Q4 => "q4",
        };
        QuantSpec::parse(mode, self.quant_group, !self.dense_head)
    }

    pub fn parse_normalized() -> Self {
        Self::parse_from(normalize_args(std::env::args()))
    }

    /// Notes on flags that ask for something this runtime does not do.
    pub fn warnings(&self) -> Vec<String> {
        let mut notes = Vec::new();
        for (name, value) in
            [("--cache-type-k", &self.cache_type_k), ("--cache-type-v", &self.cache_type_v)]
        {
            if let Some(kind) = value.as_deref()
                && !matches!(kind, "f16" | "bf16" | "f32")
            {
                notes.push(format!(
                    "{name} {kind}: quantised caches are not implemented; the cache keeps the model dtype"
                ));
            }
        }
        if self.device == Some(DeviceArg::OrtCpu) {
            if self.n_gpu_layers.is_some() {
                notes.push("--n-gpu-layers ignored: --device ort-cpu runs on the CPU".into());
            }
            if self.quant != QuantArg::None {
                notes.push(
                    "--quant ignored: an ONNX export carries the quantisation it was exported with"
                        .into(),
                );
            }
        }
        if self.mmproj.is_some() {
            notes.push("--mmproj ignored: this runtime reads text only".into());
        }
        notes
    }
}
