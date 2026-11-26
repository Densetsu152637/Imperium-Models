//! `ncm-export --model <weights> --out <dir> --buckets 128,1024 [--quant q8|q4] [--target cpu]`
//!
//! Writes one static-shape ONNX graph per phase, the shared external weights file and
//! `manifest.json`. `--describe` prints the phase graphs of the model and which of them this
//! build can write, without reading any tensor.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use ncm_export::weights::Weights;
use ncm_export::{ExportOptions, Support, Target, WeightForm, describe, export};
use ncm_model::weights::read_config;

#[derive(Parser, Debug)]
#[command(version, about = "Exports an NCM weight file to per-phase static-shape ONNX graphs")]
struct Cli {
    /// The dense weight file (safetensors with the config in its header).
    #[arg(long)]
    model: PathBuf,
    /// The export directory (created).
    #[arg(long, required_unless_present_any = ["describe", "list_tensors"])]
    out: Option<PathBuf>,
    /// Prefill bucket lengths.
    #[arg(long, value_delimiter = ',', default_value = "128,1024")]
    buckets: Vec<usize>,
    /// Context tokens the attention rings are sized for.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Weight quantisation: none, q8 (per channel) or q4 (block-wise, symmetric).
    #[arg(long, default_value = "none")]
    quant: String,
    /// The q4 group size (a multiple of 32).
    #[arg(long, default_value_t = 64)]
    group: usize,
    /// cpu, intel, qualcomm or amd (the NPU profiles wait for their smoke tests).
    #[arg(long, default_value = "cpu")]
    target: String,
    /// Print the tensors (name, dtype, shape) of the weight file's header and exit.
    #[arg(long)]
    list_tensors: bool,
    /// Print the phase graphs and their status, write nothing.
    #[arg(long)]
    describe: bool,
}

fn run(cli: Cli) -> Result<(), String> {
    if cli.list_tensors {
        for (name, dtype, shape) in
            ncm_export::weights::list_tensors(&cli.model).map_err(|e| e.to_string())?
        {
            println!("{name}	{dtype}	{shape:?}");
        }
        return Ok(());
    }
    let config = read_config(&cli.model)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("{} has no model config in its header", cli.model.display()))?;
    if cli.describe {
        for plan in describe(&config, &cli.buckets) {
            let status = match &plan.support {
                Support::Supported => "supported".to_owned(),
                Support::Unsupported(why) => format!("not written yet: {why}"),
            };
            println!("{:<16} {:>6} tokens  {status}", plan.name, plan.tokens);
        }
        return Ok(());
    }
    let form = WeightForm::parse(&cli.quant, cli.group)?;
    let target = Target::parse(&cli.target)?;
    let weights = Weights::open(&cli.model).map_err(|e| e.to_string())?;
    let out = cli.out.expect("clap requires --out without --describe");
    let opts = ExportOptions { ctx: cli.ctx, buckets: cli.buckets, form, target };
    let manifest = export(&config, &weights, &out, &opts).map_err(|e| e.to_string())?;
    eprintln!(
        "ncm-export: wrote {} graphs ({}) to {}",
        manifest.graphs.len(),
        manifest.quant,
        out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ncm-export: {message}");
            ExitCode::FAILURE
        }
    }
}
