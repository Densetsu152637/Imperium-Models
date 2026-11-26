use std::process::ExitCode;

use clap::Parser;
use ncm_bench::{Cli, run};

fn main() -> ExitCode {
    // `ncm-bench train-memory [flags]`: the training-memory estimate (no GPU).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("train-memory") {
        let rest = std::iter::once(args[0].clone()).chain(args[2..].iter().cloned());
        let cli = ncm_bench::train_memory::TrainMemoryCli::parse_from(rest);
        let payload = ncm_bench::train_memory::run(&cli);
        if let Some(path) = &cli.json {
            let text = serde_json::to_string_pretty(&payload).expect("serialisable");
            if let Err(e) = std::fs::write(path, text) {
                eprintln!("ncm-bench: cannot write {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
        return ExitCode::SUCCESS;
    }
    let cli = Cli::parse();
    match run(&cli) {
        Ok(payload) => {
            if let Some(path) = &cli.json {
                let text = serde_json::to_string_pretty(&payload).expect("serialisable");
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("ncm-bench: cannot write {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("ncm-bench: {message}");
            ExitCode::FAILURE
        }
    }
}
