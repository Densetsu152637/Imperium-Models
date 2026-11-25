use std::process::ExitCode;

use clap::Parser;
use ncm_bench::{Cli, run};

fn main() -> ExitCode {
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
