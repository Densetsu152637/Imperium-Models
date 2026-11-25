use clap::{Parser, Subcommand, ValueEnum};
use ncm_sft::pipeline::{Model, build, policy_for};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "ncm-sft", about = "Token-era teacher data to NCM fine-tuning datasets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Which {
    Emperor,
    Consul,
    All,
}

#[derive(Subcommand)]
enum Command {
    /// Convert the accepted token-era records of a model.
    Build {
        /// The token-era archive directory (holds `emperor/data` and `consul/data`).
        #[arg(long, default_value = "../../archive/token-era")]
        archive: PathBuf,
        /// Output directory; each model gets `<out>/<model>/`.
        #[arg(long, default_value = "data/sft")]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "all")]
        model: Which,
        /// Also accept records from this generator (the built-in list is every generator the
        /// token-era pipeline accepted).
        #[arg(long = "allow-generator")]
        allow_generator: Vec<String>,
        /// Refuse records from this generator, e.g. `--deny-generator QwQ-32B`.
        #[arg(long = "deny-generator")]
        deny_generator: Vec<String>,
    },
}

fn main() -> ExitCode {
    let Command::Build { archive, out, model, allow_generator, deny_generator } =
        Cli::parse().command;
    let models = match model {
        Which::Emperor => vec![Model::Emperor],
        Which::Consul => vec![Model::Consul],
        Which::All => vec![Model::Emperor, Model::Consul],
    };
    for model in models {
        let result = policy_for(&archive, model, &allow_generator, &deny_generator)
            .and_then(|policy| build(&archive, &out, model, &policy));
        match result {
            Ok(stats) => {
                let t = &stats.totals;
                println!(
                    "{}: {} in, {} accepted ({} train, {} val), {} need regeneration, {} excluded, \
                     {} characters, {:.1}% with think, {:.1}% with tool calls",
                    stats.model,
                    t.records_in,
                    t.accepted,
                    t.train,
                    t.val,
                    t.needs_regeneration,
                    t.excluded,
                    t.characters,
                    t.share_with_think * 100.0,
                    t.share_with_tool_calls * 100.0
                );
            }
            Err(error) => {
                eprintln!("ncm-sft: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
