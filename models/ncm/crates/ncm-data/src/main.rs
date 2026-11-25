//! `ncm-data`: the stage 2 data pipeline. See `models/ncm/README.md` for the full flow.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use ncm_data::error::{Error, IoContext, Result};
use ncm_data::hub::{self, Hub, Task};
use ncm_data::lock::Lock;
use ncm_data::manifest::{Mix, Source};
use ncm_data::sample::{self, SampleOptions};
use ncm_data::{model_vocab_size, parse_count, shards, stats, tokenize, tokenizer, workspace_root};

#[derive(Parser)]
#[command(name = "ncm-data", about = "NCM stage 2 data pipeline", version)]
struct Cli {
    /// Source manifest (default: configs/data-mix.toml).
    #[arg(long, global = true)]
    mix: Option<PathBuf>,
    /// Data directory (default: data/, ignored by Git).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the manifest and print the mix.
    Manifest,
    /// Download the manifest's files into data/raw/ and record them in manifest.lock.json.
    Fetch(FetchArgs),
    /// Draw a byte-limited sample of every source for tokenizer training.
    Sample(SampleArgs),
    /// Train or inspect the shared tokenizer.
    #[command(subcommand)]
    Tokenizer(TokenizerCmd),
    /// Tokenise the fetched files into train and validation shards.
    Tokenize(TokenizeArgs),
    /// Per-source documents, bytes, tokens and bytes per token.
    Stats(StatsArgs),
}

#[derive(Args)]
struct FetchArgs {
    /// Only these sources (repeatable); default all.
    #[arg(long = "source")]
    sources: Vec<String>,
    /// Take at most this many files per source.
    #[arg(long)]
    max_files: Option<usize>,
    /// Download at most this many bytes per source, e.g. 200M. Line-oriented files are cut to a
    /// prefix; Parquet files cannot be cut, so the first one is taken whole.
    #[arg(long, value_parser = parse_count)]
    limit_bytes: Option<u64>,
    /// Concurrent downloads.
    #[arg(long, default_value_t = 4)]
    jobs: usize,
    /// Print what would be downloaded and the sizes, then stop.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Args)]
struct SampleArgs {
    #[arg(long = "source")]
    sources: Vec<String>,
    /// Total sample size, split across sources by share.
    #[arg(long, default_value = "100M", value_parser = parse_count)]
    bytes: u64,
    /// Cut documents to this many bytes.
    #[arg(long, default_value = "64K", value_parser = parse_count)]
    max_doc_bytes: u64,
    /// Output (default: data/sample/sample.jsonl).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Subcommand)]
enum TokenizerCmd {
    /// Train the byte-level BPE on the sample.
    Train {
        #[arg(long)]
        sample: Option<PathBuf>,
        /// Vocabulary size including special tokens (default: the models' vocab_size).
        #[arg(long)]
        vocab_size: Option<usize>,
        /// Output (default: data/tokenizer/tokenizer.json).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Print the vocabulary size and the special token ids.
    Info {
        #[arg(long)]
        file: Option<PathBuf>,
    },
}

#[derive(Args)]
struct TokenizeArgs {
    #[arg(long = "source")]
    sources: Vec<String>,
    /// Token budget the shares are fractions of (default: the manifest's budget).
    #[arg(long, value_parser = parse_count)]
    budget_tokens: Option<u64>,
    /// Tokens per shard file.
    #[arg(long, default_value = "134217728", value_parser = parse_count)]
    shard_tokens: u64,
    /// Skip documents over this many bytes.
    #[arg(long, default_value = "8M", value_parser = parse_count)]
    max_doc_bytes: u64,
}

#[derive(Args)]
struct StatsArgs {
    /// Print the index JSON instead of the table.
    #[arg(long)]
    json: bool,
}

struct Ctx {
    root: PathBuf,
    mix_path: PathBuf,
    data: PathBuf,
}

impl Ctx {
    fn mix(&self) -> Result<Mix> {
        Mix::load(&self.mix_path)
    }
}

fn print_manifest(mix: &Mix) {
    println!(
        "budget {:.1}B tokens, validation fraction {}",
        mix.budget_tokens as f64 / 1e9,
        mix.val_fraction
    );
    println!(
        "{:<20} {:<9} {:<6} {:>6} {:>8}  {:<12} repo @ revision",
        "source", "category", "phase", "share", "Mtokens", "licence"
    );
    for s in &mix.sources {
        // The first words of the licence field; the full text is in the manifest.
        let licence: String = s.license.split([' ', '(', ',']).next().unwrap_or("").into();
        println!(
            "{:<20} {:<9} {:<6} {:>5.1}% {:>8.0}  {:<12} {} @ {}",
            s.name,
            format!("{:?}", s.category).to_lowercase(),
            format!("{:?}", s.phase).to_lowercase(),
            s.share * 100.0,
            s.target_tokens(mix.budget_tokens) as f64 / 1e6,
            licence,
            s.repo,
            &s.revision[..12]
        );
    }
    println!(
        "anneal: last {:.0}% of tokens, mix {:?}",
        mix.anneal.fraction * 100.0,
        mix.anneal.shares
    );
}

fn fetch(ctx: &Ctx, args: FetchArgs) -> Result<()> {
    let mix = ctx.mix()?;
    let hub = Hub::default();
    let lock_path = ctx.data.join("manifest.lock.json");
    let mut lock = Lock::load(&lock_path)?;
    let mut work: Vec<(&Source, Task)> = Vec::new();
    for source in mix.select(&args.sources)? {
        let listing = hub.list(&source.repo, &source.revision)?;
        let limit = args.limit_bytes.or(source.fetch_bytes);
        let tasks = hub::plan(source, &listing, args.max_files, limit);
        if tasks.is_empty() {
            return Err(Error::Manifest(format!(
                "{}: glob {:?} matches nothing in {}@{}",
                source.name,
                source.glob,
                source.repo,
                &source.revision[..12]
            )));
        }
        let bytes: u64 = tasks.iter().map(|t| t.limit.unwrap_or(t.file.size)).sum();
        println!("{:<20} {:>4} files {:>9.2} GB", source.name, tasks.len(), bytes as f64 / 1e9);
        work.extend(tasks.into_iter().map(|t| (source, t)));
    }
    let total: u64 = work.iter().map(|(_, t)| t.limit.unwrap_or(t.file.size)).sum();
    println!("total {} files, {:.2} GB", work.len(), total as f64 / 1e9);
    if args.dry_run {
        return Ok(());
    }
    let (done, failed) = hub::fetch_all(&hub, &ctx.data, &work, args.jobs);
    // The lock describes the latest plan: drop a source's earlier entries once it was re-planned.
    let planned: Vec<&str> = work.iter().map(|(s, _)| s.name.as_str()).collect();
    let all_ok = |name: &str| !failed.iter().any(|f| f.starts_with(&format!("{name}/")));
    lock.files.retain(|e| !(planned.contains(&e.source.as_str()) && all_ok(&e.source)));
    lock.merge(done);
    std::fs::create_dir_all(&ctx.data).ctx("creating the data directory")?;
    lock.save(&lock_path)?;
    if failed.is_empty() {
        Ok(())
    } else {
        Err(Error::Net(format!("{} downloads failed:\n  {}", failed.len(), failed.join("\n  "))))
    }
}

fn sample_cmd(ctx: &Ctx, args: SampleArgs) -> Result<()> {
    let mix = ctx.mix()?;
    let out = args.out.unwrap_or_else(|| ctx.data.join("sample").join("sample.jsonl"));
    let opts = SampleOptions {
        bytes: args.bytes,
        max_doc_bytes: args.max_doc_bytes as usize,
        val_fraction: mix.val_fraction,
    };
    let reports = sample::run(&mix, &ctx.data, &args.sources, &out, &opts)?;
    let total: u64 = reports.iter().map(|r| r.bytes).sum();
    println!("wrote {} ({:.1} MB)", out.display(), total as f64 / 1e6);
    Ok(())
}

fn tokenizer_cmd(ctx: &Ctx, cmd: TokenizerCmd) -> Result<()> {
    let default = ctx.data.join("tokenizer").join("tokenizer.json");
    match cmd {
        TokenizerCmd::Train { sample, vocab_size, out } => {
            let sample = sample.unwrap_or_else(|| ctx.data.join("sample").join("sample.jsonl"));
            sample::check_nonempty(&sample)?;
            let model_vocab = model_vocab_size(&ctx.root)?;
            let vocab = vocab_size.unwrap_or(model_vocab);
            if vocab != model_vocab {
                eprintln!("warning: the model configs use vocab_size {model_vocab}, not {vocab}");
            }
            let docs = sample::read_texts(&sample)?;
            let tok = tokenizer::train(docs, vocab)?;
            let out = out.unwrap_or(default);
            tokenizer::save(&tok, &out)?;
            println!("trained {} ids -> {}", tok.get_vocab_size(true), out.display());
            Ok(())
        }
        TokenizerCmd::Info { file } => {
            let tok = tokenizer::load(&file.unwrap_or(default))?;
            println!("vocab {} ids", tok.get_vocab_size(true));
            for (id, name) in tokenizer::SPECIAL_TOKENS.iter().enumerate() {
                println!("{id:>3} {name}");
            }
            Ok(())
        }
    }
}

fn tokenize_cmd(ctx: &Ctx, args: TokenizeArgs) -> Result<()> {
    let mix = ctx.mix()?;
    let opts = tokenize::Options {
        budget_tokens: args.budget_tokens.unwrap_or(mix.budget_tokens),
        shard_tokens: args.shard_tokens,
        val_fraction: mix.val_fraction,
        max_doc_bytes: args.max_doc_bytes as usize,
        batch_docs: 512,
    };
    let index = tokenize::run(&mix, &ctx.data, &args.sources, &opts)?;
    print!("{}", stats::report(&index));
    Ok(())
}

fn stats_cmd(ctx: &Ctx, args: StatsArgs) -> Result<()> {
    let index = shards::TokenIndex::load(&ctx.data.join("tokens"))?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&index)?);
    } else {
        print!("{}", stats::report(&index));
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    let root = workspace_root();
    let ctx = Ctx {
        mix_path: cli.mix.unwrap_or_else(|| root.join("configs").join("data-mix.toml")),
        data: cli.data_dir.unwrap_or_else(|| root.join("data")),
        root,
    };
    match cli.command {
        Command::Manifest => {
            print_manifest(&ctx.mix()?);
            Ok(())
        }
        Command::Fetch(a) => fetch(&ctx, a),
        Command::Sample(a) => sample_cmd(&ctx, a),
        Command::Tokenizer(c) => tokenizer_cmd(&ctx, c),
        Command::Tokenize(a) => tokenize_cmd(&ctx, a),
        Command::Stats(a) => stats_cmd(&ctx, a),
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
