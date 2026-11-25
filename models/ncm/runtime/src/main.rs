use std::process::ExitCode;

use ncm_server::app::start;
use ncm_server::cli::Cli;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse_normalized();
    let running = match start(cli).await {
        Ok(running) => running,
        Err(message) => {
            eprintln!("ncm-server: {message}");
            return ExitCode::FAILURE;
        }
    };
    let stop = running.stop_signal();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        stop.notify_one();
    });
    match running.wait().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ncm-server: {message}");
            ExitCode::FAILURE
        }
    }
}
