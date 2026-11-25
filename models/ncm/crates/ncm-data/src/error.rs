//! One error type for the whole pipeline.

use std::fmt::Display;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{context}: {source}")]
    Io { context: String, source: std::io::Error },
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("network: {0}")]
    Net(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("data: {0}")]
    Data(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Attach what was being done to an I/O failure.
pub trait IoContext<T> {
    fn ctx(self, what: impl Display) -> Result<T>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn ctx(self, what: impl Display) -> Result<T> {
        self.map_err(|source| Error::Io { context: what.to_string(), source })
    }
}
