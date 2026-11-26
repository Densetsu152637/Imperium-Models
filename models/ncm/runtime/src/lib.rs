//! `ncm-server`: an OpenAI-compatible HTTP server over the NCM and the token baseline, launched
//! and spoken to like llama.cpp's `llama-server`. See `README.md` in the crate folder.

pub mod api;
pub mod app;
pub mod batch;
pub mod chat;
pub mod cli;
pub mod constrain;
pub mod engine;
pub mod sampling;
pub mod server;
pub mod testing;
pub mod vocab;
