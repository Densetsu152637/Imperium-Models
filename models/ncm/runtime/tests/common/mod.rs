//! Shared helpers: tiny models on the CPU backend, a scripted backend that replays fixed token
//! sequences, and an in-process server on an ephemeral port.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use burn::tensor::FloatDType;
use ncm_model::{
    Config, NcmConfig, TokConfig, tiny_ncm, tiny_ncm_v2, tiny_tok, tiny_tok_v2, tiny_v2,
};
use ncm_server::api::ApiError;
use ncm_server::engine::{
    Backend, DeviceChoice, Engine, EngineError, EngineOptions, Finish, GenRequest, GenStats,
    ModelInfo, make_device,
};
use ncm_server::sampling::SamplingParams;
use ncm_server::server::{AppState, router};
use ncm_server::testing::{BYTE_TOKENIZER_SIZE, byte_tokenizer};
use ncm_server::vocab::Vocab;

pub fn tiny_tok_config() -> Config {
    Config::Tok(TokConfig { vocab_size: BYTE_TOKENIZER_SIZE, ..tiny_tok(8) })
}

pub fn tiny_ncm_config() -> Config {
    Config::Ncm(NcmConfig { vocab_size: BYTE_TOKENIZER_SIZE, ..tiny_ncm(8, 4) })
}

/// The v2 attention design: retrieval over blocks of 4 tokens, top 2 plus the last 8.
pub fn tiny_tok_v2_config() -> Config {
    Config::Tok(TokConfig { vocab_size: BYTE_TOKENIZER_SIZE, ..tiny_tok_v2(tiny_v2(4, 2, 8)) })
}

/// The v2 attention design: retrieval over blocks of 2 concepts, top 2 plus the last 4.
pub fn tiny_ncm_v2_config() -> Config {
    Config::Ncm(NcmConfig { vocab_size: BYTE_TOKENIZER_SIZE, ..tiny_ncm_v2(8, tiny_v2(2, 2, 4)) })
}

pub fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(byte_tokenizer(), BYTE_TOKENIZER_SIZE).expect("byte tokenizer"))
}

pub fn engine(config: Config, ctx: usize) -> Engine {
    let options = EngineOptions {
        config,
        ctx,
        device: make_device(DeviceChoice::Cpu, FloatDType::F32).expect("cpu device"),
        device_name: "cpu".into(),
        dtype: FloatDType::F32,
        weights: None,
        seed: 3,
        prefill_chunk: 16,
        speculate: false,
        prefix_reuse: true,
        quant: None,
    };
    Engine::new(options, vocab()).expect("a tiny engine")
}

pub fn request(prompt: Vec<u32>, max_tokens: usize, sampling: SamplingParams) -> GenRequest {
    GenRequest { prompt, max_tokens, sampling, constraint: None }
}

/// Replays `tokens` for every request and records the prompts it was given.
pub struct Scripted {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    tokens: Vec<u32>,
    pub prompts: Mutex<Vec<Vec<u32>>>,
}

impl Scripted {
    /// Replays the tokens of `reply` (special tags become their ids).
    pub fn new(reply: &str, ctx: usize) -> Self {
        let vocab = vocab();
        let tokens = vocab.encode(reply).expect("encodable");
        let info = ModelInfo {
            arch: "tok",
            params: 1234,
            ctx,
            cache_bytes: 4096,
            dtype: "f32".into(),
            device: "cpu".into(),
            vocab_size: BYTE_TOKENIZER_SIZE,
            unit: 1,
        };
        Self { info, vocab, tokens, prompts: Mutex::new(Vec::new()) }
    }
}

impl Backend for Scripted {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    fn check_prompt(&self, tokens: usize) -> Result<(), EngineError> {
        if tokens > self.info.ctx {
            return Err(EngineError::ContextExceeded { prompt: tokens, ctx: self.info.ctx });
        }
        Ok(())
    }

    fn generate(
        &self,
        request: GenRequest,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        self.prompts.lock().unwrap().push(request.prompt.clone());
        let mut sent = 0;
        let mut finish = Finish::Length;
        for &token in self.tokens.iter().take(request.max_tokens) {
            sent += 1;
            if !on_token(token) {
                finish = Finish::Stop;
                break;
            }
            if self.vocab.ids.is_stop(token) {
                finish = Finish::Stop;
                break;
            }
        }
        if sent < request.max_tokens && sent == self.tokens.len() {
            finish = Finish::Stop;
        }
        Ok(GenStats {
            prompt_tokens: request.prompt.len(),
            completion_tokens: sent,
            finish,
            prompt_s: 0.002,
            decode_s: 0.004,
            decode_tokens: sent,
            cached_tokens: 0,
            capture_s: 0.0,
            path: ncm_server::engine::DecodePath::Eager,
        })
    }
}

/// A running in-process server.
pub struct TestServer {
    pub base: String,
    pub state: Arc<AppState>,
    handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    pub async fn start(state: Arc<AppState>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = router(state.clone());
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        Self { base, state, handle }
    }

    pub async fn scripted(reply: &str, ctx: usize) -> (Self, Arc<Scripted>) {
        let backend = Arc::new(Scripted::new(reply, ctx));
        let state = Arc::new(AppState::ready(backend.clone(), "scripted", SamplingParams::GREEDY));
        (Self::start(state).await, backend)
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// The `data:` payloads of a server-sent events body.
pub fn sse_payloads(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter_map(|event| event.trim().strip_prefix("data:"))
        .map(|data| data.trim().to_owned())
        .collect()
}

pub fn error_body(error: &ApiError) -> serde_json::Value {
    error.body()
}
