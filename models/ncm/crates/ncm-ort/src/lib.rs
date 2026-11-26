//! The ONNX Runtime backend of the NCM engine interface (`docs/plans/npu.md`, stage 3):
//! `OrtBackend` runs the graphs `ncm-export` wrote, on the CPU execution provider.
//!
//! - **Static shapes, host-owned state.** Every cache and recurrent state is a host buffer the
//!   session owns: attention rings (written in place at `position % ring`, so a restore is a
//!   `memcpy` into the same buffers), DeltaNet states (f32) and convolution windows. Each run
//!   copies the buffers into input tensors and the new entries out of the outputs.
//! - **Prefill** takes the largest exported bucket that fits, the rest token by token on the
//!   decode graph. Without prefill graphs it is all decode steps.
//! - **Sampling** is on the host (`PickMode::Host`): this backend has no device sampler.
//! - **Models**: the token baseline (fallback design: GQA attention and DeltaNet layers). The
//!   hybrid's block cycle has its types here ([`OrtBlockRun`] embeds the shared
//!   [`BlockLogic`]) but no graphs yet, so `begin_block` says so; the k = 4 NCM has no ONNX path.
//!
//! The ONNX Runtime shared library is loaded at run time (`ort`'s `load-dynamic`): from
//! [`OrtOptions::dylib`], else `ORT_DYLIB_PATH`, else the platform's default name. Its version
//! must match the one `ort` was built for (1.31) unless `allow_version_skew` is set.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ncm_exec::hybrid::{BlockLogic, BlockOut, BlockPhase, DecodeOptions, Hooks, PassResult};
use ncm_exec::manifest::{
    FILE_NAME, GraphSpec, IoSpec, LayerSpec, Manifest, Phase, Role, RopeSpec,
};
use ncm_exec::retrieval::{IndexCache, candidates, select};
use ncm_exec::rope::{Yarn, rope_rows};
use ncm_exec::{
    Capabilities, Dtype, ExecBackend, ExecError, ExecSession, Footprint, FootprintRequest,
    HostChoice, ModelKind, Part, PendingOutput, PickMode, PickSpec, PrefillOut, SnapshotInfo,
    StepOut, Tier,
};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// The ONNX Runtime minor version `ort` was built for (1.31).
pub const ORT_MINOR: u32 = ort::MINOR_VERSION;

fn backend_error(what: impl std::fmt::Display) -> ExecError {
    ExecError::Backend(what.to_string())
}

/// How to open an export.
#[derive(Clone, Debug, Default)]
pub struct OrtOptions {
    /// The export directory (`manifest.json` and the graphs).
    pub dir: PathBuf,
    /// The ONNX Runtime shared library (default: `ORT_DYLIB_PATH`, else the platform name).
    pub dylib: Option<PathBuf>,
    /// Intra-op threads (0 or None: ONNX Runtime's choice).
    pub threads: Option<usize>,
    /// Accept an ONNX Runtime whose minor version differs from [`ORT_MINOR`].
    pub allow_version_skew: bool,
}

// ---------------------------------------------------------------------- the environment

/// The process-wide environment and the library's build info, created once.
fn environment(dylib: Option<&Path>, allow_skew: bool) -> Result<(Environment, String), ExecError> {
    static ENV: OnceLock<Result<(Environment, String), String>> = OnceLock::new();
    let made = ENV.get_or_init(|| {
        let path = match dylib {
            Some(p) => p.to_path_buf(),
            None => match std::env::var("ORT_DYLIB_PATH") {
                Ok(s) if !s.is_empty() => PathBuf::from(s),
                _ => PathBuf::from(format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                )),
            },
        };
        let builder = ort::init_from(&path).map_err(|e| {
            format!(
                "cannot load the ONNX Runtime library `{}`: {e} (set --ort-lib or ORT_DYLIB_PATH)",
                path.display()
            )
        })?;
        let env = builder.build().map_err(|e| e.to_string())?;
        Ok((env, ort::info().to_owned()))
    });
    let (env, info) = made.as_ref().map_err(backend_error)?;
    if let Some(found) = version_minor(info)
        && found != ORT_MINOR
        && !allow_skew
    {
        return Err(backend_error(format!(
            "ONNX Runtime 1.{found} found, this build is pinned to 1.{ORT_MINOR} \
             (pass --ort-allow-version-skew to try it anyway): {info}"
        )));
    }
    Ok((env.clone(), info.clone()))
}

/// The `m` of `rel-1.m.p` in ONNX Runtime's build info, if it names one.
pub fn version_minor(info: &str) -> Option<u32> {
    let rest = info.split("rel-1.").nth(1)?;
    rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

// ---------------------------------------------------------------------- the backend

struct Graphs {
    decode: Mutex<Session>,
    /// `(tokens, session)` of the prefill buckets, largest first.
    prefill: Vec<(usize, Mutex<Session>)>,
    /// The split decode step's segments, in order (empty without sparse retrieval).
    split: Vec<Mutex<Session>>,
}

/// The backend over one export.
pub struct OrtBackend {
    manifest: Arc<Manifest>,
    graphs: Arc<Graphs>,
    ort_info: String,
    dir: PathBuf,
}

impl OrtBackend {
    /// Loads the manifest and creates a session per graph.
    pub fn load(options: &OrtOptions) -> Result<Self, ExecError> {
        let path = options.dir.join(FILE_NAME);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| backend_error(format!("{}: {e}", path.display())))?;
        let manifest = Manifest::from_json(&text)
            .map_err(|e| backend_error(format!("{}: {e}", path.display())))?;
        if manifest.model != "tok" {
            return Err(ExecError::Unsupported(format!(
                "the export is a `{}` model: this backend runs the token baseline",
                manifest.model
            )));
        }
        if manifest.target != "cpu" {
            return Err(ExecError::Unsupported(format!(
                "the export targets `{}`: this backend runs the CPU execution provider",
                manifest.target
            )));
        }
        let (env, ort_info) = environment(options.dylib.as_deref(), options.allow_version_skew)?;
        let open = |graph: &GraphSpec| -> Result<Mutex<Session>, ExecError> {
            let file = options.dir.join(&graph.file);
            let mut builder = Session::builder(&env)
                .map_err(backend_error)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(backend_error)?;
            if let Some(n) = options.threads.filter(|&n| n > 0) {
                builder = builder.with_intra_threads(n).map_err(backend_error)?;
            }
            let session = builder
                .commit_from_file(&file)
                .map_err(|e| backend_error(format!("{}: {e}", file.display())))?;
            Ok(Mutex::new(session))
        };
        let decode = manifest
            .graph(Phase::Decode, 1)
            .ok_or_else(|| backend_error("the manifest has no decode graph"))?;
        let decode = open(decode)?;
        let mut prefill = Vec::new();
        for graph in manifest.graphs.iter().filter(|g| g.phase == Phase::Prefill) {
            prefill.push((graph.tokens, open(graph)?));
        }
        prefill.sort_by_key(|(tokens, _)| std::cmp::Reverse(*tokens));
        let mut split = Vec::new();
        for graph in manifest.split_segments() {
            split.push(open(graph)?);
        }
        Ok(Self {
            manifest: Arc::new(manifest),
            graphs: Arc::new(Graphs { decode, prefill, split }),
            ort_info,
            dir: options.dir.clone(),
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// ONNX Runtime's build info (version, commit).
    pub fn ort_info(&self) -> &str {
        &self.ort_info
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl ExecBackend for OrtBackend {
    type Session = OrtSession;

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            backend: "ort-cpu".into(),
            model: ModelKind::Tok,
            unit: 1,
            block: 0,
            prefill_buckets: self.manifest.prefill_buckets.clone(),
            dtypes: vec![Dtype::F32],
            device_sampling: false,
            graphs: false,
            speculative: false,
            batching: false,
        }
    }

    fn footprint(&self, req: &FootprintRequest) -> Footprint {
        let m = &self.manifest;
        let weights = m
            .weights_file
            .as_ref()
            .and_then(|f| std::fs::metadata(self.dir.join(f)).ok())
            .map_or(0, |md| md.len());
        let mut f = Footprint::new().with(Tier::SystemRam, Part::Weights, weights);
        for layer in &m.layers {
            match layer {
                LayerSpec::DeltaNet { heads, head_dim, conv } => {
                    let state = heads * head_dim * head_dim;
                    let window = (conv - 1) * 3 * heads * head_dim;
                    f.add(Tier::SystemRam, Part::DeltaNetState, 4 * (state + window) as u64);
                }
                LayerSpec::Attention { kv_heads, head_dim, ring, .. } => {
                    let ring = (*ring).min(req.ctx.saturating_sub(1)).max(1);
                    f.add(
                        Tier::SystemRam,
                        Part::Rings,
                        (2 * 4 * kv_heads * head_dim * ring) as u64,
                    );
                }
                LayerSpec::Retrieval { cache, latent, rope_dim, .. } => {
                    let cache = (*cache).min(req.ctx.saturating_sub(1)).max(1);
                    f.add(
                        Tier::SystemRam,
                        Part::LatentCache,
                        (4 * cache * (latent + rope_dim)) as u64,
                    );
                }
            }
        }
        let slots = req.slots.max(1) as u64;
        for part in [Part::Rings, Part::DeltaNetState, Part::LatentCache] {
            let one = f.get(Tier::SystemRam, part);
            f.add(Tier::SystemRam, part, one * (slots - 1));
        }
        f
    }

    fn new_session(&self, ctx: usize) -> Result<OrtSession, ExecError> {
        if ctx > self.manifest.ctx {
            return Err(backend_error(format!(
                "the export's rings hold {} tokens, {ctx} were asked for: export with a larger --ctx",
                self.manifest.ctx
            )));
        }
        Ok(OrtSession::new(self.manifest.clone(), self.graphs.clone(), ctx))
    }

    fn describe(&self) -> String {
        format!("ort-cpu ({})", self.ort_info)
    }
}

// ---------------------------------------------------------------------- the session

/// What a graph run returned beyond the state it folded in.
struct RunOut {
    logits: Option<Vec<f32>>,
    /// The hidden state, absorbed queries, gate and indexer outputs of a split segment, by
    /// output name.
    boundary: HashMap<String, Vec<f32>>,
}

/// One layer's host-owned state.
#[derive(Clone, Debug, PartialEq)]
enum LayerState {
    Delta {
        conv: Vec<f32>,
        state: Vec<f32>,
    },
    Attn {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    /// A retrieval layer's latent cache `[cache, latent + rope_dim]`.
    Latent {
        kv: Vec<f32>,
        index: IndexCache,
    },
}

/// A point of a session ([`ExecSession::snapshot`]): copies in system RAM.
#[derive(Clone)]
pub struct OrtSnapshot {
    layers: Vec<LayerState>,
    len: usize,
    pending: Option<Vec<f32>>,
}

impl SnapshotInfo for OrtSnapshot {
    fn has_output(&self) -> bool {
        self.pending.is_some()
    }
}

/// The hybrid's block state on this backend: the shared host logic, no graphs behind it yet.
pub struct OrtBlockRun {
    logic: BlockLogic,
}

/// One request stream: its caches, states and the logits its next step starts from.
pub struct OrtSession {
    manifest: Arc<Manifest>,
    graphs: Arc<Graphs>,
    layers: Vec<LayerState>,
    /// Positions in the stream.
    len: usize,
    ctx: usize,
    pending: Option<Vec<f32>>,
    failure: Option<String>,
}

impl OrtSession {
    fn new(manifest: Arc<Manifest>, graphs: Arc<Graphs>, ctx: usize) -> Self {
        let layers = manifest
            .layers
            .iter()
            .map(|l| match l {
                LayerSpec::DeltaNet { heads, head_dim, conv } => LayerState::Delta {
                    conv: vec![0.0; (conv - 1) * 3 * heads * head_dim],
                    state: vec![0.0; heads * head_dim * head_dim],
                },
                LayerSpec::Attention { kv_heads, head_dim, ring, .. } => LayerState::Attn {
                    k: vec![0.0; kv_heads * ring * head_dim],
                    v: vec![0.0; kv_heads * ring * head_dim],
                },
                LayerSpec::Retrieval { cache, latent, rope_dim, index_dim, .. } => {
                    LayerState::Latent {
                        kv: vec![0.0; cache * (latent + rope_dim)],
                        index: IndexCache::new(*index_dim),
                    }
                }
            })
            .collect();
        Self { manifest, graphs, layers, len: 0, ctx, pending: None, failure: None }
    }

    /// The additive attention mask `[t, ring + t]` of `t` new positions starting at `len`, for
    /// a ring of `ring` previous positions: ring slot `s` holds the latest position `p < len`
    /// with `p % ring == s`; query `i` (position `len + i`) sees the previous `ring` positions
    /// and itself.
    pub fn bias(len: usize, t: usize, ring: usize) -> Vec<f32> {
        let cols = ring + t;
        let mut bias = vec![f32::NEG_INFINITY; t * cols];
        for i in 0..t {
            let q = len + i;
            for s in 0..ring {
                if s < len {
                    let p = s + ((len - 1 - s) / ring) * ring;
                    if p + ring >= q {
                        bias[i * cols + s] = 0.0;
                    }
                }
            }
            for j in 0..=i {
                if i - j <= ring {
                    bias[i * cols + ring + j] = 0.0;
                }
            }
        }
        bias
    }

    fn rope(&self, r: usize) -> &RopeSpec {
        &self.manifest.rope[r]
    }

    /// Runs `graph` on `tokens` at the stream's end; returns the last position's logits and
    /// folds the new cache entries and states into the session.
    fn run(
        &mut self,
        graph: &GraphSpec,
        session: &Mutex<Session>,
        tokens: &[i32],
        extra: &HashMap<String, Vec<f32>>,
        advance: bool,
    ) -> Result<RunOut, ExecError> {
        let t = tokens.len();
        if self.len + t > self.ctx {
            return Err(backend_error(format!(
                "{} tokens do not fit the context of {}",
                self.len + t,
                self.ctx
            )));
        }
        let mut feed: Vec<(String, ort::session::SessionInputValue<'_>)> = Vec::new();
        let tensor_f32 = |shape: &[i64],
                          data: Vec<f32>|
         -> Result<ort::session::SessionInputValue<'_>, ExecError> {
            let shape: Vec<i64> = shape.to_vec();
            Tensor::from_array((shape, data)).map(Into::into).map_err(backend_error)
        };
        for io in &graph.inputs {
            // Values the caller computed (the split path hidden state, queries, gathered entries).
            if let Some(data) = extra.get(&io.name) {
                feed.push((io.name.clone(), tensor_f32(&io.shape, data.clone())?));
                continue;
            }
            let value = match (io.role, io.layer, io.rope) {
                (Role::Token, _, _) => {
                    let ids: Vec<i64> = tokens.iter().map(|&x| i64::from(x)).collect();
                    Tensor::from_array((io.shape.clone(), ids))
                        .map(Into::into)
                        .map_err(backend_error)?
                }
                (Role::RopeCos | Role::RopeSin, _, Some(r)) => {
                    let spec = self.rope(r);
                    let (cos, sin) = rope_rows(
                        self.len,
                        t,
                        spec.dim,
                        spec.base,
                        spec.scale,
                        Yarn { factor: spec.yarn_factor, original: spec.yarn_original },
                    );
                    tensor_f32(&io.shape, if io.role == Role::RopeCos { cos } else { sin })?
                }
                (Role::Bias, Some(l), _) => {
                    let ring = self.ring_of(l);
                    tensor_f32(&io.shape, Self::bias(self.len, t, ring))?
                }
                (Role::RingK, Some(l), _) => match &self.layers[l] {
                    LayerState::Attn { k, .. } => tensor_f32(&io.shape, k.clone())?,
                    _ => return Err(self.mismatch(io)),
                },
                (Role::RingV, Some(l), _) => match &self.layers[l] {
                    LayerState::Attn { v, .. } => tensor_f32(&io.shape, v.clone())?,
                    _ => return Err(self.mismatch(io)),
                },
                (Role::Latent, Some(l), _) => match &self.layers[l] {
                    LayerState::Latent { kv, .. } => tensor_f32(&io.shape, kv.clone())?,
                    _ => return Err(self.mismatch(io)),
                },
                (Role::ConvIn, Some(l), _) => match &self.layers[l] {
                    LayerState::Delta { conv, .. } => tensor_f32(&io.shape, conv.clone())?,
                    _ => return Err(self.mismatch(io)),
                },
                (Role::StateIn, Some(l), _) => match &self.layers[l] {
                    LayerState::Delta { state, .. } => tensor_f32(&io.shape, state.clone())?,
                    _ => return Err(self.mismatch(io)),
                },
                _ => return Err(self.mismatch(io)),
            };
            feed.push((io.name.clone(), value));
        }
        let mut guard = session.lock().unwrap_or_else(|p| p.into_inner());
        let outputs = guard.run(feed).map_err(backend_error)?;
        let mut logits = None;
        let mut boundary: HashMap<String, Vec<f32>> = HashMap::new();
        let start = self.len;
        for io in &graph.outputs {
            let value = outputs.get(&io.name).ok_or_else(|| {
                backend_error(format!("the graph `{}` returned no `{}`", graph.name, io.name))
            })?;
            let (_, data) = value.try_extract_tensor::<f32>().map_err(backend_error)?;
            match (io.role, io.layer) {
                (Role::Logits, _) => logits = Some(data.to_vec()),
                (Role::Hidden | Role::QAll | Role::GateVal | Role::IndexQ | Role::IndexW, _) => {
                    boundary.insert(io.name.clone(), data.to_vec());
                }
                (Role::NewIndexKey, Some(l)) => {
                    // [1, t, index_dim]: the normed key of each new position, kept 8-bit.
                    if let LayerState::Latent { index, .. } = &mut self.layers[l] {
                        let di = data.len() / t.max(1);
                        for j in 0..t {
                            index.put(start + j, &data[j * di..(j + 1) * di]);
                        }
                    }
                }
                (Role::ConvOut, Some(l)) => {
                    if let LayerState::Delta { conv, .. } = &mut self.layers[l] {
                        conv.copy_from_slice(data);
                    }
                }
                (Role::NewLatent, Some(l)) => {
                    // `data` is [1, t, w]: position start + j goes to row (start + j) % cache.
                    let w = self.latent_width(l);
                    let cache = self.ring_of(l);
                    if let LayerState::Latent { kv, .. } = &mut self.layers[l] {
                        for j in 0..t {
                            let row = (start + j) % cache;
                            kv[row * w..(row + 1) * w].copy_from_slice(&data[j * w..(j + 1) * w]);
                        }
                    }
                }
                (Role::StateOut, Some(l)) => {
                    if let LayerState::Delta { state, .. } = &mut self.layers[l] {
                        state.copy_from_slice(data);
                    }
                }
                (Role::NewK | Role::NewV, Some(l)) => {
                    let (kvh, hd, ring) = self.attn_dims(l);
                    let target = match &mut self.layers[l] {
                        LayerState::Attn { k, v } => {
                            if io.role == Role::NewK {
                                k
                            } else {
                                v
                            }
                        }
                        _ => continue,
                    };
                    // `data` is [1, kvh, t, hd]: position start + j goes to slot (start + j) % ring.
                    for h in 0..kvh {
                        for j in 0..t {
                            let slot = (start + j) % ring;
                            let src = (h * t + j) * hd;
                            let dst = (h * ring + slot) * hd;
                            target[dst..dst + hd].copy_from_slice(&data[src..src + hd]);
                        }
                    }
                }
                _ => {}
            }
        }
        drop(outputs);
        drop(guard);
        if advance {
            self.len += t;
        }
        Ok(RunOut { logits, boundary })
    }

    /// The first position the sparse selection differs from dense attention at (the smallest
    /// `dense_exact_until` over the retrieval layers), if the export has split graphs.
    fn dense_until(&self) -> Option<usize> {
        if self.graphs.split.is_empty() {
            return None;
        }
        self.manifest
            .layers
            .iter()
            .filter_map(|l| match l {
                LayerSpec::Retrieval { dense_exact_until, .. } => Some(*dense_exact_until),
                _ => None,
            })
            .min()
    }

    /// One token at position `len` through the split decode step: segment by segment, with the
    /// host's indexer scores, top-k and gather between them (UNVERIFIED).
    fn step_split(&mut self, token: i32) -> Result<(), ExecError> {
        let graphs = self.graphs.clone();
        let manifest = self.manifest.clone();
        let specs = manifest.split_segments();
        let retr: Vec<usize> = manifest
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, LayerSpec::Retrieval { .. }))
            .map(|(i, _)| i)
            .collect();
        let pos = self.len;
        let mut extra: HashMap<String, Vec<f32>> = HashMap::new();
        for (k, spec) in specs.iter().enumerate() {
            let last = k == retr.len();
            let out = self.run(spec, &graphs.split[k], &[token], &extra, last)?;
            if last {
                self.pending = out.logits;
                return Ok(());
            }
            let r = retr[k];
            let LayerSpec::Retrieval { block, topk, local, latent, rope_dim, cache, .. } =
                manifest.layers[r]
            else {
                unreachable!("a retrieval layer");
            };
            let take = |name: String| -> Result<Vec<f32>, ExecError> {
                out.boundary
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| backend_error(format!("segment {k} returned no `{name}`")))
            };
            let (q, w) = (take(format!("l{r}.index_q"))?, take(format!("l{r}.index_w"))?);
            let width = latent + rope_dim;
            let LayerState::Latent { kv, index } = &self.layers[r] else {
                return Err(backend_error("a retrieval layer without a latent cache"));
            };
            let n = candidates(pos, block, local);
            let scores = index.scores(&q, &w, n * block);
            let sel = select(pos, block, topk, local, &scores).ok_or_else(|| {
                backend_error("the split path ran where dense attention is exact")
            })?;
            let mut gathered = Vec::with_capacity(sel.positions.len() * width);
            for &p in &sel.positions {
                let row = p % cache;
                gathered.extend_from_slice(&kv[row * width..(row + 1) * width]);
            }
            extra.clear();
            extra.insert("hidden".into(), take("hidden_out".into())?);
            extra.insert(format!("l{r}.q_all"), take(format!("l{r}.q_all"))?);
            extra.insert(format!("l{r}.gate"), take(format!("l{r}.gate"))?);
            extra.insert(format!("l{r}.gathered"), gathered);
            extra.insert(format!("l{r}.bias"), sel.bias);
        }
        Err(backend_error("the split decode step has no final segment"))
    }

    fn mismatch(&self, io: &IoSpec) -> ExecError {
        backend_error(format!(
            "the graph input `{}` ({:?}) does not fit the model",
            io.name, io.role
        ))
    }

    fn ring_of(&self, layer: usize) -> usize {
        match &self.manifest.layers[layer] {
            LayerSpec::Attention { ring, .. } => *ring,
            LayerSpec::Retrieval { cache, .. } => *cache,
            LayerSpec::DeltaNet { .. } => 1,
        }
    }

    fn latent_width(&self, layer: usize) -> usize {
        match &self.manifest.layers[layer] {
            LayerSpec::Retrieval { latent, rope_dim, .. } => latent + rope_dim,
            _ => 0,
        }
    }

    fn attn_dims(&self, layer: usize) -> (usize, usize, usize) {
        match &self.manifest.layers[layer] {
            LayerSpec::Attention { kv_heads, head_dim, ring, .. } => (*kv_heads, *head_dim, *ring),
            _ => (0, 0, 1),
        }
    }

    fn step_one(&mut self, token: i32) -> Result<(), ExecError> {
        let graphs = self.graphs.clone();
        let manifest = self.manifest.clone();
        let spec = manifest.graph(Phase::Decode, 1).expect("loaded with a decode graph");
        let step = if self.dense_until().is_some_and(|until| self.len >= until) {
            self.step_split(token)
        } else {
            self.run(spec, &graphs.decode, &[token], &HashMap::new(), true).and_then(|out| {
                self.pending = out.logits;
                self.pending
                    .is_some()
                    .then_some(())
                    .ok_or_else(|| backend_error("the decode graph returned no logits"))
            })
        };
        step.inspect_err(|e| self.failure = Some(e.to_string()))
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

impl ExecSession for OrtSession {
    type BlockRun = OrtBlockRun;
    type Snapshot = OrtSnapshot;

    fn begin_request(&mut self, _seed: u64) {}

    fn reset(&mut self) {
        // The buffers keep their addresses; the next request's masks hide the old contents.
        for layer in &mut self.layers {
            match layer {
                LayerState::Delta { conv, state } => {
                    conv.fill(0.0);
                    state.fill(0.0);
                }
                LayerState::Attn { k, v } => {
                    k.fill(0.0);
                    v.fill(0.0);
                }
                LayerState::Latent { kv, index } => {
                    kv.fill(0.0);
                    index.truncate(0);
                }
            }
        }
        self.len = 0;
        self.pending = None;
        self.failure = None;
    }

    fn snapshot(&self) -> OrtSnapshot {
        OrtSnapshot { layers: self.layers.clone(), len: self.len, pending: self.pending.clone() }
    }

    fn restore(&mut self, snapshot: &OrtSnapshot) {
        for (mine, theirs) in self.layers.iter_mut().zip(&snapshot.layers) {
            match (mine, theirs) {
                (LayerState::Delta { conv, state }, LayerState::Delta { conv: c, state: s }) => {
                    conv.copy_from_slice(c);
                    state.copy_from_slice(s);
                }
                (LayerState::Attn { k, v }, LayerState::Attn { k: ks, v: vs }) => {
                    k.copy_from_slice(ks);
                    v.copy_from_slice(vs);
                }
                (LayerState::Latent { kv, index }, LayerState::Latent { kv: s, index: si }) => {
                    kv.copy_from_slice(s);
                    index.clone_from(si);
                }
                _ => panic!("a snapshot of another model"),
            }
        }
        self.len = snapshot.len;
        self.pending = snapshot.pending.clone();
    }

    fn prefill(&mut self, tokens: &[i32], _chunk: usize) -> Result<PrefillOut, ExecError> {
        let graphs = self.graphs.clone();
        let manifest = self.manifest.clone();
        let mut rest = tokens;
        while !rest.is_empty() {
            // The largest bucket that fits, else one token on the decode graph.
            // (A bucket runs dense attention: only where every position is below the point the
            // sparse selection starts to differ.)
            let until = self.dense_until().unwrap_or(usize::MAX);
            let bucket = graphs
                .prefill
                .iter()
                .find(|(n, _)| *n <= rest.len() && self.len.saturating_add(*n) <= until);
            match bucket {
                Some((n, session)) => {
                    let spec = manifest.graph(Phase::Prefill, *n).expect("a loaded bucket");
                    let out = self
                        .run(spec, session, &rest[..*n], &HashMap::new(), true)
                        .inspect_err(|e| {
                            self.failure = Some(e.to_string());
                        })?;
                    self.pending = out.logits;
                    rest = &rest[*n..];
                }
                None => {
                    self.step_one(rest[0])?;
                    rest = &rest[1..];
                }
            }
        }
        Ok(PrefillOut { prefilled: tokens.len(), output: PendingOutput::Logits })
    }

    fn pending_logits(&mut self) -> Option<Vec<f32>> {
        self.pending.clone()
    }

    fn decode_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError> {
        if let Some(&token) = feed.first() {
            self.step_one(token)?;
        }
        let Some(logits) = &self.pending else {
            return Err(backend_error("a step without a prompt"));
        };
        match &pick.mode {
            PickMode::Host => Ok(StepOut::Logits(logits.clone())),
            PickMode::Greedy => Ok(StepOut::Picked(vec![argmax(logits) as u32])),
            PickMode::Sampled(_) => {
                Err(ExecError::Unsupported("this backend has no device sampler".into()))
            }
        }
    }

    fn decode_concept_host(
        &mut self,
        _feed: &[i32],
        _forced: &[i32],
        _choose: &mut HostChoice<'_>,
    ) -> Result<Vec<u32>, ExecError> {
        Err(ExecError::Unsupported("the k = 4 NCM has no ONNX path".into()))
    }

    fn begin_block(
        &mut self,
        _limit: usize,
        _opts: &DecodeOptions,
    ) -> Result<OrtBlockRun, ExecError> {
        Err(ExecError::Unsupported(
            "the export has no hybrid block graphs (the exporter does not write them yet)".into(),
        ))
    }

    fn phase(&self, run: &OrtBlockRun) -> BlockPhase {
        run.logic.phase()
    }

    fn logic<'a>(&self, run: &'a OrtBlockRun) -> &'a BlockLogic {
        &run.logic
    }

    fn pass(
        &mut self,
        _run: &mut OrtBlockRun,
        _opts: &DecodeOptions,
        _hooks: &mut Hooks,
    ) -> Result<PassResult, ExecError> {
        Err(ExecError::Unsupported("no hybrid block graphs".into()))
    }

    fn fill(
        &mut self,
        _run: &mut OrtBlockRun,
        _opts: &DecodeOptions,
        _hooks: &mut Hooks,
    ) -> Result<i32, ExecError> {
        Err(ExecError::Unsupported("no hybrid block graphs".into()))
    }

    fn end_block(&mut self, _run: OrtBlockRun) -> Result<BlockOut, ExecError> {
        Err(ExecError::Unsupported("no hybrid block graphs".into()))
    }

    fn failure(&self) -> Option<String> {
        self.failure.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_comes_out_of_the_build_info() {
        let info = "ORT Build Info: git-branch=rel-1.31.0, git-commit-id=abc, build type=Release";
        assert_eq!(version_minor(info), Some(31));
        assert_eq!(version_minor("something else"), None);
        assert_eq!(version_minor("git-branch=rel-1.9.1"), Some(9));
    }

    #[test]
    fn a_single_new_token_sees_the_filled_slots_and_itself() {
        // Ring of 4, 2 positions so far: slots 0 and 1 hold them.
        let b = OrtSession::bias(2, 1, 4);
        assert_eq!(b.len(), 5);
        assert_eq!(&b[..2], &[0.0, 0.0]);
        assert!(b[2].is_infinite() && b[3].is_infinite());
        assert_eq!(b[4], 0.0);
    }

    #[test]
    fn a_full_ring_wraps_and_every_slot_is_visible() {
        let b = OrtSession::bias(10, 1, 4);
        assert!(b.iter().all(|&v| v == 0.0), "{b:?}");
    }

    #[test]
    fn a_prefill_chunk_is_causal_and_windowed() {
        // Ring 2 (window 3), 3 earlier positions (0, 1, 2: slots 0, 1, 0), 4 new ones.
        let (ring, len, t) = (2, 3, 4);
        let b = OrtSession::bias(len, t, ring);
        let cols = ring + t;
        let at = |i: usize, c: usize| b[i * cols + c];
        // Slot 0 holds position 2, slot 1 holds position 1. Query 0 (position 3) sees both.
        assert_eq!((at(0, 0), at(0, 1)), (0.0, 0.0));
        // Query 1 (position 4) sees positions 2 (slot 0) and 3 (new 0), not position 1 (slot 1).
        assert_eq!(at(1, 0), 0.0);
        assert!(at(1, 1).is_infinite());
        // Causal: query 1 does not see new position 2.
        assert!(at(1, ring + 2).is_infinite());
        assert_eq!(at(1, ring), 0.0);
        // Query 3 (position 6) sees new 1, 2 and itself (window 3), not new 0.
        assert!(at(3, ring).is_infinite());
        assert_eq!((at(3, ring + 1), at(3, ring + 2), at(3, ring + 3)), (0.0, 0.0, 0.0));
        // And no ring slot (positions 1 and 2 are out of window).
        assert!(at(3, 0).is_infinite() && at(3, 1).is_infinite());
    }

    #[test]
    fn an_empty_stream_sees_only_itself() {
        let b = OrtSession::bias(0, 1, 3);
        assert_eq!(b.iter().filter(|v| **v == 0.0).count(), 1);
        assert_eq!(b[3], 0.0);
    }
}
