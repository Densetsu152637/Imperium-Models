//! The token baseline's phase graphs and the export driver.
//!
//! Per phase one static-shape graph (decode step; one per prefill bucket), every ring, state and
//! convolution window an explicit input and output, every weight named by its Burn module path
//! (so all graphs share one external-data file and a weight appears once on disk).

use std::path::Path;

use ncm_exec::manifest::{
    FILE_NAME, FORMAT, GraphSpec, IoSpec, LayerSpec, Manifest, Phase, Role, RopeSpec,
};
use ncm_model::layers::LayerKind;
use ncm_model::tok::tok_spec;
use ncm_model::{Config, TokConfig};

use crate::ExportError;
use crate::graph::{Graph, Val, WeightPool};
use crate::layers::{AttnDims, Cx, DeltaDims, DeltaWeights};
use crate::onnx::DType;
use crate::quant::WeightForm;
use crate::weights::{F32Tensor, Weights};

/// Where the graphs will run: per-target profiles set the NPU-specific choices (which DQ form,
/// whether the f32 inverse is split out). Only the CPU profile exists until each NPU target's
/// smoke test (plan stage 0b) has settled them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cpu,
    Intel,
    Qualcomm,
    Amd,
}

impl Target {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Target::Cpu),
            "intel" => Ok(Target::Intel),
            "qualcomm" => Ok(Target::Qualcomm),
            "amd" => Ok(Target::Amd),
            other => Err(format!("unknown target `{other}` (cpu, intel, qualcomm, amd)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Target::Cpu => "cpu",
            Target::Intel => "intel",
            Target::Qualcomm => "qualcomm",
            Target::Amd => "amd",
        }
    }
}

/// What to export.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Context tokens the attention rings are sized for.
    pub ctx: usize,
    /// Prefill bucket lengths (each a multiple of the DeltaNet chunk when the model has one).
    pub buckets: Vec<usize>,
    pub form: WeightForm,
    pub target: Target,
}

/// Whether a phase graph can be written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Support {
    Supported,
    /// Not implemented, and why.
    Unsupported(String),
}

/// One phase graph of an export, as [`describe`] reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhasePlan {
    pub name: String,
    pub phase: Option<Phase>,
    pub tokens: usize,
    pub support: Support,
}

fn kinds_support(cfg: &TokConfig) -> Support {
    let spec = tok_spec(cfg);
    for (i, kind) in spec.kinds.iter().enumerate() {
        match kind {
            LayerKind::DeltaNet => {}
            LayerKind::Attention { .. } if spec.gated => {
                return Support::Unsupported(format!(
                    "layer {i}: the v2 design's gated attention with QK-norm"
                ));
            }
            LayerKind::Attention { .. } => {}
            LayerKind::Retrieval => {
                return Support::Unsupported(format!(
                    "layer {i} is a sparse-retrieval layer: the absorbed-MLA attention and the \
                     indexer are not exported yet"
                ));
            }
        }
    }
    Support::Supported
}

/// The phase graphs an export of `config` consists of, and which of them can be written.
pub fn describe(config: &Config, buckets: &[usize]) -> Vec<PhasePlan> {
    match config {
        Config::Tok(cfg) => {
            let support = kinds_support(cfg);
            let mut plans = vec![PhasePlan {
                name: "decode".into(),
                phase: Some(Phase::Decode),
                tokens: 1,
                support: support.clone(),
            }];
            let chunk = cfg.v2.dn_chunk;
            for &b in buckets {
                let has_delta = tok_spec(cfg).kinds.contains(&LayerKind::DeltaNet);
                let support = if has_delta && !b.is_multiple_of(chunk) {
                    Support::Unsupported(format!(
                        "bucket {b} is not a multiple of the DeltaNet chunk {chunk}"
                    ))
                } else {
                    support.clone()
                };
                plans.push(PhasePlan {
                    name: format!("prefill_{b}"),
                    phase: Some(Phase::Prefill),
                    tokens: b,
                    support,
                });
            }
            plans
        }
        Config::Ncm(c) if c.hybrid.is_some() => {
            ["backbone_step", "encoder_step", "block_pass", "fill_pass", "token_check"]
                .iter()
                .map(|n| {
                    PhasePlan {
            name: (*n).into(),
            phase: None,
            tokens: c.hybrid.as_ref().map_or(0, |h| h.block),
            support: Support::Unsupported(
                "the hybrid's phase graphs (generator levels, codebook lookup, indexer split) \
                 are planned, not written"
                    .into(),
            ),
        }
                })
                .collect()
        }
        Config::Ncm(_) => vec![PhasePlan {
            name: "decode".into(),
            phase: None,
            tokens: 1,
            support: Support::Unsupported(
                "the k = 4 NCM has no ONNX path (plan scope: Burn adapter only)".into(),
            ),
        }],
    }
}

// ---------------------------------------------------------------------- the weights of a model

/// Looks a layer weight up by its candidate paths.
struct Names<'a> {
    w: &'a Weights,
}

impl Names<'_> {
    fn get(&self, name: &str) -> Result<F32Tensor, ExportError> {
        self.w.get(name)
    }

    /// A mixer weight. `Mixer` is a Burn module enum and the weight files are written with
    /// `skip_enum_variants` off (`ncm_model::weights::save_weights`), so the variant name is a
    /// path segment (burn-derive `codegen_enum` enters the module under the variant): exactly
    /// `stack.blocks.{block}.attn.{Variant}.{leaf}`. `ncm-export --list-tensors` prints a file's.
    fn mixer(&self, block: usize, variant: &str, leaf: &str) -> Result<F32Tensor, ExportError> {
        self.w.get(&format!("stack.blocks.{block}.attn.{variant}.{leaf}"))
    }

    /// The weight key a mixer weight is stored under in the graphs (variant-free, so the
    /// manifest and the graphs agree whichever path the file used).
    fn key(block: usize, leaf: &str) -> String {
        format!("stack.blocks.{block}.mixer.{leaf}")
    }
}

fn expect_shape(t: &F32Tensor, shape: &[usize], what: &str) -> Result<(), ExportError> {
    if t.shape != shape {
        return Err(ExportError::Shape(format!("{what} is {:?}, expected {shape:?}", t.shape)));
    }
    Ok(())
}

// ---------------------------------------------------------------------- one graph

struct Built {
    graph: Graph,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
}

fn io(name: &str, dtype: DType, shape: &[i64], role: Role, layer: Option<usize>) -> IoSpec {
    IoSpec {
        name: name.into(),
        dtype: dtype.name().into(),
        shape: shape.to_vec(),
        role,
        layer,
        rope: None,
    }
}

/// Everything the graph builders need about a model, resolved once.
struct Plan<'a> {
    cfg: &'a TokConfig,
    names: Names<'a>,
    kinds: Vec<LayerKind>,
    layers: Vec<LayerSpec>,
    rope: Vec<RopeSpec>,
    /// The rope table index of each attention layer.
    rope_of: Vec<Option<usize>>,
    ctx: usize,
    form: WeightForm,
}

impl Plan<'_> {
    fn delta_dims(&self) -> DeltaDims {
        let v2 = &self.cfg.v2;
        DeltaDims {
            d: self.cfg.d_model,
            heads: self.cfg.d_model / v2.dn_head_dim,
            head_dim: v2.dn_head_dim,
            conv: v2.dn_conv,
            chunk: v2.dn_chunk,
            eps: self.cfg.norm_eps,
        }
    }

    fn attn_dims(&self) -> AttnDims {
        AttnDims {
            d: self.cfg.d_model,
            heads: self.cfg.n_heads,
            kv_heads: self.cfg.n_kv_heads,
            head_dim: self.cfg.head_dim,
        }
    }

    fn build(&self, name: &str, t: usize, pool: &mut WeightPool) -> Result<Built, ExportError> {
        let cfg = self.cfg;
        let (d, v) = (cfg.d_model, cfg.vocab_size);
        let ti = t as i64;
        let mut g = Graph::new(name);
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let token = g.input("token", DType::I64, &[1, ti]);
        inputs.push(io("token", DType::I64, &[1, ti], Role::Token, None));
        // RoPE tables of the rope kinds the layers use.
        let mut tables: Vec<Option<(Val, Val)>> = vec![None; self.rope.len()];
        for (r, spec) in self.rope.iter().enumerate() {
            let shape = [1, 1, ti, spec.dim as i64];
            let cos = g.input(&format!("rope_cos_{}", spec.name), DType::F32, &shape);
            let sin = g.input(&format!("rope_sin_{}", spec.name), DType::F32, &shape);
            let mut c =
                io(&format!("rope_cos_{}", spec.name), DType::F32, &shape, Role::RopeCos, None);
            c.rope = Some(r);
            let mut s =
                io(&format!("rope_sin_{}", spec.name), DType::F32, &shape, Role::RopeSin, None);
            s.rope = Some(r);
            inputs.push(c);
            inputs.push(s);
            tables[r] = Some((cos, sin));
        }
        let emb = self.names.get("emb.weight")?;
        expect_shape(&emb, &[v, d], "emb.weight")?;
        let emb_v =
            g.weight(pool, "emb.weight", crate::onnx::HostTensor::f32(&emb.shape, &emb.data));
        let mut x = g.gather(&emb_v, &token, 0);
        let mut cx = Cx { g: &mut g, pool, form: self.form };
        for (i, kind) in self.kinds.iter().enumerate() {
            let base = format!("stack.blocks.{i}");
            let n1 = self.names.get(&format!("{base}.n1.gamma"))?;
            let normed = cx.rms_norm(&format!("{base}.n1.gamma"), &x, &n1, cfg.norm_eps);
            let mixed = match (kind, &self.layers[i]) {
                (LayerKind::DeltaNet, LayerSpec::DeltaNet { .. }) => {
                    let dm = self.delta_dims();
                    let dw = DeltaWeights {
                        proj: self.names.mixer(i, "DeltaNet", "proj.weight")?,
                        conv: self.names.mixer(i, "DeltaNet", "conv")?,
                        a_log: self.names.mixer(i, "DeltaNet", "a_log")?,
                        dt_bias: self.names.mixer(i, "DeltaNet", "dt_bias")?,
                        norm: self.names.mixer(i, "DeltaNet", "norm.gamma")?,
                        o: self.names.mixer(i, "DeltaNet", "o.weight")?,
                    };
                    expect_shape(&dw.proj, &[dm.d, dm.proj_width()], "DeltaNet proj")?;
                    let (c, h, hd) = (dm.channels() as i64, dm.heads as i64, dm.head_dim as i64);
                    let conv_shape = [1, dm.conv as i64 - 1, c];
                    let state_shape = [1, h, hd, hd];
                    let (cn, sn) = (format!("l{i}.conv_in"), format!("l{i}.state_in"));
                    let conv_in = cx.g.input(&cn, DType::F32, &conv_shape);
                    let state_in = cx.g.input(&sn, DType::F32, &state_shape);
                    inputs.push(io(&cn, DType::F32, &conv_shape, Role::ConvIn, Some(i)));
                    inputs.push(io(&sn, DType::F32, &state_shape, Role::StateIn, Some(i)));
                    let key = Names::key(i, "delta");
                    let out = if t == 1 {
                        cx.delta_step(&key, &dm, &dw, &normed, &conv_in, &state_in)?
                    } else {
                        cx.delta_chunked(&key, &dm, &dw, &normed, &conv_in, &state_in, t)?
                    };
                    let (co, so) = (format!("l{i}.conv_out"), format!("l{i}.state_out"));
                    cx.g.output(&co, &out.conv, DType::F32, &conv_shape);
                    cx.g.output(&so, &out.state, DType::F32, &state_shape);
                    outputs.push(io(&co, DType::F32, &conv_shape, Role::ConvOut, Some(i)));
                    outputs.push(io(&so, DType::F32, &state_shape, Role::StateOut, Some(i)));
                    out.y
                }
                (LayerKind::Attention { .. }, LayerSpec::Attention { ring, rope, .. }) => {
                    let dm = self.attn_dims();
                    let qkv = self.names.mixer(i, "Attention", "qkv.weight")?;
                    let o = self.names.mixer(i, "Attention", "o.weight")?;
                    let (kvh, hd) = (dm.kv_heads as i64, dm.head_dim as i64);
                    let ring_shape = [1, kvh, *ring as i64, hd];
                    let bias_shape = [ti, (*ring + t) as i64];
                    let (kn, vn, bn) =
                        (format!("l{i}.k_ring"), format!("l{i}.v_ring"), format!("l{i}.bias"));
                    let rk = cx.g.input(&kn, DType::F32, &ring_shape);
                    let rv = cx.g.input(&vn, DType::F32, &ring_shape);
                    let bias = cx.g.input(&bn, DType::F32, &bias_shape);
                    inputs.push(io(&kn, DType::F32, &ring_shape, Role::RingK, Some(i)));
                    inputs.push(io(&vn, DType::F32, &ring_shape, Role::RingV, Some(i)));
                    inputs.push(io(&bn, DType::F32, &bias_shape, Role::Bias, Some(i)));
                    let (cos, sin) = tables[*rope].clone().expect("the layer's rope table");
                    let out = cx.attention(
                        &Names::key(i, "attn"),
                        &dm,
                        &qkv,
                        &o,
                        &normed,
                        &cos,
                        &sin,
                        &rk,
                        &rv,
                        &bias,
                        t,
                    )?;
                    let new_shape = [1, kvh, ti, hd];
                    let (nk, nv) = (format!("l{i}.k_new"), format!("l{i}.v_new"));
                    cx.g.output(&nk, &out.k_new, DType::F32, &new_shape);
                    cx.g.output(&nv, &out.v_new, DType::F32, &new_shape);
                    outputs.push(io(&nk, DType::F32, &new_shape, Role::NewK, Some(i)));
                    outputs.push(io(&nv, DType::F32, &new_shape, Role::NewV, Some(i)));
                    out.y
                }
                (kind, _) => {
                    return Err(ExportError::Unsupported(format!("layer {i} ({kind:?})")));
                }
            };
            x = cx.g.add(&x, &mixed);
            let n2 = self.names.get(&format!("{base}.n2.gamma"))?;
            let normed = cx.rms_norm(&format!("{base}.n2.gamma"), &x, &n2, cfg.norm_eps);
            let gate_up = self.names.get(&format!("{base}.mlp.gate_up.weight"))?;
            let down = self.names.get(&format!("{base}.mlp.down.weight"))?;
            let mlp = cx.mlp(&format!("{base}.mlp"), &normed, &gate_up, &down, cfg.ff_dim)?;
            x = cx.g.add(&x, &mlp);
        }
        // Only the last position's logits are needed (the norm is per position).
        let last = if t == 1 { x } else { cx.g.slice(&x, 1, ti - 1, ti) };
        let norm = self.names.get("stack.norm.gamma")?;
        let last = cx.rms_norm("stack.norm.gamma", &last, &norm, cfg.norm_eps);
        let emb_t = cx.g.transpose(&emb_v, &[1, 0]);
        let logits = cx.g.matmul(&last, &emb_t);
        g.output("logits", &logits, DType::F32, &[1, 1, v as i64]);
        outputs.insert(0, io("logits", DType::F32, &[1, 1, v as i64], Role::Logits, None));
        g.validate().map_err(|problem| ExportError::Graph { graph: name.into(), problem })?;
        Ok(Built { graph: g, inputs, outputs })
    }
}

// ---------------------------------------------------------------------- the driver

/// Exports the token baseline `config` with `weights` into `out`: one graph per phase, the
/// shared weights file and the manifest. Returns the manifest.
pub fn export(
    config: &Config,
    weights: &Weights,
    out: &Path,
    opts: &ExportOptions,
) -> Result<Manifest, ExportError> {
    let Config::Tok(cfg) = config else {
        let why = describe(config, &opts.buckets)
            .into_iter()
            .find_map(|p| match p.support {
                Support::Unsupported(why) => Some(why),
                Support::Supported => None,
            })
            .unwrap_or_default();
        return Err(ExportError::Unsupported(why));
    };
    if opts.target != Target::Cpu {
        return Err(ExportError::Unsupported(format!(
            "the `{}` target profile: its DQ form and inverse placement wait for that target's \
             NPU smoke test (plan stage 0b)",
            opts.target.name()
        )));
    }
    if let Support::Unsupported(why) = kinds_support(cfg) {
        return Err(ExportError::Unsupported(why));
    }
    if opts.ctx < 2 {
        return Err(ExportError::Config("--ctx must be at least 2".into()));
    }
    let spec = tok_spec(cfg);
    // Rope kinds and layers.
    let mut rope: Vec<RopeSpec> = Vec::new();
    let mut rope_of = Vec::new();
    let mut layers = Vec::new();
    let find_or_add = |rope: &mut Vec<RopeSpec>, name: &str, base: f64, scale: f64| {
        if let Some(i) = rope.iter().position(|r| r.name == name) {
            return i;
        }
        rope.push(RopeSpec { name: name.into(), dim: cfg.head_dim, base, scale });
        rope.len() - 1
    };
    for kind in &spec.kinds {
        match kind {
            LayerKind::DeltaNet => {
                rope_of.push(None);
                layers.push(LayerSpec::DeltaNet {
                    heads: cfg.d_model / cfg.v2.dn_head_dim,
                    head_dim: cfg.v2.dn_head_dim,
                    conv: cfg.v2.dn_conv,
                });
            }
            LayerKind::Attention { window } => {
                let (name, base, scale) = match window {
                    Some(_) => ("local", cfg.rope_base, cfg.rope_scale),
                    None => ("full", cfg.rope_base_full, cfg.rope_scale_full),
                };
                let r = find_or_add(&mut rope, name, base, scale);
                rope_of.push(Some(r));
                // The window counts the position itself: the ring holds the previous ones.
                let ring = window.map_or(opts.ctx - 1, |w| (w - 1).min(opts.ctx - 1));
                layers.push(LayerSpec::Attention {
                    heads: cfg.n_heads,
                    kv_heads: cfg.n_kv_heads,
                    head_dim: cfg.head_dim,
                    ring: ring.max(1),
                    rope: r,
                });
            }
            LayerKind::Retrieval => unreachable!("rejected above"),
        }
    }
    let plan = Plan {
        cfg,
        names: Names { w: weights },
        kinds: spec.kinds.clone(),
        layers: layers.clone(),
        rope: rope.clone(),
        rope_of,
        ctx: opts.ctx,
        form: opts.form,
    };
    let _ = (&plan.rope_of, plan.ctx);
    std::fs::create_dir_all(out).map_err(|e| ExportError::Io(format!("{}: {e}", out.display())))?;
    let mut pool = WeightPool::default();
    let data_file = "weights.bin";
    let mut specs = Vec::new();
    let mut writes: Vec<(String, Vec<u8>)> = Vec::new();
    let mut builds: Vec<(String, Phase, usize)> = vec![("decode".into(), Phase::Decode, 1)];
    for &b in &opts.buckets {
        builds.push((format!("prefill_{b}"), Phase::Prefill, b));
    }
    let mut graphs = Vec::new();
    for (name, phase, tokens) in &builds {
        let built = plan.build(name, *tokens, &mut pool)?;
        let file = format!("{name}.onnx");
        specs.push(GraphSpec {
            name: name.clone(),
            file: file.clone(),
            phase: *phase,
            tokens: *tokens,
            inputs: built.inputs,
            outputs: built.outputs,
        });
        graphs.push((file, built.graph));
    }
    for (file, graph) in &graphs {
        writes.push((file.clone(), graph.bytes(&pool, data_file)));
    }
    let weights_file = (!pool.external_bytes().is_empty()).then(|| data_file.to_owned());
    let manifest = Manifest {
        format: FORMAT.into(),
        model: "tok".into(),
        config_toml: config.to_toml(),
        quant: opts.form.name(),
        target: opts.target.name().into(),
        ctx: opts.ctx,
        vocab: cfg.vocab_size,
        d_model: cfg.d_model,
        unit: 1,
        prefill_buckets: opts.buckets.clone(),
        weights_file: weights_file.clone(),
        rope,
        layers,
        graphs: specs,
    };
    manifest.check().map_err(ExportError::Config)?;
    let write = |name: &str, bytes: &[u8]| {
        std::fs::write(out.join(name), bytes)
            .map_err(|e| ExportError::Io(format!("{}: {e}", out.join(name).display())))
    };
    for (file, bytes) in &writes {
        write(file, bytes)?;
    }
    if weights_file.is_some() {
        write(data_file, pool.external_bytes())?;
    }
    write(FILE_NAME, manifest.to_json().as_bytes())?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::safetensors_bytes;
    use ncm_model::{TokConfig, tiny_tok, tiny_tok_v2, tiny_v2};

    fn tiny() -> TokConfig {
        TokConfig { vocab_size: 272, ..tiny_tok(8) }
    }

    /// A weight file with every tensor the tiny fallback model needs, deterministic values.
    fn tiny_weights(cfg: &TokConfig) -> Weights {
        let spec = tok_spec(cfg);
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut state = 0xDEAD_BEEFu64;
        let mut fill = |shape: &[usize], scale: f32| -> Vec<f32> {
            (0..shape.iter().product::<usize>())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
                })
                .collect()
        };
        let mut add = |name: String, shape: Vec<usize>, scale: f32, ones: bool| {
            let data = if ones { vec![1.0; shape.iter().product()] } else { fill(&shape, scale) };
            tensors.push((name, shape, data));
        };
        let (d, ff) = (cfg.d_model, cfg.ff_dim);
        add("emb.weight".into(), vec![cfg.vocab_size, d], 0.05, false);
        add("stack.norm.gamma".into(), vec![d], 0.0, true);
        for (i, kind) in spec.kinds.iter().enumerate() {
            let b = format!("stack.blocks.{i}");
            add(format!("{b}.n1.gamma"), vec![d], 0.0, true);
            add(format!("{b}.n2.gamma"), vec![d], 0.0, true);
            add(format!("{b}.mlp.gate_up.weight"), vec![d, 2 * ff], 0.05, false);
            add(format!("{b}.mlp.down.weight"), vec![ff, d], 0.05, false);
            if let LayerKind::Attention { .. } = kind {
                let (h, kvh, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
                add(
                    format!("{b}.attn.Attention.qkv.weight"),
                    vec![d, (h + 2 * kvh) * hd],
                    0.05,
                    false,
                );
                add(format!("{b}.attn.Attention.o.weight"), vec![h * hd, d], 0.05, false);
            }
        }
        let refs: Vec<(&str, &[usize], &[f32])> =
            tensors.iter().map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice())).collect();
        Weights::from_bytes(safetensors_bytes(&refs, &[]), "synthetic").unwrap()
    }

    #[test]
    fn the_fallback_token_model_exports_decode_and_prefill_graphs() {
        let cfg = tiny();
        let weights = tiny_weights(&cfg);
        let dir = tempfile::tempdir().unwrap();
        let opts = ExportOptions {
            ctx: 32,
            buckets: vec![8, 16],
            form: WeightForm::F32,
            target: Target::Cpu,
        };
        let m = export(&Config::Tok(cfg.clone()), &weights, dir.path(), &opts).unwrap();
        assert_eq!(m.graphs.len(), 3);
        for g in &m.graphs {
            let bytes = std::fs::read(dir.path().join(&g.file)).unwrap();
            assert!(!bytes.is_empty());
        }
        assert!(dir.path().join(FILE_NAME).is_file());
        let back =
            Manifest::from_json(&std::fs::read_to_string(dir.path().join(FILE_NAME)).unwrap())
                .unwrap();
        assert_eq!(back, m);
        let decode = m.graph(Phase::Decode, 1).unwrap();
        assert_eq!(decode.outputs[0].role, Role::Logits);
        assert!(decode.inputs.iter().any(|i| i.role == Role::RingK));
    }

    #[test]
    fn sliding_layers_ring_the_window_minus_one_and_full_layers_the_context() {
        let cfg = tiny();
        let weights = tiny_weights(&cfg);
        let dir = tempfile::tempdir().unwrap();
        let opts =
            ExportOptions { ctx: 64, buckets: vec![], form: WeightForm::F32, target: Target::Cpu };
        let m = export(&Config::Tok(cfg.clone()), &weights, dir.path(), &opts).unwrap();
        let rings: Vec<usize> = m
            .layers
            .iter()
            .filter_map(|l| match l {
                LayerSpec::Attention { ring, .. } => Some(*ring),
                _ => None,
            })
            .collect();
        assert!(rings.contains(&(cfg.window - 1)) || rings.contains(&63), "{rings:?}");
        assert!(rings.iter().all(|&r| r <= 63));
    }

    #[test]
    fn a_quantised_export_stores_codes_and_scales() {
        let cfg = tiny();
        let weights = tiny_weights(&cfg);
        let dir = tempfile::tempdir().unwrap();
        let opts =
            ExportOptions { ctx: 16, buckets: vec![], form: WeightForm::Q8, target: Target::Cpu };
        let f32_dir = tempfile::tempdir().unwrap();
        let f32_opts = ExportOptions { form: WeightForm::F32, ..opts.clone() };
        export(&Config::Tok(cfg.clone()), &weights, dir.path(), &opts).unwrap();
        export(&Config::Tok(cfg), &weights, f32_dir.path(), &f32_opts).unwrap();
        let q = std::fs::metadata(dir.path().join("weights.bin")).unwrap().len();
        let f = std::fs::metadata(f32_dir.path().join("weights.bin")).unwrap().len();
        assert!(q < f, "int8 weights take less room than f32 ({q} vs {f})");
    }

    #[test]
    fn the_v2_design_names_the_retrieval_layer_it_cannot_export_yet() {
        let cfg = TokConfig { vocab_size: 272, ..tiny_tok_v2(tiny_v2(4, 2, 8)) };
        let plans = describe(&Config::Tok(cfg.clone()), &[8]);
        assert!(matches!(&plans[0].support, Support::Unsupported(w) if w.contains("retrieval")));
        let weights = Weights::from_bytes(safetensors_bytes(&[], &[]), "empty").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let opts =
            ExportOptions { ctx: 16, buckets: vec![], form: WeightForm::F32, target: Target::Cpu };
        let err = export(&Config::Tok(cfg), &weights, dir.path(), &opts).unwrap_err();
        assert!(err.to_string().contains("sparse-retrieval"), "{err}");
    }

    #[test]
    fn npu_targets_wait_for_their_smoke_tests() {
        let cfg = tiny();
        let weights = tiny_weights(&cfg);
        let dir = tempfile::tempdir().unwrap();
        let opts = ExportOptions {
            ctx: 16,
            buckets: vec![],
            form: WeightForm::F32,
            target: Target::Intel,
        };
        let err = export(&Config::Tok(cfg), &weights, dir.path(), &opts).unwrap_err();
        assert!(err.to_string().contains("stage 0b"), "{err}");
    }

    #[test]
    fn describe_lists_the_decode_graph_and_one_per_bucket() {
        let plans = describe(&Config::Tok(tiny()), &[8, 16]);
        let names: Vec<&str> = plans.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["decode", "prefill_8", "prefill_16"]);
        assert!(plans.iter().all(|p| p.support == Support::Supported));
    }
}
