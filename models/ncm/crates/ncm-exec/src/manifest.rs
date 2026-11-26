//! The manifest `ncm-export` writes beside its ONNX graphs and `ncm-ort` reads: the config
//! facts a backend needs without Burn (layer shapes, rope tables, ring sizes), the graphs, and
//! the name, type, shape and role of every graph input and output.
//!
//! A graph input or output has a *role* (`token`, `ring_k`, `state_in`, ...) and, for per-layer
//! ones, the layer it belongs to, so a backend binds buffers by role rather than by parsing
//! names.

use serde::{Deserialize, Serialize};

/// The manifest's format tag.
pub const FORMAT: &str = "ncm-onnx-v1";
/// The manifest's file name in an export directory.
pub const FILE_NAME: &str = "manifest.json";

/// What a graph input or output carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Token ids `[1, T]` (i64).
    Token,
    /// The activation entering a graph that starts mid-model (split graphs).
    Hidden,
    /// Logits of the last position `[1, 1, V]`.
    Logits,
    /// RoPE tables of a layer's rope kind `[1, 1, T, D]` (see [`IoSpec::rope`]).
    RopeCos,
    RopeSin,
    /// An attention layer's additive mask `[T, S + T]`.
    Bias,
    /// An attention layer's ring of keys / values `[1, KVH, S, D]`.
    RingK,
    RingV,
    /// A retrieval layer's latent cache `[1, S, latent + rope_dim]` (input) and the entries of
    /// the positions just processed `[1, T, latent + rope_dim]` (output), and the normed
    /// indexer keys of those positions `[1, T, index_dim]` (output).
    Latent,
    NewLatent,
    NewIndexKey,
    /// The split sparse path (UNVERIFIED), per retrieval layer `i` at a segment boundary: the
    /// absorbed queries `[1, H, 1, latent + rope_dim]`, the sigmoid gate `[1, 1, H * v_dim]`,
    /// the indexer queries `[1, 1, index_heads * index_dim]` and head weights
    /// `[1, 1, index_heads]` (outputs of the segment ending at layer `i`), and the gathered
    /// entries `[1, topk * block + local, latent + rope_dim]` (input of the next segment, with
    /// its `Bias` `[1, topk * block + local]`).
    QAll,
    GateVal,
    IndexQ,
    IndexW,
    Gathered,
    /// The new keys / values of the positions just processed `[1, KVH, T, D]`, for the host to
    /// write into the ring.
    NewK,
    NewV,
    /// A DeltaNet layer's short-convolution inputs `[1, conv - 1, C]`, in and out.
    ConvIn,
    ConvOut,
    /// A DeltaNet layer's state `[1, H, dk, dv]` (f32, host-owned), in and out.
    StateIn,
    StateOut,
    /// A triangular matrix to invert / its inverse (the split f32 subgraph).
    InverseIn,
    InverseOut,
}

/// One graph input or output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IoSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub role: Role,
    /// The layer it belongs to (per-layer roles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    /// The rope table it carries (`RopeCos` / `RopeSin`): an index into [`Manifest::rope`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope: Option<usize>,
}

/// The phase a graph serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// One token (token baseline decode step).
    Decode,
    /// A prefill bucket of `tokens` tokens.
    Prefill,
    /// The f32 triangular inverse of the DeltaNet chunk (split out on NPU targets).
    DeltaInverse,
    /// One decode step cut at every retrieval layer (UNVERIFIED): segment `k` ends at the
    /// layer's projection and the next one starts at its attention, with the host's top-k and
    /// gather ([`crate::retrieval`]) between. `segments = retrieval layers + 1` graphs, in
    /// [`GraphSpec::segment`] order.
    DecodeSplit,
}

/// One exported graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub name: String,
    pub file: String,
    pub phase: Phase,
    /// Tokens the graph processes (1 for decode, the bucket for prefill).
    pub tokens: usize,
    /// Which segment of a [`Phase::DecodeSplit`] step this is (0 otherwise).
    #[serde(default)]
    pub segment: usize,
    pub inputs: Vec<IoSpec>,
    pub outputs: Vec<IoSpec>,
}

/// A RoPE table family: layers of one rope kind share its base and scale.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    pub name: String,
    pub dim: usize,
    pub base: f64,
    pub scale: f64,
    /// YaRN (the retrieval key's table); a factor of 1 is plain RoPE.
    #[serde(default = "one")]
    pub yarn_factor: f64,
    #[serde(default)]
    pub yarn_original: usize,
}

fn one() -> f64 {
    1.0
}

/// One layer of the stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerSpec {
    DeltaNet {
        heads: usize,
        head_dim: usize,
        conv: usize,
    },
    Attention {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        /// Previous positions the ring holds (the window minus the position itself, or the
        /// context minus one for a full layer).
        ring: usize,
        /// The layer's rope table, an index into [`Manifest::rope`].
        rope: usize,
    },
    /// A sparse-retrieval layer exported as dense attention over its whole latent cache
    /// (UNVERIFIED): exact while the sparse selection would pick every block, i.e. for
    /// positions below `dense_exact_until`.
    Retrieval {
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        latent: usize,
        index_heads: usize,
        index_dim: usize,
        block: usize,
        topk: usize,
        local: usize,
        /// Positions the latent cache holds before the new ones.
        cache: usize,
        rope: usize,
        dense_exact_until: usize,
    },
}

/// An export directory's description.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    /// `tok`.
    pub model: String,
    /// The model config as TOML (for `ModelInfo` and the parameter report).
    pub config_toml: String,
    /// `f32`, `q8` or `q4-g<group>`.
    pub quant: String,
    /// `cpu`, `intel`, `qualcomm` or `amd`.
    pub target: String,
    /// The context the rings were sized for.
    pub ctx: usize,
    pub vocab: usize,
    pub d_model: usize,
    /// Tokens per decoder pass.
    pub unit: usize,
    pub prefill_buckets: Vec<usize>,
    /// The external weights file shared by every graph, if any weight was big enough.
    pub weights_file: Option<String>,
    pub rope: Vec<RopeSpec>,
    pub layers: Vec<LayerSpec>,
    pub graphs: Vec<GraphSpec>,
}

impl Manifest {
    pub fn graph(&self, phase: Phase, tokens: usize) -> Option<&GraphSpec> {
        self.graphs.iter().find(|g| g.phase == phase && g.tokens == tokens)
    }

    /// The segments of the split decode step, in order (empty without one).
    pub fn split_segments(&self) -> Vec<&GraphSpec> {
        let mut v: Vec<&GraphSpec> =
            self.graphs.iter().filter(|g| g.phase == Phase::DecodeSplit).collect();
        v.sort_by_key(|g| g.segment);
        v
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a manifest serialises")
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if m.format != FORMAT {
            return Err(format!("the manifest is `{}`, this build reads `{FORMAT}`", m.format));
        }
        m.check()?;
        Ok(m)
    }

    /// Cross-checks the manifest: layer indices and rope indices are in range, every graph has
    /// a token input and the per-layer ring and state inputs its layers need.
    pub fn check(&self) -> Result<(), String> {
        let retrievals =
            self.layers.iter().filter(|l| matches!(l, LayerSpec::Retrieval { .. })).count();
        let split = self.split_segments();
        if !split.is_empty() && split.len() != retrievals + 1 {
            return Err(format!(
                "{} split segments for {retrievals} retrieval layers (expected one more)",
                split.len()
            ));
        }
        for g in self.graphs.iter().filter(|g| g.phase != Phase::DeltaInverse) {
            for io in g.inputs.iter().chain(&g.outputs) {
                if io.layer.is_some_and(|l| l >= self.layers.len()) {
                    return Err(format!(
                        "{}: `{}` names a layer that does not exist",
                        g.name, io.name
                    ));
                }
                if io.rope.is_some_and(|r| r >= self.rope.len()) {
                    return Err(format!(
                        "{}: `{}` names a rope table that does not exist",
                        g.name, io.name
                    ));
                }
            }
            if g.phase == Phase::DecodeSplit {
                // Segments carry only the layers between two cuts: the first has the token, the
                // last the logits, the per-layer checks below do not apply.
                let first = g.segment == 0;
                let last = g.segment == retrievals;
                if first && !g.inputs.iter().any(|i| i.role == Role::Token) {
                    return Err(format!("{}: no token input", g.name));
                }
                if last && !g.outputs.iter().any(|o| o.role == Role::Logits) {
                    return Err(format!("{}: no logits output", g.name));
                }
                continue;
            }
            if !g.inputs.iter().any(|i| i.role == Role::Token) {
                return Err(format!("{}: no token input", g.name));
            }
            if !g.outputs.iter().any(|o| o.role == Role::Logits) {
                return Err(format!("{}: no logits output", g.name));
            }
            for (i, layer) in self.layers.iter().enumerate() {
                let has = |role: Role, io: &[IoSpec]| {
                    io.iter().any(|s| s.role == role && s.layer == Some(i))
                };
                let needs: &[(Role, bool)] = match layer {
                    LayerSpec::DeltaNet { .. } => &[
                        (Role::ConvIn, true),
                        (Role::StateIn, true),
                        (Role::ConvOut, false),
                        (Role::StateOut, false),
                    ],
                    LayerSpec::Retrieval { .. } => &[
                        (Role::Latent, true),
                        (Role::Bias, true),
                        (Role::NewLatent, false),
                        (Role::NewIndexKey, false),
                    ],
                    LayerSpec::Attention { .. } => &[
                        (Role::RingK, true),
                        (Role::RingV, true),
                        (Role::Bias, true),
                        (Role::NewK, false),
                        (Role::NewV, false),
                    ],
                };
                for &(role, input) in needs {
                    let side = if input { &g.inputs } else { &g.outputs };
                    if !has(role, side) {
                        return Err(format!("{}: layer {i} has no {role:?}", g.name));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(name: &str, role: Role, layer: Option<usize>) -> IoSpec {
        IoSpec { name: name.into(), dtype: "f32".into(), shape: vec![1], role, layer, rope: None }
    }

    fn manifest() -> Manifest {
        Manifest {
            format: FORMAT.into(),
            model: "tok".into(),
            config_toml: String::new(),
            quant: "f32".into(),
            target: "cpu".into(),
            ctx: 64,
            vocab: 272,
            d_model: 32,
            unit: 1,
            prefill_buckets: vec![],
            weights_file: None,
            rope: vec![],
            layers: vec![LayerSpec::DeltaNet { heads: 2, head_dim: 16, conv: 4 }],
            graphs: vec![GraphSpec {
                name: "decode".into(),
                file: "decode.onnx".into(),
                phase: Phase::Decode,
                tokens: 1,
                segment: 0,
                inputs: vec![
                    io("token", Role::Token, None),
                    io("l0.conv_in", Role::ConvIn, Some(0)),
                    io("l0.state_in", Role::StateIn, Some(0)),
                ],
                outputs: vec![
                    io("logits", Role::Logits, None),
                    io("l0.conv_out", Role::ConvOut, Some(0)),
                    io("l0.state_out", Role::StateOut, Some(0)),
                ],
            }],
        }
    }

    #[test]
    fn round_trips_through_json() {
        let m = manifest();
        let back = Manifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
        assert!(back.graph(Phase::Decode, 1).is_some());
        assert!(back.graph(Phase::Prefill, 128).is_none());
    }

    #[test]
    fn a_missing_state_is_caught() {
        let mut m = manifest();
        m.graphs[0].inputs.pop();
        assert!(m.check().unwrap_err().contains("StateIn"));
    }

    #[test]
    fn a_wrong_format_is_refused() {
        let mut m = manifest();
        m.format = "other".into();
        assert!(Manifest::from_json(&m.to_json()).unwrap_err().contains("this build reads"));
    }
}
