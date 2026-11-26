//! Standard QDQ weights (`docs/plans/npu.md`, decision 6): a dense `[d_in, d_out]` weight
//! becomes integer codes plus scales feeding a `DequantizeLinear`, never a contrib op.
//!
//! - **int8**: one scale per output channel (`axis = 1`), codes in -127..=127.
//! - **int4**: opset-21 int4 codes, symmetric (no zero point), one scale per `group` input
//!   values and output channel (`axis = 0`, `block_size = group`); symmetric is what the
//!   OpenVINO NPU path requires.
//!
//! The codes and scales come from the repository's own host quantiser
//! (`ncm_model::quant::format::quantize`), so an exported weight dequantises to exactly the
//! values the Burn backend computes with `--quant q8` / `q4` at the same group size.

use ncm_model::quant::Format;
use ncm_model::quant::format::quantize;

use crate::graph::{Graph, Val, WeightPool};
use crate::onnx::{Attr, HostTensor};

/// How a linear weight is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightForm {
    F32,
    /// Per-channel int8.
    Q8,
    /// Block-wise symmetric int4 with this group size (a multiple of 32).
    Q4 {
        group: usize,
    },
}

impl WeightForm {
    pub fn name(self) -> String {
        match self {
            WeightForm::F32 => "f32".into(),
            WeightForm::Q8 => "q8".into(),
            WeightForm::Q4 { group } => format!("q4-g{group}"),
        }
    }

    /// Parses `none`/`f32`, `q8`, `q4` (with `group`).
    pub fn parse(mode: &str, group: usize) -> Result<Self, String> {
        match mode {
            "none" | "f32" => Ok(WeightForm::F32),
            "q8" => Ok(WeightForm::Q8),
            "q4" if group > 0 && group.is_multiple_of(32) => Ok(WeightForm::Q4 { group }),
            "q4" => Err(format!("q4 group size {group} is not a multiple of 32")),
            other => Err(format!("unknown quantisation `{other}` (none, q8, q4)")),
        }
    }
}

/// The QDQ tensors of one weight.
#[derive(Debug, PartialEq)]
pub struct Qdq {
    /// Codes, `[d_in, d_out]`.
    pub codes: HostTensor,
    /// `[d_out]` for int8, `[d_in / group, d_out]` for int4.
    pub scales: HostTensor,
    pub axis: i64,
    /// 0 for per-channel.
    pub block_size: i64,
}

/// Quantises the dense weight `w` (row-major `[d_in, d_out]`) as `form`; None for `F32`.
pub fn qdq(w: &[f32], d_in: usize, d_out: usize, form: WeightForm) -> Result<Option<Qdq>, String> {
    assert_eq!(w.len(), d_in * d_out, "weight is not [{d_in}, {d_out}]");
    let format = match form {
        WeightForm::F32 => return Ok(None),
        WeightForm::Q8 => Format::q8(),
        WeightForm::Q4 { group } => {
            if !d_in.is_multiple_of(group) {
                return Err(format!(
                    "int4 export needs the input width {d_in} to be a multiple of the group {group}"
                ));
            }
            Format::q4(group)
        }
    };
    // The host quantiser wants rows of the output channel: `[d_out, d_in]`.
    let mut rows = vec![0f32; d_in * d_out];
    for i in 0..d_in {
        for o in 0..d_out {
            rows[o * d_in + i] = w[i * d_out + o];
        }
    }
    let packed = quantize(&rows, d_out, d_in, format);
    let (per, offset, mask) = (format.per_word(), format.offset(), (1i32 << format.bits) - 1);
    let words = format.words(d_in);
    let groups = format.groups(d_in);
    let mut codes = vec![0i8; d_in * d_out];
    for o in 0..d_out {
        for i in 0..d_in {
            let word = packed.words[o * words + i / per];
            let code = (word >> ((i % per) as u32 * format.bits)) & mask;
            codes[i * d_out + o] = (code - offset) as i8;
        }
    }
    Ok(Some(match form {
        WeightForm::Q8 => {
            // One group per row: the scale per output channel.
            debug_assert_eq!(groups, 1);
            Qdq {
                codes: HostTensor::i8(&[d_in, d_out], &codes),
                scales: HostTensor::f32(&[d_out], &packed.scales),
                axis: 1,
                block_size: 0,
            }
        }
        WeightForm::Q4 { group } => {
            let mut scales = vec![0f32; groups * d_out];
            for o in 0..d_out {
                for b in 0..groups {
                    scales[b * d_out + o] = packed.scales[o * groups + b];
                }
            }
            Qdq {
                codes: HostTensor::i4(&[d_in, d_out], &codes),
                scales: HostTensor::f32(&[groups, d_out], &scales),
                axis: 0,
                block_size: group as i64,
            }
        }
        WeightForm::F32 => unreachable!("returned above"),
    }))
}

/// The weight a graph multiplies by: a plain initializer, or codes and scales through a
/// `DequantizeLinear`. `key` names the weight in the pool.
pub fn weight_val(
    g: &mut Graph,
    pool: &mut WeightPool,
    key: &str,
    w: &[f32],
    d_in: usize,
    d_out: usize,
    form: WeightForm,
) -> Result<Val, String> {
    match qdq(w, d_in, d_out, form)? {
        None => Ok(g.weight(pool, key, HostTensor::f32(&[d_in, d_out], w))),
        Some(q) => {
            let codes = g.weight(pool, &format!("{key}.q"), q.codes);
            let scales = g.weight(pool, &format!("{key}.scale"), q.scales);
            let mut attrs = vec![("axis", Attr::Int(q.axis))];
            if q.block_size > 0 {
                attrs.push(("block_size", Attr::Int(q.block_size)));
            }
            Ok(g.op("DequantizeLinear", &[&codes, &scales], &attrs))
        }
    }
}

/// What a QDQ weight dequantises to, row-major `[d_in, d_out]` (the host mirror of the
/// `DequantizeLinear`, for tests and tolerances).
pub fn dequantize(q: &Qdq, d_in: usize, d_out: usize) -> Vec<f32> {
    let code = |i: usize, o: usize| -> f32 {
        let n = i * d_out + o;
        match q.codes.dtype {
            crate::onnx::DType::I8 => q.codes.bytes[n] as i8 as f32,
            _ => {
                let byte = q.codes.bytes[n / 2];
                let nibble = if n.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 };
                // Sign-extend the nibble.
                (((nibble << 4) as i8) >> 4) as f32
            }
        }
    };
    let scale = |i: usize, o: usize| -> f32 {
        let at = if q.block_size > 0 { (i / q.block_size as usize) * d_out + o } else { o };
        f32::from_le_bytes(q.scales.bytes[at * 4..at * 4 + 4].try_into().unwrap())
    };
    let mut out = vec![0f32; d_in * d_out];
    for i in 0..d_in {
        for o in 0..d_out {
            out[i * d_out + o] = code(i, o) * scale(i, o);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(d_in: usize, d_out: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..d_in * d_out)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    /// The QDQ weight equals the repository quantiser's own dequantisation, transposed.
    fn matches_the_host_quantiser(form: WeightForm, format: Format, d_in: usize, d_out: usize) {
        let w = weights(d_in, d_out);
        let q = qdq(&w, d_in, d_out, form).unwrap().expect("quantised");
        let got = dequantize(&q, d_in, d_out);
        let mut rows = vec![0f32; d_in * d_out];
        for i in 0..d_in {
            for o in 0..d_out {
                rows[o * d_in + i] = w[i * d_out + o];
            }
        }
        let want = quantize(&rows, d_out, d_in, format).dequantize();
        for i in 0..d_in {
            for o in 0..d_out {
                let (a, b) = (got[i * d_out + o], want[o * d_in + i]);
                assert!((a - b).abs() <= 1e-7, "[{i}, {o}]: {a} vs {b}");
            }
        }
    }

    #[test]
    fn int8_is_per_channel_and_matches_the_host_quantiser() {
        matches_the_host_quantiser(WeightForm::Q8, Format::q8(), 96, 40);
        let q = qdq(&weights(96, 40), 96, 40, WeightForm::Q8).unwrap().unwrap();
        assert_eq!((q.axis, q.block_size), (1, 0));
        assert_eq!(q.scales.dims, vec![40]);
        assert_eq!(q.codes.dims, vec![96, 40]);
    }

    #[test]
    fn int4_is_blockwise_symmetric_and_matches_the_host_quantiser() {
        matches_the_host_quantiser(WeightForm::Q4 { group: 64 }, Format::q4(64), 192, 24);
        let q = qdq(&weights(192, 24), 192, 24, WeightForm::Q4 { group: 64 }).unwrap().unwrap();
        assert_eq!((q.axis, q.block_size), (0, 64));
        assert_eq!(q.scales.dims, vec![3, 24]);
        assert_eq!(q.codes.dtype as i32, 22);
        assert_eq!(q.codes.bytes.len(), 192 * 24 / 2);
    }

    #[test]
    fn int4_refuses_a_ragged_input_width() {
        let err = qdq(&weights(70, 8), 70, 8, WeightForm::Q4 { group: 32 }).unwrap_err();
        assert!(err.contains("multiple of the group"), "{err}");
    }

    #[test]
    fn f32_stays_unquantised() {
        assert!(qdq(&weights(4, 4), 4, 4, WeightForm::F32).unwrap().is_none());
    }

    #[test]
    fn the_quant_mode_parses() {
        assert_eq!(WeightForm::parse("q8", 0).unwrap(), WeightForm::Q8);
        assert_eq!(WeightForm::parse("q4", 128).unwrap(), WeightForm::Q4 { group: 128 });
        assert!(WeightForm::parse("q4", 48).is_err());
        assert_eq!(WeightForm::parse("none", 0).unwrap(), WeightForm::F32);
    }

    #[test]
    fn a_quantised_weight_is_a_dequantize_node_over_pool_tensors() {
        let mut pool = WeightPool::default();
        let mut g = Graph::new("g");
        let x = g.input("x", crate::onnx::DType::F32, &[1, 64]);
        let w =
            weight_val(&mut g, &mut pool, "lin", &weights(64, 32), 64, 32, WeightForm::Q8).unwrap();
        let y = g.matmul(&x, &w);
        g.output("y", &y, crate::onnx::DType::F32, &[1, 32]);
        g.validate().unwrap();
        let ops: Vec<&str> = g.op_types().collect();
        assert!(ops.contains(&"DequantizeLinear") && ops.contains(&"MatMul"));
        assert!(pool.tensor("lin.q").is_some() && pool.tensor("lin.scale").is_some());
    }
}
