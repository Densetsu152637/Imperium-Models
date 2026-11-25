//! The packed weight format and its host-side quantiser and dequantiser.
//!
//! A quantised `[d_out, d_in]` weight (output rows, the layout a GEMV streams) is stored as:
//!
//! * `qweight` `[d_out, words]`, i32: each row's codes packed little end first, `32 / bits` codes
//!   per word (code `j` of a word in bits `j * bits ..`). Rows are padded with zero codes to
//!   `k_pad` columns, a multiple of [`Format::align`], so every word lies inside one group.
//! * `scales` `[d_out, groups]`, f32: one scale per row (q8) or per `group` columns (q4).
//!
//! A code `u` decodes to `(u - offset) * scale` with `offset = 2^(bits - 1)`: q8 is symmetric
//! int8 (codes 1..=255, absmax / 127 scales); q4 uses all 16 levels with the scale set so that
//! the group's largest-magnitude value maps to -8 (as llama.cpp's Q4_0 does).

/// Bits per code and columns per scale of a quantised weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub bits: u32,
    /// Columns sharing one scale; `None` for one scale per row.
    pub group: Option<usize>,
}

/// Padding unit of a per-row (q8) format's columns: 16 words of 4 codes.
const ROW_ALIGN: usize = 64;

impl Format {
    pub const fn q8() -> Self {
        Self { bits: 8, group: None }
    }

    pub const fn q4(group: usize) -> Self {
        Self { bits: 4, group: Some(group) }
    }

    /// Codes per i32 word.
    pub const fn per_word(&self) -> usize {
        32 / self.bits as usize
    }

    pub const fn offset(&self) -> i32 {
        1 << (self.bits - 1)
    }

    /// Columns are padded to a multiple of this.
    pub fn align(&self) -> usize {
        self.group.unwrap_or(ROW_ALIGN)
    }

    /// Padded row length for `d_in` columns.
    pub fn k_pad(&self, d_in: usize) -> usize {
        d_in.div_ceil(self.align()) * self.align()
    }

    pub fn words(&self, d_in: usize) -> usize {
        self.k_pad(d_in) / self.per_word()
    }

    /// Columns per scale for `d_in` columns.
    pub fn group_len(&self, d_in: usize) -> usize {
        self.group.unwrap_or_else(|| self.k_pad(d_in))
    }

    pub fn groups(&self, d_in: usize) -> usize {
        self.k_pad(d_in) / self.group_len(d_in)
    }
}

/// A quantised weight on the host.
#[derive(Debug, Clone, PartialEq)]
pub struct Packed {
    pub format: Format,
    pub d_out: usize,
    pub d_in: usize,
    /// `[d_out, words]`.
    pub words: Vec<i32>,
    /// `[d_out, groups]`.
    pub scales: Vec<f32>,
}

/// The scale and inverse of one group, from its values.
fn group_scale(format: Format, values: &[f32]) -> (f32, f32) {
    let extreme = values.iter().copied().fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    let scale = if format.bits == 8 { extreme.abs() / 127.0 } else { extreme / -8.0 };
    let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    (scale, inverse)
}

/// Quantises `w`, row-major `[d_out, d_in]`.
pub fn quantize(w: &[f32], d_out: usize, d_in: usize, format: Format) -> Packed {
    assert_eq!(w.len(), d_out * d_in, "weight is not [{d_out}, {d_in}]");
    assert!(matches!(format.bits, 4 | 8), "only 4- and 8-bit codes are supported");
    if let Some(g) = format.group {
        assert!(g > 0 && g.is_multiple_of(32), "the group size must be a positive multiple of 32");
    }
    let (per, offset) = (format.per_word(), format.offset());
    let (max_code, words, groups, len) =
        ((1 << format.bits) - 1, format.words(d_in), format.groups(d_in), format.group_len(d_in));
    let mut packed = vec![0i32; d_out * words];
    let mut scales = vec![0f32; d_out * groups];
    for r in 0..d_out {
        let row = &w[r * d_in..(r + 1) * d_in];
        for g in 0..groups {
            let (start, end) = ((g * len).min(d_in), ((g + 1) * len).min(d_in));
            let (scale, inverse) = group_scale(format, &row[start..end]);
            scales[r * groups + g] = scale;
            for c in g * len..(g + 1) * len {
                let code = match row.get(c) {
                    Some(v) => ((v * inverse).round() as i32 + offset).clamp(0, max_code),
                    None => offset, // padding
                };
                let (word, slot) = (c / per, c % per);
                packed[r * words + word] |= code << (slot as u32 * format.bits);
            }
        }
    }
    Packed { format, d_out, d_in, words: packed, scales }
}

impl Packed {
    /// The decoded weight, row-major `[d_out, d_in]`.
    pub fn dequantize(&self) -> Vec<f32> {
        let f = self.format;
        let (per, offset, mask) = (f.per_word(), f.offset(), (1i32 << f.bits) - 1);
        let (words, groups, len) =
            (f.words(self.d_in), f.groups(self.d_in), f.group_len(self.d_in));
        let mut out = Vec::with_capacity(self.d_out * self.d_in);
        for r in 0..self.d_out {
            for c in 0..self.d_in {
                let word = self.words[r * words + c / per];
                let code = (word >> ((c % per) as u32 * f.bits)) & mask;
                out.push((code - offset) as f32 * self.scales[r * groups + c / len]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(d_out: usize, d_in: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..d_out * d_in)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    /// Largest decode error over the largest magnitude of the weight's rows.
    fn relative_error(format: Format, d_out: usize, d_in: usize) -> f32 {
        let w = weights(d_out, d_in);
        let back = quantize(&w, d_out, d_in, format).dequantize();
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        w.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale
    }

    #[test]
    fn codes_round_to_half_a_step() {
        // Half a step of the absmax scale for q8 (1/254). q4 maps the extreme to -8, so values
        // of the other sign near it clamp at +7: up to one step (1/8) there, half a step else.
        assert!(relative_error(Format::q8(), 7, 200) <= 0.5 / 127.0 + 1e-6);
        assert!(relative_error(Format::q4(64), 7, 200) <= 1.0 / 8.0 + 1e-6);
        assert!(relative_error(Format::q4(128), 3, 1700) <= 1.0 / 8.0 + 1e-6);
    }

    #[test]
    fn ragged_rows_are_padded_with_zero_codes() {
        let format = Format::q4(64);
        let p = quantize(&weights(2, 70), 2, 70, format);
        assert_eq!((format.k_pad(70), format.words(70), format.groups(70)), (128, 16, 2));
        assert_eq!((p.words.len(), p.scales.len()), (32, 4));
        // Columns 70..128 hold the zero code 8 in every slot.
        let last = p.words[15] as u32;
        assert_eq!(last, 0x8888_8888);
        assert_eq!(p.dequantize().len(), 140);
    }

    #[test]
    fn exact_values_survive() {
        // Multiples of the scale decode exactly.
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
        let p = quantize(&w, 1, 64, Format::q8());
        let back = p.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() <= 8.0 / 127.0 * 0.5 + 1e-6, "{a} vs {b}");
        }
        let w = vec![-8.0, -4.0, 0.0, 2.0, 7.0, 1.0, -1.0, 3.0];
        let mut row = w.clone();
        row.resize(64, 0.0);
        let back = quantize(&row, 1, 64, Format::q4(64)).dequantize();
        assert_eq!(&back[..8], &w[..]);
    }

    #[test]
    fn a_zero_row_decodes_to_zero() {
        let p = quantize(&[0.0; 96], 1, 96, Format::q4(32));
        assert!(p.dequantize().iter().all(|v| *v == 0.0));
        assert!(p.scales.iter().all(|s| *s == 0.0));
    }
}
