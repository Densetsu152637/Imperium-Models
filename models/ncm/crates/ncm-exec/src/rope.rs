//! RoPE tables on the host, bit for bit what `ncm-model` computes with YaRN off: the `cos` and
//! `sin` of `pos / scale * base^(-2i / dim)`, each value repeated across both halves of the head
//! dimension (the layout the exported graphs' rotation matrix expects).

/// Writes row `pos`'s tables into `cos` and `sin` (each `dim` long).
pub fn rope_row(pos: usize, dim: usize, base: f64, scale: f64, cos: &mut [f32], sin: &mut [f32]) {
    let half = dim / 2;
    let p = pos as f64 / scale;
    for i in 0..half {
        let angle = p * (1.0 / base.powf((2 * i) as f64 / dim as f64));
        let (c, s) = (angle.cos() as f32, angle.sin() as f32);
        (cos[i], cos[i + half], sin[i], sin[i + half]) = (c, c, s, s);
    }
}

/// The tables of positions `start..start + len`, row-major `[len, dim]` each.
pub fn rope_rows(
    start: usize,
    len: usize,
    dim: usize,
    base: f64,
    scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; len * dim];
    let mut sin = vec![0f32; len * dim];
    for (row, (c, s)) in cos.chunks_mut(dim).zip(sin.chunks_mut(dim)).enumerate() {
        rope_row(start + row, dim, base, scale, c, s);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let (cos, sin) = rope_rows(0, 1, 8, 10_000.0, 1.0);
        assert!(cos.iter().all(|&c| c == 1.0));
        assert!(sin.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn halves_repeat_and_the_first_frequency_is_one_radian_per_position() {
        let (cos, sin) = rope_rows(3, 1, 8, 10_000.0, 1.0);
        assert_eq!(cos[0], cos[4]);
        assert_eq!(sin[1], sin[5]);
        assert!((cos[0] - 3f32.cos()).abs() < 1e-6);
        assert!((sin[0] - 3f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn scale_divides_the_position() {
        let (a, _) = rope_rows(4, 1, 8, 10_000.0, 2.0);
        let (b, _) = rope_rows(2, 1, 8, 10_000.0, 1.0);
        assert_eq!(a, b);
    }
}
