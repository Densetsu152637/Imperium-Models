//! RoPE tables on the host, bit for bit what `ncm-model` computes with YaRN off: the `cos` and
//! `sin` of `pos / scale * base^(-2i / dim)`, each value repeated across both halves of the head
//! dimension (the layout the exported graphs' rotation matrix expects).

/// YaRN context extension of a rope table (`ncm_model::layers::Yarn`): frequencies whose
/// wavelength fits fewer than one turn into `original` positions are slowed by `factor`, those
/// with more than 32 turns keep their rate, a linear ramp between; both tables are scaled by the
/// attention temperature `0.1 ln(factor) + 1`. A `factor` of 1 is plain RoPE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    pub original: usize,
}

impl Yarn {
    pub const OFF: Self = Self { factor: 1.0, original: 1 };
    const ALPHA: f64 = 1.0;
    const BETA: f64 = 32.0;

    fn on(&self) -> bool {
        self.factor > 1.0
    }

    fn rate(&self, theta: f64) -> f64 {
        if !self.on() {
            return theta;
        }
        let turns = self.original as f64 * theta / std::f64::consts::TAU;
        let keep = ((turns - Self::ALPHA) / (Self::BETA - Self::ALPHA)).clamp(0.0, 1.0);
        keep * theta + (1.0 - keep) * theta / self.factor
    }

    fn temperature(&self) -> f64 {
        if self.on() { 0.1 * self.factor.ln() + 1.0 } else { 1.0 }
    }
}

/// Writes row `pos`'s tables into `cos` and `sin` (each `dim` long).
pub fn rope_row(
    pos: usize,
    dim: usize,
    base: f64,
    scale: f64,
    yarn: Yarn,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let half = dim / 2;
    let p = pos as f64 / scale;
    let temperature = yarn.temperature();
    for i in 0..half {
        let angle = p * yarn.rate(1.0 / base.powf((2 * i) as f64 / dim as f64));
        let (c, s) = ((angle.cos() * temperature) as f32, (angle.sin() * temperature) as f32);
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
    yarn: Yarn,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; len * dim];
    let mut sin = vec![0f32; len * dim];
    for (row, (c, s)) in cos.chunks_mut(dim).zip(sin.chunks_mut(dim)).enumerate() {
        rope_row(start + row, dim, base, scale, yarn, c, s);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let (cos, sin) = rope_rows(0, 1, 8, 10_000.0, 1.0, Yarn::OFF);
        assert!(cos.iter().all(|&c| c == 1.0));
        assert!(sin.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn halves_repeat_and_the_first_frequency_is_one_radian_per_position() {
        let (cos, sin) = rope_rows(3, 1, 8, 10_000.0, 1.0, Yarn::OFF);
        assert_eq!(cos[0], cos[4]);
        assert_eq!(sin[1], sin[5]);
        assert!((cos[0] - 3f32.cos()).abs() < 1e-6);
        assert!((sin[0] - 3f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn yarn_slows_low_frequencies_and_scales_the_tables() {
        let yarn = Yarn { factor: 4.0, original: 64 };
        let (c, _) = rope_rows(0, 1, 8, 10_000.0, 1.0, yarn);
        let temperature = (0.1 * 4f64.ln() + 1.0) as f32;
        assert!((c[0] - temperature).abs() < 1e-6, "position 0 is the temperature");
        let (plain, _) = rope_rows(100, 1, 8, 10_000.0, 1.0, Yarn::OFF);
        let (slow, _) = rope_rows(100, 1, 8, 10_000.0, 1.0, yarn);
        assert_ne!(plain[3], slow[3], "the lowest frequency is interpolated");
    }

    #[test]
    fn scale_divides_the_position() {
        let (a, _) = rope_rows(4, 1, 8, 10_000.0, 2.0, Yarn::OFF);
        let (b, _) = rope_rows(2, 1, 8, 10_000.0, 1.0, Yarn::OFF);
        assert_eq!(a, b);
    }
}
