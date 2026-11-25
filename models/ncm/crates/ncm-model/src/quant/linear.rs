//! [`QLinear`]: a bias-free linear layer whose weight is dense or quantised.

use burn::module::{Module, Param, ParamId};
use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData};

use super::building;
use super::format::{Format, Packed, quantize};
use crate::layers::init_normal;

/// A bias-free linear layer, `y = x W`. Dense, it holds `weight` `[d_in, d_out]` exactly like
/// `burn::nn::Linear` (same tensor name and layout, so dense weight files are unchanged).
/// Quantised, it holds `qweight` and `scales` in the [packed format](super::format) instead and
/// multiplies through the fused kernels on CUDA or a dequantised reference elsewhere.
#[derive(Module, Debug)]
pub struct QLinear {
    weight: Option<Param<Tensor<2>>>,
    qweight: Option<Param<Tensor<2, Int>>>,
    scales: Option<Param<Tensor<2>>>,
    d_in: usize,
    d_out: usize,
    /// 0 when dense.
    bits: u32,
    /// Columns per scale; 0 for one scale per row.
    group: usize,
}

impl QLinear {
    /// A layer with N(0, 0.02) weights: dense, or quantised from such weights inside
    /// [`super::build`].
    pub fn new(d_in: usize, d_out: usize, device: &Device) -> Self {
        match building() {
            None => Self::dense(init_normal().init([d_in, d_out], device), d_in, d_out),
            Some(b) if b.placeholder => {
                Self::from_packed(&zeros(d_out, d_in, b.spec.format), device)
            }
            Some(b) => {
                let weight = init_normal().init::<2, _>([d_out, d_in], device);
                Self::quantized(weight.val(), b.spec.format, device)
            }
        }
    }

    fn dense(weight: Param<Tensor<2>>, d_in: usize, d_out: usize) -> Self {
        Self { weight: Some(weight), qweight: None, scales: None, d_in, d_out, bits: 0, group: 0 }
    }

    /// Quantises `w` `[d_out, d_in]` (rows are output features).
    pub fn quantized(w: Tensor<2>, format: Format, device: &Device) -> Self {
        let [d_out, d_in] = w.dims();
        let values = w.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 weight");
        Self::from_packed(&quantize(&values, d_out, d_in, format), device)
    }

    pub fn from_packed(p: &Packed, device: &Device) -> Self {
        let f = p.format;
        let qweight = TensorData::new(p.words.clone(), [p.d_out, f.words(p.d_in)]);
        let scales = TensorData::new(p.scales.clone(), [p.d_out, f.groups(p.d_in)]);
        Self {
            weight: None,
            qweight: Some(Param::initialized(
                ParamId::new(),
                Tensor::<2, Int>::from_data(qweight, (device, DType::I32)),
            )),
            scales: Some(Param::from_tensor(Tensor::from_data(scales, (device, DType::F32)))),
            d_in: p.d_in,
            d_out: p.d_out,
            bits: f.bits,
            group: f.group.unwrap_or(0),
        }
    }

    /// The quantised format, or None when dense.
    pub fn format(&self) -> Option<Format> {
        (self.bits != 0)
            .then_some(Format { bits: self.bits, group: (self.group != 0).then_some(self.group) })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.d_in, self.d_out)
    }

    /// The packed weight read back to the host, or None when dense.
    pub fn packed(&self) -> Option<Packed> {
        let format = self.format()?;
        let words = self.qweight.as_ref()?.val().into_data().convert::<i32>();
        let scales = self.scales.as_ref()?.val().into_data().convert::<f32>();
        Some(Packed {
            format,
            d_out: self.d_out,
            d_in: self.d_in,
            words: words.try_into_vec().expect("i32 codes"),
            scales: scales.try_into_vec().expect("f32 scales"),
        })
    }

    /// `x` `[..., d_in]` to `[..., d_out]`. On CUDA with the fused kernels on, decode-sized
    /// inputs take the matrix-vector kernels: [`crate::fused::linear`] when dense,
    /// [`crate::kernels::qmatmul`] when quantised.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        if let Some(weight) = &self.weight {
            return crate::fused::linear(x, weight.val());
        }
        let dims = x.dims();
        let rows = dims[..D - 1].iter().product::<usize>();
        let y = self.matmul(x.reshape([rows, self.d_in]));
        let mut out = dims;
        out[D - 1] = self.d_out;
        y.reshape(out)
    }

    /// `x` `[M, d_in]` times the quantised weight.
    fn matmul(&self, x: Tensor<2>) -> Tensor<2> {
        let format = self.format().expect("quantised layer");
        let (qweight, scales) = (self.qweight_val(), self.scales_val());
        #[cfg(feature = "cuda")]
        if super::kernels_active(&x.device()) {
            return crate::kernels::qmatmul(x, qweight, scales, self.d_in, format);
        }
        let w = dequantize(qweight, scales, self.d_in, format);
        x.clone().matmul(w.cast(float_dtype(&x)))
    }

    fn qweight_val(&self) -> Tensor<2, Int> {
        self.qweight.as_ref().expect("quantised layer").val()
    }

    fn scales_val(&self) -> Tensor<2> {
        self.scales.as_ref().expect("quantised layer").val()
    }

    /// The weight as used, `[d_in, d_out]` in f32: the dense weight, or the decoded one.
    pub fn dense_weight(&self) -> Tensor<2> {
        match (&self.weight, self.format()) {
            (Some(w), _) => w.val().cast(FloatDType::F32),
            (None, Some(format)) => {
                dequantize(self.qweight_val(), self.scales_val(), self.d_in, format)
            }
            (None, None) => unreachable!("a layer is dense or quantised"),
        }
    }
}

/// An all-zero quantised weight (every code at the offset, every scale 0).
fn zeros(d_out: usize, d_in: usize, format: Format) -> Packed {
    let code = format.offset();
    let word = (0..format.per_word()).fold(0i32, |w, j| w | code << (j as u32 * format.bits));
    Packed {
        format,
        d_out,
        d_in,
        words: vec![word; d_out * format.words(d_in)],
        scales: vec![0.0; d_out * format.groups(d_in)],
    }
}

fn float_dtype(x: &Tensor<2>) -> FloatDType {
    match x.dtype() {
        DType::F64 => FloatDType::F64,
        DType::F16 => FloatDType::F16,
        DType::BF16 => FloatDType::BF16,
        _ => FloatDType::F32,
    }
}

/// The reference decode with tensor ops (any backend): `qweight` `[d_out, words]` and `scales`
/// `[d_out, groups]` to the f32 weight `[d_in, d_out]`.
pub fn dequantize(
    qweight: Tensor<2, Int>,
    scales: Tensor<2>,
    d_in: usize,
    format: Format,
) -> Tensor<2> {
    let [d_out, words] = qweight.dims();
    let (per, bits) = (format.per_word(), format.bits as i32);
    let mask = (1i32 << bits) - 1;
    let codes: Vec<Tensor<2, Int>> = (0..per as i32)
        .map(|j| qweight.clone().bitwise_right_shift_scalar(j * bits).bitwise_and_scalar(mask))
        .collect();
    let k_pad = words * per;
    let codes = Tensor::stack::<3>(codes, 2).reshape([d_out, k_pad]);
    let values = codes.float().cast(FloatDType::F32) - format.offset() as f64;
    let (groups, len) = (format.groups(d_in), format.group_len(d_in));
    let scales = scales
        .cast(FloatDType::F32)
        .reshape([d_out, groups, 1])
        .expand([d_out, groups, len])
        .reshape([d_out, k_pad]);
    (values * scales).narrow(1, 0, d_in).transpose()
}
