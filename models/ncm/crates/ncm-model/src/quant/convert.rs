//! Converting a dense model, and quantised weight files.
//!
//! A quantised weight file is a [weight file](crate::weights) whose quantised layers hold
//! `<layer>.qweight` (i32) and `<layer>.scales` (f32) in place of `<layer>.weight`, and whose
//! header adds `ncm_quant` (the [`QuantSpec::name`]). A quantised output head is `head.qweight`
//! and `head.scales`, quantised from `emb.weight`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData};

use super::format::{Format, Packed, quantize};
use super::{QuantSpec, build_placeholder};
use crate::config::Config;
use crate::weights::{self, CONFIG_KEY, FORMAT, FORMAT_KEY, WeightsError};

/// Header key holding the quantisation ([`QuantSpec::name`]).
pub const QUANT_KEY: &str = "ncm_quant";

#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error("{path} is not a quantised weight file (no `{QUANT_KEY}` in its header)")]
    NotQuantized { path: String },
    #[error("{path}: invalid `{QUANT_KEY}` value `{value}`")]
    BadSpec { path: String, value: String },
    #[error("the dense model does not match the quantised one: {0}")]
    Mismatch(String),
}

impl QuantSpec {
    /// Inverse of [`QuantSpec::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        let (base, head) = match name.strip_suffix("+head") {
            Some(base) => (base, true),
            None => (name, false),
        };
        let format = match base.split_once("-g") {
            None if base == "q8" => Format::q8(),
            Some(("q4", group)) => {
                Format::q4(group.parse().ok().filter(|g: &usize| *g > 0 && g.is_multiple_of(32))?)
            }
            _ => return None,
        };
        Some(Self { format, head })
    }
}

/// Path of the module being visited or mapped, joined with dots as tensor names are.
#[derive(Default)]
struct PathStack(Vec<String>);

impl PathStack {
    fn enter(&mut self, name: &str) {
        self.0.push(name.to_owned());
    }

    fn exit(&mut self) {
        self.0.pop();
    }

    fn joined(&self) -> String {
        self.0.join(".")
    }
}

/// Collects every float parameter of the dense model by name, flattened.
#[derive(Default)]
struct Sources {
    path: PathStack,
    tensors: HashMap<String, (Vec<usize>, Tensor<1>)>,
}

impl ModuleVisitor for Sources {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.enter(name);
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.exit();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param.val();
        let dims = tensor.dims().to_vec();
        let n = dims.iter().product::<usize>();
        self.tensors.insert(self.path.joined(), (dims, tensor.reshape([n])));
    }
}

/// Fills a placeholder quantised model from the dense model's tensors.
struct Filler {
    path: PathStack,
    sources: HashMap<String, (Vec<usize>, Tensor<1>)>,
    /// Source names that found a place.
    used: HashSet<String>,
    format: Format,
    device: Device,
    packed: HashMap<String, Packed>,
    errors: Vec<String>,
}

/// Name of the dense `[d_out, d_in]` source of a quantised layer and whether it is stored
/// transposed (`[d_in, d_out]`, a linear layer's own weight) rather than as output rows (the
/// embedding table behind the tied head).
fn source_of(layer: &str) -> (String, bool) {
    if layer == "head" { ("emb.weight".into(), false) } else { (format!("{layer}.weight"), true) }
}

impl Filler {
    fn packed(&mut self, layer: &str) -> Option<&Packed> {
        if !self.packed.contains_key(layer) {
            let (name, transposed) = source_of(layer);
            self.used.insert(name.clone());
            let Some((dims, flat)) = self.sources.get(&name).cloned() else {
                self.errors.push(format!("no dense `{name}` for quantised layer `{layer}`"));
                return None;
            };
            let w = flat.reshape([dims[0], dims[1]]);
            let w = if transposed { w.transpose() } else { w };
            let [d_out, d_in] = w.dims();
            let values = w.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 weight");
            self.packed.insert(layer.to_owned(), quantize(&values, d_out, d_in, self.format));
        }
        self.packed.get(layer)
    }

    /// The layer a `qweight` or `scales` parameter belongs to.
    fn layer(&self, leaf: &str) -> Option<String> {
        let path = self.path.joined();
        path.strip_suffix(leaf).and_then(|p| p.strip_suffix('.')).map(str::to_owned)
    }

    fn check_shape(&mut self, name: &str, want: &[usize], got: &[usize]) -> bool {
        if want != got {
            self.errors.push(format!("`{name}`: dense {got:?}, quantised model {want:?}"));
        }
        want == got
    }
}

impl ModuleMapper for Filler {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.enter(name);
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.exit();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let name = self.path.joined();
        let want = param.shape().dims::<D>().to_vec();
        let device = self.device.clone();
        if let Some(layer) = self.layer("scales") {
            let Some(p) = self.packed(&layer) else { return param };
            let data = TensorData::new(p.scales.clone(), [p.d_out, p.format.groups(p.d_in)]);
            let shape = data.shape().to_vec();
            if !self.check_shape(&name, &want, &shape) {
                return param;
            }
            return param.map(|_| Tensor::from_data(data, (&device, DType::F32)));
        }
        let Some((dims, flat)) = self.sources.get(&name).cloned() else {
            self.errors.push(format!("no dense `{name}`"));
            return param;
        };
        self.used.insert(name.clone());
        if !self.check_shape(&name, &want, &dims) {
            return param;
        }
        let dims: [usize; D] = dims.try_into().expect("rank");
        param.map(|_| flat.reshape(dims))
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let name = self.path.joined();
        let want = param.shape().dims::<D>().to_vec();
        let device = self.device.clone();
        let Some(layer) = self.layer("qweight") else {
            self.errors.push(format!("unexpected int parameter `{name}`"));
            return param;
        };
        let Some(p) = self.packed(&layer) else { return param };
        let data = TensorData::new(p.words.clone(), [p.d_out, p.format.words(p.d_in)]);
        let shape = data.shape().to_vec();
        if !self.check_shape(&name, &want, &shape) {
            return param;
        }
        param.map(|_| Tensor::from_data(data, (&device, DType::I32)))
    }
}

/// Quantises the dense model `dense` per `spec`: `build` constructs the same architecture
/// (called inside the quantising build mode), and its parameters are filled from `dense`. The
/// quantised layers' weights are quantised on the host; everything else is carried over.
pub fn quantize_model<M: Module>(
    dense: &M,
    spec: QuantSpec,
    device: &Device,
    build: impl FnOnce() -> M,
) -> Result<M, QuantError> {
    let mut sources = Sources::default();
    dense.visit(&mut sources);
    let target = build_placeholder(spec, build);
    let mut filler = Filler {
        path: PathStack::default(),
        sources: sources.tensors,
        used: HashSet::new(),
        format: spec.format,
        device: device.clone(),
        packed: HashMap::new(),
        errors: Vec::new(),
    };
    let target = target.map(&mut filler);
    let mut unused: Vec<_> =
        filler.sources.keys().filter(|k| !filler.used.contains(*k)).cloned().collect();
    unused.sort();
    if !unused.is_empty() {
        filler.errors.push(format!("dense tensors without a place: {}", unused.join(", ")));
    }
    if filler.errors.is_empty() {
        Ok(target)
    } else {
        Err(QuantError::Mismatch(filler.errors.join("; ")))
    }
}

/// Saves a quantised model like [`weights::save_weights`], with `spec` in the header.
pub fn save_quantized<M: Module>(
    module: &M,
    config: &Config,
    spec: QuantSpec,
    path: &Path,
) -> Result<(), QuantError> {
    let mut store = SafetensorsStore::from_file(path)
        .overwrite(true)
        .metadata(FORMAT_KEY, FORMAT)
        .metadata(CONFIG_KEY, config.to_toml())
        .metadata(QUANT_KEY, spec.name());
    module.save_into(&mut store).map_err(|e| {
        WeightsError::Store { path: path.display().to_string(), message: e.to_string() }.into()
    })
}

/// The quantisation recorded in a weight file's header; None for a dense file.
pub fn read_spec(path: &Path) -> Result<Option<QuantSpec>, QuantError> {
    let Some(value) = weights::read_metadata(path)?.remove(QUANT_KEY) else { return Ok(None) };
    QuantSpec::from_name(&value)
        .map(Some)
        .ok_or_else(|| QuantError::BadSpec { path: path.display().to_string(), value })
}

/// Casts every float parameter but the quantisation scales (kept f32) to one dtype.
struct CastExceptScales {
    path: PathStack,
    dtype: FloatDType,
}

impl ModuleMapper for CastExceptScales {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.enter(name);
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.exit();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if self.path.0.last().is_some_and(|leaf| leaf == "scales") {
            param
        } else {
            param.map(|t| t.cast(self.dtype))
        }
    }
}

/// Loads a quantised weight file: `build` constructs the architecture (inside the file's
/// quantising build mode, with placeholder weights), the file's tensors replace them and the
/// float parameters other than scales are cast to `dtype`. Every tensor must match. Returns the
/// model and the file's spec.
pub fn load_quantized<M: Module>(
    path: &Path,
    dtype: FloatDType,
    build: impl FnOnce() -> M,
) -> Result<(M, QuantSpec), QuantError> {
    let shown = path.display().to_string();
    let spec = read_spec(path)?.ok_or_else(|| QuantError::NotQuantized { path: shown.clone() })?;
    let mut module = build_placeholder(spec, build);
    let mut store = SafetensorsStore::from_file(path);
    let result = module
        .load_from(&mut store)
        .map_err(|e| WeightsError::Store { path: shown.clone(), message: e.to_string() })?;
    if !result.is_success() || !result.unused.is_empty() {
        return Err(WeightsError::Mismatch { path: shown, details: result.to_string() }.into());
    }
    let module = module.map(&mut CastExceptScales { path: PathStack::default(), dtype });
    Ok((module, spec))
}
