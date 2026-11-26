//! A small graph builder: values are names, every op returns its output's name, shapes are
//! static and tracked by the caller (the builders in [`crate::layers`] know them).
//!
//! Weights register by key in a [`WeightPool`] shared by every graph of an export, so the
//! per-phase graphs of one model reference the same bytes (one external-data file, one offset
//! per weight). Small tensors stay inline.

use std::collections::HashMap;

use prost::Message;

use crate::onnx::{
    Attr, DType, GraphProto, HostTensor, IR_VERSION, ModelProto, NodeProto, OPSET, OperatorSetId,
    TensorProto, ValueInfoProto, value_info,
};

/// A value in a graph: the name of a node output, an input or an initializer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Val(pub String);

impl Val {
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Tensors at or above this many bytes go to the external data file.
pub const EXTERNAL_MIN_BYTES: usize = 1024;
/// Offsets in the external file are multiples of this (ONNX Runtime memory maps them).
pub const EXTERNAL_ALIGN: usize = 4096;

/// The weights of an export, deduplicated by key, and the bytes of the external data file.
#[derive(Default)]
pub struct WeightPool {
    tensors: HashMap<String, (HostTensor, Option<u64>)>,
    order: Vec<String>,
    /// The external file's bytes so far.
    data: Vec<u8>,
}

impl WeightPool {
    /// Registers `tensor` under `key` (a second registration of the key must be the same
    /// tensor) and returns whether it lives in the external file.
    pub fn add(&mut self, key: &str, tensor: HostTensor) {
        if let Some((existing, _)) = self.tensors.get(key) {
            assert_eq!(*existing, tensor, "weight `{key}` registered twice with different data");
            return;
        }
        let offset = (tensor.bytes.len() >= EXTERNAL_MIN_BYTES).then(|| {
            let at = self.data.len().next_multiple_of(EXTERNAL_ALIGN);
            self.data.resize(at, 0);
            self.data.extend_from_slice(&tensor.bytes);
            at as u64
        });
        self.order.push(key.to_owned());
        self.tensors.insert(key.to_owned(), (tensor, offset));
    }

    /// The initializer of `key` for a graph that uses it, with the data file's name.
    pub fn proto(&self, key: &str, location: &str) -> TensorProto {
        let (tensor, offset) = &self.tensors[key];
        match offset {
            Some(offset) => tensor.external_proto(key, location, *offset),
            None => tensor.proto(key),
        }
    }

    /// The external data file's bytes.
    pub fn external_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Bytes of every registered weight (inline and external).
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|(t, _)| t.bytes.len()).sum()
    }

    pub fn keys(&self) -> &[String] {
        &self.order
    }

    pub fn tensor(&self, key: &str) -> Option<&HostTensor> {
        self.tensors.get(key).map(|(t, _)| t)
    }
}

/// One graph under construction.
pub struct Graph {
    name: String,
    nodes: Vec<NodeProto>,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    /// Weights this graph uses, by key.
    used: Vec<String>,
    /// Inline constants of this graph.
    constants: Vec<TensorProto>,
    counter: usize,
}

impl Graph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            used: Vec::new(),
            constants: Vec::new(),
            counter: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The ops of the graph, in order (for tests and the describe command).
    pub fn op_types(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|n| n.op_type.as_str())
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    /// A graph input.
    pub fn input(&mut self, name: &str, dtype: DType, shape: &[i64]) -> Val {
        self.inputs.push(value_info(name, dtype, shape));
        Val(name.into())
    }

    /// Marks `v` as a graph output named `name` (an Identity renames it).
    pub fn output(&mut self, name: &str, v: &Val, dtype: DType, shape: &[i64]) {
        self.nodes.push(NodeProto {
            input: vec![v.0.clone()],
            output: vec![name.into()],
            name: format!("out_{name}"),
            op_type: "Identity".into(),
            attribute: Vec::new(),
            domain: String::new(),
        });
        self.outputs.push(value_info(name, dtype, shape));
    }

    /// A weight from the pool, used by this graph.
    pub fn weight(&mut self, pool: &mut WeightPool, key: &str, tensor: HostTensor) -> Val {
        pool.add(key, tensor);
        if !self.used.iter().any(|k| k == key) {
            self.used.push(key.to_owned());
        }
        Val(key.into())
    }

    /// A small inline constant.
    pub fn constant(&mut self, hint: &str, tensor: HostTensor) -> Val {
        let name = self.fresh(hint);
        self.constants.push(tensor.proto(&name));
        Val(name)
    }

    pub fn const_f32(&mut self, dims: &[usize], data: &[f32]) -> Val {
        self.constant("c", HostTensor::f32(dims, data))
    }

    pub fn scalar(&mut self, v: f32) -> Val {
        self.const_f32(&[], &[v])
    }

    pub fn const_i64(&mut self, data: &[i64]) -> Val {
        self.constant("i", HostTensor::i64(&[data.len()], data))
    }

    /// A node with one output.
    pub fn op(&mut self, op: &str, inputs: &[&Val], attrs: &[(&str, Attr)]) -> Val {
        self.op_n(op, inputs, attrs, 1).remove(0)
    }

    /// A node with `outputs` outputs.
    pub fn op_n(
        &mut self,
        op: &str,
        inputs: &[&Val],
        attrs: &[(&str, Attr)],
        outputs: usize,
    ) -> Vec<Val> {
        let names: Vec<String> = (0..outputs).map(|_| self.fresh(&op.to_lowercase())).collect();
        let node_name = self.fresh(&format!("n_{op}"));
        self.nodes.push(NodeProto {
            input: inputs.iter().map(|v| v.0.clone()).collect(),
            output: names.clone(),
            name: node_name,
            op_type: op.into(),
            attribute: attrs.iter().map(|(n, a)| a.proto(n)).collect(),
            domain: String::new(),
        });
        names.into_iter().map(Val).collect()
    }

    // ------------------------------------------------------------------ common ops

    pub fn matmul(&mut self, a: &Val, b: &Val) -> Val {
        self.op("MatMul", &[a, b], &[])
    }

    pub fn add(&mut self, a: &Val, b: &Val) -> Val {
        self.op("Add", &[a, b], &[])
    }

    pub fn sub(&mut self, a: &Val, b: &Val) -> Val {
        self.op("Sub", &[a, b], &[])
    }

    pub fn mul(&mut self, a: &Val, b: &Val) -> Val {
        self.op("Mul", &[a, b], &[])
    }

    pub fn div(&mut self, a: &Val, b: &Val) -> Val {
        self.op("Div", &[a, b], &[])
    }

    pub fn neg(&mut self, a: &Val) -> Val {
        self.op("Neg", &[a], &[])
    }

    pub fn exp(&mut self, a: &Val) -> Val {
        self.op("Exp", &[a], &[])
    }

    pub fn sqrt(&mut self, a: &Val) -> Val {
        self.op("Sqrt", &[a], &[])
    }

    pub fn reciprocal(&mut self, a: &Val) -> Val {
        self.op("Reciprocal", &[a], &[])
    }

    pub fn sigmoid(&mut self, a: &Val) -> Val {
        self.op("Sigmoid", &[a], &[])
    }

    pub fn softplus(&mut self, a: &Val) -> Val {
        self.op("Softplus", &[a], &[])
    }

    pub fn silu(&mut self, a: &Val) -> Val {
        let s = self.sigmoid(a);
        self.mul(a, &s)
    }

    pub fn add_scalar(&mut self, a: &Val, s: f32) -> Val {
        let c = self.scalar(s);
        self.add(a, &c)
    }

    pub fn mul_scalar(&mut self, a: &Val, s: f32) -> Val {
        let c = self.scalar(s);
        self.mul(a, &c)
    }

    pub fn reshape(&mut self, a: &Val, shape: &[i64]) -> Val {
        let s = self.const_i64(shape);
        self.op("Reshape", &[a, &s], &[])
    }

    pub fn transpose(&mut self, a: &Val, perm: &[i64]) -> Val {
        self.op("Transpose", &[a], &[("perm", Attr::Ints(perm.to_vec()))])
    }

    pub fn concat(&mut self, parts: &[&Val], axis: i64) -> Val {
        self.op("Concat", parts, &[("axis", Attr::Int(axis))])
    }

    pub fn softmax(&mut self, a: &Val, axis: i64) -> Val {
        self.op("Softmax", &[a], &[("axis", Attr::Int(axis))])
    }

    /// `a[..., start..end, ...]` along `axis`.
    pub fn slice(&mut self, a: &Val, axis: i64, start: i64, end: i64) -> Val {
        let starts = self.const_i64(&[start]);
        let ends = self.const_i64(&[end]);
        let axes = self.const_i64(&[axis]);
        self.op("Slice", &[a, &starts, &ends, &axes], &[])
    }

    pub fn reduce_sum(&mut self, a: &Val, axes: &[i64], keepdims: bool) -> Val {
        let axes = self.const_i64(axes);
        self.op("ReduceSum", &[a, &axes], &[("keepdims", Attr::Int(keepdims as i64))])
    }

    pub fn reduce_mean(&mut self, a: &Val, axes: &[i64], keepdims: bool) -> Val {
        let axes = self.const_i64(axes);
        self.op("ReduceMean", &[a, &axes], &[("keepdims", Attr::Int(keepdims as i64))])
    }

    pub fn gather(&mut self, data: &Val, indices: &Val, axis: i64) -> Val {
        self.op("Gather", &[data, indices], &[("axis", Attr::Int(axis))])
    }

    pub fn cast(&mut self, a: &Val, to: DType) -> Val {
        self.op("Cast", &[a], &[("to", Attr::Int(to as i64))])
    }

    // ------------------------------------------------------------------ serialising

    /// The model with this graph, its weights resolved against `pool` (external ones point at
    /// `data_file`).
    pub fn model(&self, pool: &WeightPool, data_file: &str) -> ModelProto {
        let mut initializer: Vec<TensorProto> =
            self.used.iter().map(|k| pool.proto(k, data_file)).collect();
        initializer.extend(self.constants.iter().cloned());
        ModelProto {
            ir_version: IR_VERSION,
            producer_name: "ncm-export".into(),
            producer_version: env!("CARGO_PKG_VERSION").into(),
            opset_import: vec![OperatorSetId { domain: String::new(), version: OPSET }],
            graph: Some(GraphProto {
                name: self.name.clone(),
                node: self.nodes.clone(),
                initializer,
                input: self.inputs.clone(),
                output: self.outputs.clone(),
            }),
        }
    }

    /// The serialised model.
    pub fn bytes(&self, pool: &WeightPool, data_file: &str) -> Vec<u8> {
        self.model(pool, data_file).encode_to_vec()
    }

    /// Checks the graph is well formed: every node input is a graph input, an initializer or an
    /// earlier node's output, every output exists. Returns the first problem.
    pub fn validate(&self) -> Result<(), String> {
        let mut known: std::collections::HashSet<&str> = std::collections::HashSet::new();
        known.extend(self.inputs.iter().map(|v| v.name.as_str()));
        known.extend(self.used.iter().map(String::as_str));
        known.extend(self.constants.iter().map(|c| c.name.as_str()));
        for node in &self.nodes {
            for input in &node.input {
                if !known.contains(input.as_str()) {
                    return Err(format!(
                        "{} ({}): input `{input}` is not defined before it",
                        node.name, node.op_type
                    ));
                }
            }
            known.extend(node.output.iter().map(String::as_str));
        }
        for out in &self.outputs {
            if !known.contains(out.name.as_str()) {
                return Err(format!("graph output `{}` is not produced", out.name));
            }
        }
        if self.outputs.is_empty() {
            return Err("the graph has no outputs".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_weights_stay_inline_and_big_ones_go_external() {
        let mut pool = WeightPool::default();
        let mut g = Graph::new("g");
        let x = g.input("x", DType::F32, &[1, 4]);
        let small = g.weight(&mut pool, "small", HostTensor::f32(&[4], &[1.0; 4]));
        let big = g.weight(&mut pool, "big", HostTensor::f32(&[64, 64], &[0.5; 4096]));
        let y = g.add(&x, &small);
        let big_row = g.reshape(&big, &[1, 4096]);
        let z = g.reduce_sum(&big_row, &[1], false);
        g.output("y", &y, DType::F32, &[1, 4]);
        g.output("z", &z, DType::F32, &[1]);
        g.validate().expect("well formed");
        let model = g.model(&pool, "weights.bin");
        let inits = &model.graph.as_ref().unwrap().initializer;
        let small = inits.iter().find(|t| t.name == "small").unwrap();
        let big = inits.iter().find(|t| t.name == "big").unwrap();
        assert_eq!(small.data_location, 0);
        assert!(!small.raw_data.is_empty());
        assert_eq!(big.data_location, 1);
        assert!(big.raw_data.is_empty());
        assert_eq!(big.external_data[0].value, "weights.bin");
        assert_eq!(pool.external_bytes().len(), 4096 * 4);
    }

    #[test]
    fn a_weight_shared_by_two_graphs_has_one_offset() {
        let mut pool = WeightPool::default();
        let t = HostTensor::f32(&[32, 32], &[1.0; 1024]);
        let mut a = Graph::new("a");
        let mut b = Graph::new("b");
        a.weight(&mut pool, "w", t.clone());
        b.weight(&mut pool, "w", t);
        let offset = |g: &Graph| {
            g.model(&pool, "w.bin").graph.unwrap().initializer[0].external_data[1].value.clone()
        };
        assert_eq!(offset(&a), offset(&b));
        assert_eq!(pool.external_bytes().len(), 4096);
    }

    #[test]
    fn validate_catches_a_dangling_input() {
        let mut g = Graph::new("g");
        let x = g.input("x", DType::F32, &[1]);
        let ghost = Val("ghost".into());
        let y = g.add(&x, &ghost);
        g.output("y", &y, DType::F32, &[1]);
        assert!(g.validate().unwrap_err().contains("ghost"));
    }
}
