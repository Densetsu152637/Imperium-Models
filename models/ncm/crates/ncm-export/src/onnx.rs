//! The subset of the ONNX protobuf schema the exporter writes (`onnx.proto`, Apache-2.0),
//! derived by hand with `prost` so that no `protoc` or build script is needed. Field numbers
//! follow `onnx.proto3`; only the messages and fields this crate fills are defined, and any
//! ONNX reader (ONNX Runtime, `onnx.load`) decodes them.

use prost::Message;

/// IR version 10: the one that goes with opset 21 (int4 types, block-wise quantisation).
pub const IR_VERSION: i64 = 10;
/// The opset every graph targets.
pub const OPSET: i64 = 21;

/// ONNX `TensorProto.DataType` values this crate uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32 = 1,
    U8 = 2,
    I8 = 3,
    I32 = 6,
    I64 = 7,
    Bool = 9,
    F16 = 10,
    Bf16 = 16,
    U4 = 21,
    I4 = 22,
}

impl DType {
    /// Bytes of one element (4-bit types: half a byte, so 0 here: use [`Self::bits`]).
    pub fn bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::I64 => 64,
            DType::F16 | DType::Bf16 => 16,
            DType::U8 | DType::I8 | DType::Bool => 8,
            DType::U4 | DType::I4 => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::U8 => "u8",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::U4 => "u4",
            DType::I4 => "i4",
        }
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntry {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntry>,
    /// 0 default (inline), 1 external.
    #[prost(int32, tag = "14")]
    pub data_location: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    /// 1 float, 2 int, 3 string, 4 tensor, 6 floats, 7 ints.
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(oneof = "dimension::Value", tags = "1, 2")]
    pub value: Option<dimension::Value>,
}

pub mod dimension {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(int64, tag = "1")]
        DimValue(i64),
        #[prost(string, tag = "2")]
        DimParam(String),
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeTensor>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetId {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetId>,
}

/// An attribute value.
#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Str(String),
}

impl Attr {
    pub fn proto(&self, name: &str) -> AttributeProto {
        let base = AttributeProto {
            name: name.into(),
            f: 0.0,
            i: 0,
            s: Vec::new(),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            r#type: 0,
        };
        match self {
            Attr::Int(i) => AttributeProto { i: *i, r#type: 2, ..base },
            Attr::Float(f) => AttributeProto { f: *f, r#type: 1, ..base },
            Attr::Ints(v) => AttributeProto { ints: v.clone(), r#type: 7, ..base },
            Attr::Floats(v) => AttributeProto { floats: v.clone(), r#type: 6, ..base },
            Attr::Str(s) => AttributeProto { s: s.as_bytes().to_vec(), r#type: 3, ..base },
        }
    }
}

/// A host tensor to store as an initializer: its element type, shape and little-endian bytes
/// (4-bit types packed two per byte, the first element in the low nibble).
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub dtype: DType,
    pub dims: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl HostTensor {
    pub fn f32(dims: &[usize], data: &[f32]) -> Self {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "f32 tensor {dims:?} size");
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { dtype: DType::F32, dims: dims.iter().map(|&d| d as i64).collect(), bytes }
    }

    pub fn i64(dims: &[usize], data: &[i64]) -> Self {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "i64 tensor {dims:?} size");
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { dtype: DType::I64, dims: dims.iter().map(|&d| d as i64).collect(), bytes }
    }

    pub fn i8(dims: &[usize], data: &[i8]) -> Self {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "i8 tensor {dims:?} size");
        let bytes = data.iter().map(|&v| v as u8).collect();
        Self { dtype: DType::I8, dims: dims.iter().map(|&d| d as i64).collect(), bytes }
    }

    /// A signed 4-bit tensor from codes in -8..=7, packed two per byte (low nibble first).
    pub fn i4(dims: &[usize], codes: &[i8]) -> Self {
        assert_eq!(dims.iter().product::<usize>(), codes.len(), "i4 tensor {dims:?} size");
        let mut bytes = vec![0u8; codes.len().div_ceil(2)];
        for (i, &c) in codes.iter().enumerate() {
            debug_assert!((-8..=7).contains(&c), "an int4 code");
            let nibble = (c as u8) & 0x0F;
            bytes[i / 2] |= if i % 2 == 0 { nibble } else { nibble << 4 };
        }
        Self { dtype: DType::I4, dims: dims.iter().map(|&d| d as i64).collect(), bytes }
    }

    pub fn elements(&self) -> usize {
        self.dims.iter().product::<i64>() as usize
    }

    /// The tensor as an inline initializer.
    pub fn proto(&self, name: &str) -> TensorProto {
        TensorProto {
            dims: self.dims.clone(),
            data_type: self.dtype as i32,
            name: name.into(),
            raw_data: self.bytes.clone(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    /// The tensor as an initializer whose bytes live in an external file.
    pub fn external_proto(&self, name: &str, location: &str, offset: u64) -> TensorProto {
        TensorProto {
            dims: self.dims.clone(),
            data_type: self.dtype as i32,
            name: name.into(),
            raw_data: Vec::new(),
            external_data: vec![
                StringStringEntry { key: "location".into(), value: location.into() },
                StringStringEntry { key: "offset".into(), value: offset.to_string() },
                StringStringEntry { key: "length".into(), value: self.bytes.len().to_string() },
            ],
            data_location: 1,
        }
    }
}

/// A graph input or output of a given type and static shape.
pub fn value_info(name: &str, dtype: DType, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TypeTensor {
                elem_type: dtype as i32,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&d| Dimension { value: Some(dimension::Value::DimValue(d)) })
                        .collect(),
                }),
            }),
        }),
    }
}

/// The model's serialised bytes.
pub fn encode(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_codes_pack_low_nibble_first() {
        let t = HostTensor::i4(&[4], &[-8, 7, -1, 0]);
        assert_eq!(t.bytes, vec![0x78, 0x0F]);
        assert_eq!(t.dtype as i32, 22);
        assert_eq!(t.elements(), 4);
    }

    #[test]
    fn odd_int4_counts_pad_the_last_byte() {
        let t = HostTensor::i4(&[3], &[1, 2, 3]);
        assert_eq!(t.bytes, vec![0x21, 0x03]);
    }

    #[test]
    fn a_model_round_trips_through_protobuf() {
        let model = ModelProto {
            ir_version: IR_VERSION,
            producer_name: "ncm-export".into(),
            producer_version: "0".into(),
            opset_import: vec![OperatorSetId { domain: String::new(), version: OPSET }],
            graph: Some(GraphProto {
                name: "g".into(),
                node: vec![NodeProto {
                    input: vec!["x".into(), "w".into()],
                    output: vec!["y".into()],
                    name: "mm".into(),
                    op_type: "MatMul".into(),
                    attribute: vec![Attr::Int(3).proto("axis")],
                    domain: String::new(),
                }],
                initializer: vec![HostTensor::f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).proto("w")],
                input: vec![value_info("x", DType::F32, &[1, 2])],
                output: vec![value_info("y", DType::F32, &[1, 2])],
            }),
        };
        let back = ModelProto::decode(encode(&model).as_slice()).expect("decodes");
        assert_eq!(back, model);
    }
}
