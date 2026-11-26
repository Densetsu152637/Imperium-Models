//! Footprint types, shared by the NPU plan and the CPU offload plan (`docs/plans/npu.md`,
//! decision 2). `ncm-exec` owns the types; the offload planner (in `ncm-model`) fills them.

use std::fmt;

/// Where bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Device memory (VRAM).
    Device,
    /// Page-locked host memory the device reads over the bus.
    PinnedHost,
    /// Ordinary system RAM (also the NPU backends' working memory: NPUs share system DRAM).
    SystemRam,
    /// An NPU compile cache on disk: reported, never budgeted.
    NpuCompileCache,
}

impl Tier {
    /// Every tier, in report order.
    pub const ALL: [Tier; 4] =
        [Tier::Device, Tier::PinnedHost, Tier::SystemRam, Tier::NpuCompileCache];

    /// Whether a budget (VRAM or RAM) applies to the tier.
    pub fn budgeted(self) -> bool {
        self != Tier::NpuCompileCache
    }

    fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Tier::Device => "device",
            Tier::PinnedHost => "pinned-host",
            Tier::SystemRam => "system-ram",
            Tier::NpuCompileCache => "npu-compile-cache",
        }
    }
}

/// What the bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Part {
    Weights,
    /// The latent (MLA) key/value cache.
    LatentCache,
    /// The sparse-retrieval indexer's keys.
    IndexerKeys,
    /// The local rings (attention rings, encoder rings).
    Rings,
    DeltaNetState,
    /// Prefill staging buffers.
    PrefillStaging,
    /// The hybrid's union slab.
    UnionSlab,
    /// Graph workspace (CUDA graph pools, ORT arenas).
    GraphWorkspace,
}

impl Part {
    pub const ALL: [Part; 8] = [
        Part::Weights,
        Part::LatentCache,
        Part::IndexerKeys,
        Part::Rings,
        Part::DeltaNetState,
        Part::PrefillStaging,
        Part::UnionSlab,
        Part::GraphWorkspace,
    ];

    fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Part::Weights => "weights",
            Part::LatentCache => "latent-cache",
            Part::IndexerKeys => "indexer-keys",
            Part::Rings => "rings",
            Part::DeltaNetState => "deltanet-state",
            Part::PrefillStaging => "prefill-staging",
            Part::UnionSlab => "union-slab",
            Part::GraphWorkspace => "graph-workspace",
        }
    }
}

/// Bytes by [`Tier`] x [`Part`].
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    bytes: [[u64; Part::ALL.len()]; Tier::ALL.len()],
}

impl Footprint {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `bytes` of `part` in `tier`.
    pub fn add(&mut self, tier: Tier, part: Part, bytes: u64) -> &mut Self {
        self.bytes[tier.index()][part.index()] += bytes;
        self
    }

    /// Builder form of [`Self::add`].
    pub fn with(mut self, tier: Tier, part: Part, bytes: u64) -> Self {
        self.add(tier, part, bytes);
        self
    }

    pub fn get(&self, tier: Tier, part: Part) -> u64 {
        self.bytes[tier.index()][part.index()]
    }

    /// Bytes of every part in `tier`.
    pub fn tier_total(&self, tier: Tier) -> u64 {
        self.bytes[tier.index()].iter().sum()
    }

    /// Bytes of `part` over every tier.
    pub fn part_total(&self, part: Part) -> u64 {
        self.bytes.iter().map(|row| row[part.index()]).sum()
    }

    /// Bytes over every budgeted tier (all but the NPU compile cache).
    pub fn budgeted_total(&self) -> u64 {
        Tier::ALL.iter().filter(|t| t.budgeted()).map(|&t| self.tier_total(t)).sum()
    }

    /// Bytes over every tier.
    pub fn total(&self) -> u64 {
        Tier::ALL.iter().map(|&t| self.tier_total(t)).sum()
    }

    /// The sum of two footprints.
    pub fn merged(mut self, other: &Footprint) -> Self {
        for (a, b) in self.bytes.iter_mut().flatten().zip(other.bytes.iter().flatten()) {
            *a += *b;
        }
        self
    }

    /// The non-zero entries, tier-major.
    pub fn entries(&self) -> impl Iterator<Item = (Tier, Part, u64)> + '_ {
        Tier::ALL.iter().flat_map(move |&t| {
            Part::ALL.iter().filter_map(move |&p| {
                let b = self.get(t, p);
                (b > 0).then_some((t, p, b))
            })
        })
    }
}

impl fmt::Debug for Footprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut m = f.debug_map();
        for (t, p, b) in self.entries() {
            m.entry(&format_args!("{}/{}", t.name(), p.name()), &b);
        }
        m.finish()
    }
}

/// The float type of activations and caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    pub fn itemsize(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }
}

/// Weight quantisation of the linear layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quant {
    None,
    /// 8-bit codes, one scale per `group` input values (0: per output channel).
    Int8 {
        group: usize,
    },
    /// 4-bit codes, one scale per `group` input values.
    Int4 {
        group: usize,
    },
}

impl Quant {
    /// Bits of one weight code, given the bits of a dense weight.
    pub fn bits(self, dense_bits: usize) -> usize {
        match self {
            Quant::None => dense_bits,
            Quant::Int8 { .. } => 8,
            Quant::Int4 { .. } => 4,
        }
    }
}

/// Where each layer's weights and each cache live; the offload planner's output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The tier of each layer's weights (indexed by layer).
    pub weights: Vec<Tier>,
    /// The tier of each layer's cache (indexed by layer).
    pub caches: Vec<Tier>,
}

impl Placement {
    /// Every layer's weights and cache in `tier`.
    pub fn uniform(layers: usize, tier: Tier) -> Self {
        Self { weights: vec![tier; layers], caches: vec![tier; layers] }
    }

    /// Layers whose weights are in `tier`.
    pub fn weights_in(&self, tier: Tier) -> usize {
        self.weights.iter().filter(|&&t| t == tier).count()
    }

    pub fn caches_in(&self, tier: Tier) -> usize {
        self.caches.iter().filter(|&&t| t == tier).count()
    }
}

/// What a backend is asked to size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FootprintRequest {
    /// Context tokens per slot.
    pub ctx: usize,
    /// Concurrent requests (slots).
    pub slots: usize,
    /// Cache positions per token, as `(numerator, denominator)` to stay exact: 1/1 for the token
    /// baseline, 1/block for the hybrid's backbone positions.
    pub tokens_per_position: (usize, usize),
    pub dtype: Dtype,
    pub quant: Quant,
    /// None: everything resident in the backend's own tier.
    pub placement: Option<Placement>,
}

impl FootprintRequest {
    pub fn new(ctx: usize, slots: usize, dtype: Dtype, quant: Quant) -> Self {
        Self { ctx, slots, tokens_per_position: (1, 1), dtype, quant, placement: None }
    }

    /// Cache positions needed for `ctx` tokens: `ceil(ctx * num / den)`.
    pub fn positions(&self) -> usize {
        let (num, den) = self.tokens_per_position;
        (self.ctx * num).div_ceil(den.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_by_tier_and_part() {
        let mut f = Footprint::new();
        f.add(Tier::Device, Part::Weights, 100)
            .add(Tier::Device, Part::Rings, 10)
            .add(Tier::SystemRam, Part::Weights, 40)
            .add(Tier::NpuCompileCache, Part::GraphWorkspace, 7);
        assert_eq!(f.tier_total(Tier::Device), 110);
        assert_eq!(f.tier_total(Tier::SystemRam), 40);
        assert_eq!(f.part_total(Part::Weights), 140);
        assert_eq!(f.total(), 157);
        assert_eq!(f.budgeted_total(), 150, "the compile cache is reported, not budgeted");
    }

    #[test]
    fn add_accumulates_and_merge_sums() {
        let a = Footprint::new().with(Tier::Device, Part::LatentCache, 5);
        let b = Footprint::new().with(Tier::Device, Part::LatentCache, 6).with(
            Tier::PinnedHost,
            Part::IndexerKeys,
            2,
        );
        let m = a.merged(&b);
        assert_eq!(m.get(Tier::Device, Part::LatentCache), 11);
        assert_eq!(m.get(Tier::PinnedHost, Part::IndexerKeys), 2);
        assert_eq!(m.total(), a.total() + b.total());
        assert_eq!(m.entries().count(), 2);
    }

    #[test]
    fn positions_round_up() {
        let mut r = FootprintRequest::new(1000, 1, Dtype::Bf16, Quant::None);
        assert_eq!(r.positions(), 1000);
        r.tokens_per_position = (1, 32);
        assert_eq!(r.positions(), 32, "ceil(1000 / 32)");
        r.ctx = 1024;
        assert_eq!(r.positions(), 32);
    }

    #[test]
    fn placement_counts() {
        let mut p = Placement::uniform(4, Tier::Device);
        p.weights[3] = Tier::SystemRam;
        assert_eq!(p.weights_in(Tier::Device), 3);
        assert_eq!(p.weights_in(Tier::SystemRam), 1);
        assert_eq!(p.caches_in(Tier::Device), 4);
    }

    #[test]
    fn itemsize_and_bits() {
        assert_eq!(Dtype::F32.itemsize(), 4);
        assert_eq!(Dtype::Bf16.itemsize(), 2);
        assert_eq!(Quant::Int4 { group: 64 }.bits(16), 4);
        assert_eq!(Quant::None.bits(16), 16);
    }
}
