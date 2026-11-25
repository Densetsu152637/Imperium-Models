//! In-place writes into an existing buffer, whoever else holds it.
//!
//! Burn's `slice_assign` writes in place only when the tensor's buffer has no other holder and
//! copies otherwise. A captured CUDA graph holds every buffer it touched, so a cache or input
//! it reads would be copied away by `slice_assign` and the graph would keep reading the stale
//! buffer. These writes always land in the buffer itself.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube.
pub(crate) const CUBE: usize = 256;

/// `dst[.., at + i, ..] = src[.., i, ..]` along axis `axis` (any rank, both strided): one unit
/// per element of `src` (`count` of them).
#[cube(launch_unchecked)]
pub(crate) fn assign_kernel<F: Numeric>(
    dst: &mut Tensor<F>,
    src: &Tensor<F>,
    count: usize,
    axis: usize,
    at: usize,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= count {
        terminate!();
    }
    let mut rem = pos;
    let mut from = 0usize;
    let mut to = 0usize;
    let mut d = src.rank();
    while d > 0 {
        d -= 1;
        let n = src.shape(d);
        let c = rem % n;
        rem /= n;
        from += c * src.stride(d);
        let shift = select(d == axis, at, 0usize);
        to += (c + shift) * dst.stride(d);
    }
    dst[to] = src[from];
}

/// [`assign_kernel`] at the offset held by the device tensor `at` (`[1]`, i32): the write lands
/// where the host put the position before a replay, so one captured launch serves every step.
#[cube(launch_unchecked)]
pub(crate) fn assign_at_kernel<F: Numeric>(
    dst: &mut Tensor<F>,
    src: &Tensor<F>,
    at: &Tensor<i32>,
    count: usize,
    axis: usize,
    #[define(F)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= count {
        terminate!();
    }
    let offset = usize::cast_from(at[0]);
    let mut rem = pos;
    let mut from = 0usize;
    let mut to = 0usize;
    let mut d = src.rank();
    while d > 0 {
        d -= 1;
        let n = src.shape(d);
        let c = rem % n;
        rem /= n;
        from += c * src.stride(d);
        let shift = select(d == axis, offset, 0usize);
        to += (c + shift) * dst.stride(d);
    }
    dst[to] = src[from];
}
