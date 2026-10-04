//! The pixel kernels, with SIMD versions chosen at run time.
//!
//! [`scalar`] holds the reference kernels: plain Rust, the definition of
//! what every other version must compute bit for bit. [`x86`] has SSE4.1
//! and AVX2 versions, [`neon`] aarch64 NEON ones; [`dsp`] picks the best
//! the CPU supports, once. `VP8_FORCE_SCALAR=1` in the environment (read
//! when the first kernel is chosen) keeps the scalar kernels, so the
//! scalar path can be timed and tested on any machine.
//!
//! The kernels that touch a picture take raw pointers: the decoder's
//! macroblock rows run on several threads, each writing its own pixels of
//! one shared picture (see `decoder::SharedFrame`), so no thread may hold a
//! `&mut [u8]` over the whole plane. Every such kernel is an `unsafe fn`
//! whose contract says which bytes it reads and writes; the safe wrappers
//! in `predict`, `transform` and `decoder` check the bounds before calling.

use std::sync::OnceLock;

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

/// Thresholds for one loop filter edge (RFC 6386 section 15), as the
/// kernels take them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EdgeLimits {
    /// Edge limit `E`: `|p0 - q0| * 2 + |p1 - q1| / 2` must not exceed it.
    pub edge: u8,
    /// Interior limit `I` (normal filter only).
    pub interior: u8,
    /// High edge variance threshold.
    pub hev: u8,
}

/// Filters one edge of 16 segments. Segments 0-7 start at `a`, 8-15 at `b`
/// (the second half of a luma edge, or the V plane's edge when `a` is the
/// U plane's). `a` and `b` point at q0, the first pixel past the edge.
///
/// The `_h` kernels filter a horizontal edge: segment `k` is the column at
/// `a + k` (or `b + k - 8`), and the pixels across the edge are `stride`
/// apart. The `_v` kernels filter a vertical edge: segment `k` is the line
/// at `a + k * stride`, and the pixels across the edge are adjacent.
///
/// # Safety
///
/// For every segment, the four pixels before the edge and the four from
/// it (p3 to q3) must be valid for reads and writes, and no other thread
/// may access them during the call.
pub(crate) type EdgeFn = unsafe fn(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits);

/// Six-tap (or bilinear) subpixel prediction of a `w`x`h` block (`w` 4, 8
/// or 16, `h` at most 16) with filter indices `fx`, `fy` (0-7, eighths of a
/// sample) from `filters`: the horizontal pass over `h + 5` rows, then the
/// vertical pass, each rounded and saturated to 8 bits (section 18.3).
///
/// # Safety
///
/// `src` points at the sample two left of and two above the block's
/// origin. For each of the `h + 5` rows, [`SUBPEL_READ`] bytes from `src +
/// row * sstride` must be readable (the filter uses `w + 5` of them; the
/// SIMD kernels load whole vectors). `dst` must be valid for writes of `w`
/// bytes on each of `h` rows `dstride` apart.
pub(crate) type SubpelFn = unsafe fn(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    h: usize,
    fx: usize,
    fy: usize,
    filters: &'static [[i32; 6]; 8],
);

/// Bytes the subpixel kernels may read from each source row.
pub(crate) const SUBPEL_READ: usize = 32;

/// Adds the inverse DCT of `coeffs` to the 4x4 block at `dst` (section 14),
/// saturating to 8 bits.
///
/// # Safety
///
/// `dst` must be valid for reads and writes of 4 bytes on each of 4 rows
/// `stride` apart.
pub(crate) type IdctAddFn = unsafe fn(coeffs: &[i16; 16], dst: *mut u8, stride: usize);

/// The sum of absolute differences of two `w`x`h` blocks.
///
/// # Safety
///
/// `a` and `b` must be valid for reads of `w` bytes on each of `h` rows
/// (`astride`, `bstride` apart).
pub(crate) type SadFn = unsafe fn(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u32;

/// The sum of squared differences of two `w`x`h` blocks; same contract as
/// [`SadFn`].
pub(crate) type SseFn = unsafe fn(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u64;

/// The encoder's forward DCT of one 4x4 block of residue (raster order).
pub(crate) type FdctFn = fn(input: &[i16; 16]) -> [i16; 16];

/// The encoder's dead-zone quantiser for one block (raster order): for
/// each coefficient, `level = min((|c| + round) / step, 2048)` with the
/// coefficient's sign, the division done exactly as `(n * recip) >> 32`
/// (see [`QuantParams`]); returns the levels and the dequantised values
/// `level * step` stored in 16 bits (wrapping, as the decoder's products
/// are). Position 0 uses the DC parameters; with `skip_dc` it is left 0.
pub(crate) type QuantFn =
    fn(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]);

/// A quantiser's parameters, DC (index 0) and AC (index 1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantParams {
    pub step: [i32; 2],
    pub round: [i32; 2],
    /// `ceil(2^32 / step)`: for numerators and steps below 2^16,
    /// `(n * recip) >> 32` is `n / step` exactly.
    pub recip: [u32; 2],
}

impl QuantParams {
    /// From the step sizes (DC, AC) and the rounding offsets.
    pub(crate) fn new(step: [i32; 2], round: [i32; 2]) -> QuantParams {
        assert!(step.iter().all(|&s| (2..1 << 15).contains(&s)));
        assert!(round.iter().zip(&step).all(|(&r, &s)| (0..s).contains(&r)));
        QuantParams {
            step,
            round,
            recip: step.map(|s| (1u64 << 32).div_ceil(s as u64) as u32),
        }
    }
}

/// One set of kernels.
pub(crate) struct Dsp {
    /// What the set is: "scalar", "sse4.1", "avx2", "neon".
    pub name: &'static str,
    pub subpel: SubpelFn,
    pub idct_add: IdctAddFn,
    pub lf_mb_h: EdgeFn,
    pub lf_mb_v: EdgeFn,
    pub lf_sub_h: EdgeFn,
    pub lf_sub_v: EdgeFn,
    pub lf_simple_h: EdgeFn,
    pub lf_simple_v: EdgeFn,
    pub sad: SadFn,
    pub sse: SseFn,
    pub fdct: FdctFn,
    pub quant: QuantFn,
}

/// The scalar reference kernels.
pub(crate) static SCALAR: Dsp = Dsp {
    name: "scalar",
    subpel: scalar::subpel,
    idct_add: scalar::idct_add,
    lf_mb_h: scalar::lf_mb_h,
    lf_mb_v: scalar::lf_mb_v,
    lf_sub_h: scalar::lf_sub_h,
    lf_sub_v: scalar::lf_sub_v,
    lf_simple_h: scalar::lf_simple_h,
    lf_simple_v: scalar::lf_simple_v,
    sad: scalar::sad,
    sse: scalar::sse,
    fdct: scalar::fdct,
    quant: scalar::quant,
};

/// The best kernels this CPU runs (or the scalar ones under
/// `VP8_FORCE_SCALAR`).
#[inline]
pub(crate) fn dsp() -> &'static Dsp {
    static CHOSEN: OnceLock<&'static Dsp> = OnceLock::new();
    CHOSEN.get_or_init(|| {
        if std::env::var_os("VP8_FORCE_SCALAR").is_some_and(|v| v != "0" && !v.is_empty()) {
            return &SCALAR;
        }
        best()
    })
}

/// The best kernel set the CPU supports, ignoring `VP8_FORCE_SCALAR`.
pub(crate) fn best() -> &'static Dsp {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(d) = x86::avx2() {
            return d;
        }
        if let Some(d) = x86::sse41() {
            return d;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return neon::neon();
    }
    #[allow(unreachable_code)]
    &SCALAR
}

/// The name of the kernel set in use ("scalar", "sse4.1", "avx2", "neon").
pub fn simd_level() -> &'static str {
    dsp().name
}

#[cfg(test)]
mod tests;
