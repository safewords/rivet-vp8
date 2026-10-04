//! The reference kernels: what every SIMD version must reproduce exactly.

use super::{EdgeLimits, QuantParams};

// ---------------------------------------------------------------------------
// Loop filter (RFC 6386 section 15)

/// Saturates to the signed 8-bit range (section 15.2's `c`).
#[inline]
fn c8(v: i32) -> i32 {
    v.clamp(-128, 127)
}

/// Pixel to signed (`u2s`).
#[inline]
fn s(v: u8) -> i32 {
    v as i32 - 128
}

/// Signed back to pixel, saturating (`s2u`).
#[inline]
fn u(v: i32) -> u8 {
    (c8(v) + 128) as u8
}

// One segment across an edge: p3 p2 p1 p0 | q0 q1 q2 q3.
const P3: usize = 0;
const P2: usize = 1;
const P1: usize = 2;
const P0: usize = 3;
const Q0: usize = 4;
const Q1: usize = 5;
const Q2: usize = 6;
const Q3: usize = 7;

/// The kinds of edge filter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EdgeKind {
    /// The normal filter at a macroblock edge.
    Mb,
    /// The normal filter at a subblock edge.
    Sub,
    /// The simple filter (either edge; only the limit differs).
    Simple,
}

/// Section 15.2's `common_adjust`: moves p0 and q0 towards each other and
/// returns the adjustment `a`.
#[inline]
fn common_adjust(x: &mut [u8; 8], use_outer_taps: bool) -> i32 {
    let p1 = s(x[P1]);
    let p0 = s(x[P0]);
    let q0 = s(x[Q0]);
    let q1 = s(x[Q1]);
    let mut a = c8(if use_outer_taps { c8(p1 - q1) } else { 0 } + 3 * (q0 - p0));
    let b = c8(a + 3) >> 3;
    a = c8(a + 4) >> 3;
    x[Q0] = u(q0 - a);
    x[P0] = u(p0 + b);
    a
}

#[inline]
fn edge_ok(x: &[u8; 8], limit: i32) -> bool {
    let d = |a: usize, b: usize| (x[a] as i32 - x[b] as i32).abs();
    d(P0, Q0) * 2 + d(P1, Q1) / 2 <= limit
}

/// Section 15.3's `filter_yes`: the edge test plus every interior
/// difference within `interior`.
#[inline]
fn filter_yes(x: &[u8; 8], e: i32, i: i32) -> bool {
    let d = |a: usize, b: usize| (x[a] as i32 - x[b] as i32).abs();
    edge_ok(x, e)
        && d(P3, P2) <= i
        && d(P2, P1) <= i
        && d(P1, P0) <= i
        && d(Q3, Q2) <= i
        && d(Q2, Q1) <= i
        && d(Q1, Q0) <= i
}

#[inline]
fn hev(x: &[u8; 8], t: i32) -> bool {
    let d = |a: usize, b: usize| (x[a] as i32 - x[b] as i32).abs();
    d(P1, P0) > t || d(Q1, Q0) > t
}

/// Filters one segment `x` (p3 to q3) of an edge of the given kind.
pub(crate) fn filter_segment(kind: EdgeKind, x: &mut [u8; 8], lim: EdgeLimits) {
    let (e, i, t) = (lim.edge as i32, lim.interior as i32, lim.hev as i32);
    match kind {
        EdgeKind::Simple => {
            if edge_ok(x, e) {
                common_adjust(x, true);
            }
        }
        EdgeKind::Sub => {
            if filter_yes(x, e, i) {
                let hv = hev(x, t);
                let p1 = s(x[P1]);
                let q1 = s(x[Q1]);
                let a = (common_adjust(x, hv) + 1) >> 1;
                if !hv {
                    x[Q1] = u(q1 - a);
                    x[P1] = u(p1 + a);
                }
            }
        }
        EdgeKind::Mb => {
            if !filter_yes(x, e, i) {
                return;
            }
            if hev(x, t) {
                common_adjust(x, true);
                return;
            }
            let p2 = s(x[P2]);
            let p1 = s(x[P1]);
            let p0 = s(x[P0]);
            let q0 = s(x[Q0]);
            let q1 = s(x[Q1]);
            let q2 = s(x[Q2]);
            let w = c8(c8(p1 - q1) + 3 * (q0 - p0));
            let a = c8((27 * w + 63) >> 7);
            x[Q0] = u(q0 - a);
            x[P0] = u(p0 + a);
            let a = c8((18 * w + 63) >> 7);
            x[Q1] = u(q1 - a);
            x[P1] = u(p1 + a);
            let a = c8((9 * w + 63) >> 7);
            x[Q2] = u(q2 - a);
            x[P2] = u(p2 + a);
        }
    }
}

/// Gathers each segment of an edge, filters it, and writes it back.
///
/// # Safety
///
/// As [`super::EdgeFn`]; `across` is the distance between pixels across
/// the edge, `along` that between segments.
#[inline]
unsafe fn edge(
    kind: EdgeKind,
    a: *mut u8,
    b: *mut u8,
    across: usize,
    along: usize,
    lim: EdgeLimits,
) {
    for k in 0..16 {
        let at = if k < 8 {
            a.wrapping_add(k * along)
        } else {
            b.wrapping_add((k - 8) * along)
        };
        let first = at.wrapping_sub(4 * across);
        let mut x = [0u8; 8];
        for (j, v) in x.iter_mut().enumerate() {
            // SAFETY: the caller guarantees p3..q3 of every segment are
            // valid and not accessed by another thread.
            *v = unsafe { *first.add(j * across) };
        }
        filter_segment(kind, &mut x, lim);
        for (j, &v) in x.iter().enumerate() {
            // SAFETY: as above.
            unsafe { *first.add(j * across) = v };
        }
    }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_mb_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Mb, a, b, stride, 1, lim) }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_mb_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Mb, a, b, 1, stride, lim) }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_sub_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Sub, a, b, stride, 1, lim) }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_sub_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Sub, a, b, 1, stride, lim) }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_simple_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Simple, a, b, stride, 1, lim) }
}

/// # Safety
/// See [`super::EdgeFn`].
pub(crate) unsafe fn lf_simple_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe { edge(EdgeKind::Simple, a, b, 1, stride, lim) }
}

// ---------------------------------------------------------------------------
// Subpixel interpolation (section 18)

/// # Safety
/// See [`super::SubpelFn`] (this kernel reads only `w + 5` bytes per row).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn subpel(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    h: usize,
    fx: usize,
    fy: usize,
    filters: &'static [[i32; 6]; 8],
) {
    const MAXW: usize = 16 + 5;
    let (ww, wh) = (w + 5, h + 5);
    let mut win = [0u8; MAXW * MAXW];
    for r in 0..wh {
        // SAFETY: the caller guarantees `w + 5` readable bytes on each of
        // the `h + 5` rows.
        let row = unsafe { std::slice::from_raw_parts(src.add(r * sstride), ww) };
        win[r * MAXW..r * MAXW + ww].copy_from_slice(row);
    }
    let mut out = [0u8; 16 * 16];
    if fx == 0 && fy == 0 {
        for r in 0..h {
            out[r * 16..r * 16 + w]
                .copy_from_slice(&win[(r + 2) * MAXW + 2..(r + 2) * MAXW + 2 + w]);
        }
    } else {
        // Horizontal pass over every window row, then the vertical pass;
        // each rounds and saturates to 8 bits.
        let hf = &filters[fx];
        let mut mid = [0u8; MAXW * 16];
        for r in 0..wh {
            let row = &win[r * MAXW..];
            for c in 0..w {
                let t = &row[c..c + 6];
                let s: i32 = (0..6).map(|i| t[i] as i32 * hf[i]).sum();
                mid[r * 16 + c] = ((s + 64) >> 7).clamp(0, 255) as u8;
            }
        }
        let vf = &filters[fy];
        for r in 0..h {
            for c in 0..w {
                let s: i32 = (0..6).map(|i| mid[(r + i) * 16 + c] as i32 * vf[i]).sum();
                out[r * 16 + c] = ((s + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    }
    for r in 0..h {
        // SAFETY: the caller guarantees `w` writable bytes on each of `h`
        // rows of `dst`.
        let d = unsafe { std::slice::from_raw_parts_mut(dst.add(r * dstride), w) };
        d.copy_from_slice(&out[r * 16..r * 16 + w]);
    }
}

// ---------------------------------------------------------------------------
// Inverse transform (section 14.4)

/// sqrt(2) * cos(pi / 8) - 1, and sqrt(2) * sin(pi / 8), in 16-bit fixed
/// point (section 14.4).
pub(crate) const COS_PI8_SQRT2_MINUS1: i32 = 20091;
pub(crate) const SIN_PI8_SQRT2: i32 = 35468;

#[inline]
fn mul_sin(x: i32) -> i32 {
    (x * SIN_PI8_SQRT2) >> 16
}

#[inline]
fn mul_cos(x: i32) -> i32 {
    x + ((x * COS_PI8_SQRT2_MINUS1) >> 16)
}

/// The inverse DCT of one 4x4 block (section 14.4): dequantised
/// coefficients in raster order in, residue in raster order out. Section 14
/// stores the input, the intermediate and the output in 16 bits; the
/// `as i16` casts are those stores.
pub(crate) fn inverse_dct(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i16; 16];
    // Vertical pass: one column at a time.
    for i in 0..4 {
        let (i0, i4, i8, i12) = (
            input[i] as i32,
            input[4 + i] as i32,
            input[8 + i] as i32,
            input[12 + i] as i32,
        );
        let a1 = i0 + i8;
        let b1 = i0 - i8;
        let c1 = mul_sin(i4) - mul_cos(i12);
        let d1 = mul_cos(i4) + mul_sin(i12);
        tmp[i] = (a1 + d1) as i16;
        tmp[12 + i] = (a1 - d1) as i16;
        tmp[4 + i] = (b1 + c1) as i16;
        tmp[8 + i] = (b1 - c1) as i16;
    }
    let mut out = [0i16; 16];
    // Horizontal pass with the final rounding.
    for r in 0..4 {
        let row = &tmp[r * 4..r * 4 + 4];
        let (i0, i1, i2, i3) = (row[0] as i32, row[1] as i32, row[2] as i32, row[3] as i32);
        let a1 = i0 + i2;
        let b1 = i0 - i2;
        let c1 = mul_sin(i1) - mul_cos(i3);
        let d1 = mul_cos(i1) + mul_sin(i3);
        out[r * 4] = ((a1 + d1 + 4) >> 3) as i16;
        out[r * 4 + 3] = ((a1 - d1 + 4) >> 3) as i16;
        out[r * 4 + 1] = ((b1 + c1 + 4) >> 3) as i16;
        out[r * 4 + 2] = ((b1 - c1 + 4) >> 3) as i16;
    }
    out
}

/// # Safety
/// See [`super::IdctAddFn`].
pub(crate) unsafe fn idct_add(coeffs: &[i16; 16], dst: *mut u8, stride: usize) {
    let res = inverse_dct(coeffs);
    for r in 0..4 {
        // SAFETY: the caller guarantees 4 bytes on each of 4 rows.
        let row = unsafe { std::slice::from_raw_parts_mut(dst.add(r * stride), 4) };
        for (c, p) in row.iter_mut().enumerate() {
            *p = (*p as i32 + res[r * 4 + c] as i32).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder kernels

/// # Safety
/// See [`super::SadFn`].
pub(crate) unsafe fn sad(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut s = 0u32;
    for r in 0..h {
        // SAFETY: the caller guarantees `w` bytes on each of `h` rows.
        let (ra, rb) = unsafe {
            (
                std::slice::from_raw_parts(a.add(r * astride), w),
                std::slice::from_raw_parts(b.add(r * bstride), w),
            )
        };
        s += ra
            .iter()
            .zip(rb)
            .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
            .sum::<u32>();
    }
    s
}

/// # Safety
/// See [`super::SseFn`].
pub(crate) unsafe fn sse(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut s = 0u64;
    for r in 0..h {
        // SAFETY: the caller guarantees `w` bytes on each of `h` rows.
        let (ra, rb) = unsafe {
            (
                std::slice::from_raw_parts(a.add(r * astride), w),
                std::slice::from_raw_parts(b.add(r * bstride), w),
            )
        };
        s += ra
            .iter()
            .zip(rb)
            .map(|(&x, &y)| ((x as i32 - y as i32) * (x as i32 - y as i32)) as u64)
            .sum::<u64>();
    }
    s
}

/// The rows of the orthonormal 4-point DCT-II in 13-bit fixed point:
/// `round(8192 * c(k) * cos((2n + 1) k pi / 8))`, `c(0) = 1/2`,
/// `c(k) = sqrt(1/2)` otherwise.
pub(crate) const FDCT_BASIS: [[i32; 4]; 4] = [
    [4096, 4096, 4096, 4096],
    [5352, 2217, -2217, -5352],
    [4096, -4096, -4096, 4096],
    [2217, -5352, 5352, -2217],
];

/// The encoder's forward DCT: twice the orthonormal 4x4 DCT-II of the
/// residue (the scale section 14.4's inverse expects), in fixed point. A
/// row pass, kept to 5 fractional bits, then a column pass; both round to
/// nearest (half up). Residue within +-255 keeps every product in 32 bits.
pub(crate) fn fdct(input: &[i16; 16]) -> [i16; 16] {
    let mut t = [0i32; 16];
    for y in 0..4 {
        for v in 0..4 {
            let s: i32 = (0..4)
                .map(|x| input[y * 4 + x] as i32 * FDCT_BASIS[v][x])
                .sum();
            t[y * 4 + v] = (s + (1 << 7)) >> 8;
        }
    }
    let mut out = [0i16; 16];
    for u in 0..4 {
        for v in 0..4 {
            let s: i32 = (0..4).map(|y| FDCT_BASIS[u][y] * t[y * 4 + v]).sum();
            out[u * 4 + v] = ((s + (1 << 16)) >> 17) as i16;
        }
    }
    out
}

/// See [`super::QuantFn`].
pub(crate) fn quant(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]) {
    let mut lv = [0i16; 16];
    let mut deq = [0i16; 16];
    for i in (skip_dc as usize)..16 {
        let k = (i > 0) as usize;
        let v = c[i] as i32;
        let n = (v.abs() + q.round[k]) as u64;
        let l = (((n * q.recip[k] as u64) >> 32) as i32).min(2048);
        let l = if v < 0 { -l } else { l };
        lv[i] = l as i16;
        deq[i] = (l * q.step[k]) as i16;
    }
    (lv, deq)
}
