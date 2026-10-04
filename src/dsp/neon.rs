//! NEON kernels (aarch64, where NEON is always present). Each reproduces
//! its scalar reference bit for bit; `dsp::tests` checks that.
//!
//! The loop filter uses saturating signed-byte arithmetic as the SSE4.1
//! version does (see `x86`). The subpixel filters accumulate the positive
//! and the negative taps separately in unsigned 16-bit lanes (neither sum
//! can overflow), subtract with saturation at zero (a negative sum rounds
//! to 0 anyway) and narrow with a saturating rounding shift by 7, which is
//! `clamp((x + 64) >> 7, 0, 255)`.

use super::{Dsp, EdgeLimits, QuantParams, scalar};
use std::arch::aarch64::*;

/// The NEON set.
pub(crate) fn neon() -> &'static Dsp {
    static NEON: Dsp = Dsp {
        name: "neon",
        subpel,
        idct_add,
        lf_mb_h,
        lf_mb_v,
        lf_sub_h,
        lf_sub_v,
        lf_simple_h,
        lf_simple_v,
        sad,
        sse,
        fdct: scalar::fdct,
        quant,
    };
    &NEON
}

// ---------------------------------------------------------------------------
// Loop filter

struct Seg {
    p3: uint8x16_t,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    q3: uint8x16_t,
}

#[inline]
#[target_feature(enable = "neon")]
fn flip(x: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(veorq_u8(x, vdupq_n_u8(0x80)))
}

#[inline]
#[target_feature(enable = "neon")]
fn unflip(x: int8x16_t) -> uint8x16_t {
    veorq_u8(vreinterpretq_u8_s8(x), vdupq_n_u8(0x80))
}

/// Lanes where `|p0 - q0| * 2 + |p1 - q1| / 2 <= e`.
#[inline]
#[target_feature(enable = "neon")]
fn edge_mask(s: &Seg, e: u8) -> uint8x16_t {
    let a = vabdq_u8(s.p0, s.q0);
    let a = vqaddq_u8(a, a);
    let b = vshrq_n_u8::<1>(vabdq_u8(s.p1, s.q1));
    vcleq_u8(vqaddq_u8(a, b), vdupq_n_u8(e))
}

#[inline]
#[target_feature(enable = "neon")]
fn normal_masks(s: &Seg, lim: EdgeLimits) -> (uint8x16_t, uint8x16_t) {
    let d10 = vabdq_u8(s.p1, s.p0);
    let e10 = vabdq_u8(s.q1, s.q0);
    let m = vmaxq_u8(vabdq_u8(s.p3, s.p2), vabdq_u8(s.p2, s.p1));
    let m = vmaxq_u8(m, vmaxq_u8(d10, e10));
    let m = vmaxq_u8(m, vmaxq_u8(vabdq_u8(s.q3, s.q2), vabdq_u8(s.q2, s.q1)));
    let mask = vandq_u8(
        vcleq_u8(m, vdupq_n_u8(lim.interior)),
        edge_mask(s, lim.edge),
    );
    let hev = vcgtq_u8(vmaxq_u8(d10, e10), vdupq_n_u8(lim.hev));
    (mask, hev)
}

#[inline]
#[target_feature(enable = "neon")]
fn and_s8(x: int8x16_t, m: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(x), m))
}

#[inline]
#[target_feature(enable = "neon")]
fn andnot_s8(x: int8x16_t, m: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(vbicq_u8(vreinterpretq_u8_s8(x), m))
}

/// `c(base + 3 * (q0 - p0))`.
#[inline]
#[target_feature(enable = "neon")]
fn plus_3d(base: int8x16_t, ps0: int8x16_t, qs0: int8x16_t) -> int8x16_t {
    let d = vqsubq_s8(qs0, ps0);
    vqaddq_s8(vqaddq_s8(vqaddq_s8(base, d), d), d)
}

/// `c((k * w + 63) >> 7)`.
#[inline]
#[target_feature(enable = "neon")]
fn tap(w: int8x16_t, k: i16) -> int8x16_t {
    let lo = vmovl_s8(vget_low_s8(w));
    let hi = vmovl_s8(vget_high_s8(w));
    let r = vdupq_n_s16(63);
    let lo = vshrq_n_s16::<7>(vmlaq_n_s16(r, lo, k));
    let hi = vshrq_n_s16::<7>(vmlaq_n_s16(r, hi, k));
    vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_mb(s: &mut Seg, lim: EdgeLimits) {
    let (mask, hev) = normal_masks(s, lim);
    let (ps2, ps1, ps0) = (flip(s.p2), flip(s.p1), flip(s.p0));
    let (qs0, qs1, qs2) = (flip(s.q0), flip(s.q1), flip(s.q2));
    let w = and_s8(plus_3d(vqsubq_s8(ps1, qs1), ps0, qs0), mask);
    let wh = and_s8(w, hev);
    let f1 = vshrq_n_s8::<3>(vqaddq_s8(wh, vdupq_n_s8(4)));
    let f2 = vshrq_n_s8::<3>(vqaddq_s8(wh, vdupq_n_s8(3)));
    let qs0 = vqsubq_s8(qs0, f1);
    let ps0 = vqaddq_s8(ps0, f2);
    let wn = andnot_s8(w, hev);
    let a = tap(wn, 27);
    let qs0 = vqsubq_s8(qs0, a);
    let ps0 = vqaddq_s8(ps0, a);
    let a = tap(wn, 18);
    let qs1 = vqsubq_s8(qs1, a);
    let ps1 = vqaddq_s8(ps1, a);
    let a = tap(wn, 9);
    let qs2 = vqsubq_s8(qs2, a);
    let ps2 = vqaddq_s8(ps2, a);
    s.p2 = unflip(ps2);
    s.p1 = unflip(ps1);
    s.p0 = unflip(ps0);
    s.q0 = unflip(qs0);
    s.q1 = unflip(qs1);
    s.q2 = unflip(qs2);
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_sub(s: &mut Seg, lim: EdgeLimits) {
    let (mask, hev) = normal_masks(s, lim);
    let (ps1, ps0, qs0, qs1) = (flip(s.p1), flip(s.p0), flip(s.q0), flip(s.q1));
    let outer = and_s8(vqsubq_s8(ps1, qs1), hev);
    let a = and_s8(plus_3d(outer, ps0, qs0), mask);
    let f1 = vshrq_n_s8::<3>(vqaddq_s8(a, vdupq_n_s8(4)));
    let f2 = vshrq_n_s8::<3>(vqaddq_s8(a, vdupq_n_s8(3)));
    let qs0 = vqsubq_s8(qs0, f1);
    let ps0 = vqaddq_s8(ps0, f2);
    let a2 = andnot_s8(vshrq_n_s8::<1>(vqaddq_s8(f1, vdupq_n_s8(1))), hev);
    s.q1 = unflip(vqsubq_s8(qs1, a2));
    s.p1 = unflip(vqaddq_s8(ps1, a2));
    s.p0 = unflip(ps0);
    s.q0 = unflip(qs0);
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_simple(s: &mut Seg, lim: EdgeLimits) {
    let mask = edge_mask(s, lim.edge);
    let (ps1, ps0, qs0, qs1) = (flip(s.p1), flip(s.p0), flip(s.q0), flip(s.q1));
    let a = and_s8(plus_3d(vqsubq_s8(ps1, qs1), ps0, qs0), mask);
    let f1 = vshrq_n_s8::<3>(vqaddq_s8(a, vdupq_n_s8(4)));
    let f2 = vshrq_n_s8::<3>(vqaddq_s8(a, vdupq_n_s8(3)));
    s.q0 = unflip(vqsubq_s8(qs0, f1));
    s.p0 = unflip(vqaddq_s8(ps0, f2));
}

/// # Safety
/// 8 bytes readable at `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_pair(a: *const u8, b: *const u8) -> uint8x16_t {
    // SAFETY: the caller guarantees both reads.
    unsafe { vcombine_u8(vld1_u8(a), vld1_u8(b)) }
}

/// # Safety
/// 8 bytes writable at `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_pair(a: *mut u8, b: *mut u8, v: uint8x16_t) {
    // SAFETY: the caller guarantees both writes.
    unsafe {
        vst1_u8(a, vget_low_u8(v));
        vst1_u8(b, vget_high_u8(v));
    }
}

/// # Safety
/// As [`super::EdgeFn`] for a horizontal edge.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_h(a: *mut u8, b: *mut u8, stride: usize) -> Seg {
    let row = |k: isize| {
        let off = k * stride as isize;
        // SAFETY: rows p3..q3 are valid by contract.
        unsafe { load_pair(a.wrapping_offset(off), b.wrapping_offset(off)) }
    };
    Seg {
        p3: row(-4),
        p2: row(-3),
        p1: row(-2),
        p0: row(-1),
        q0: row(0),
        q1: row(1),
        q2: row(2),
        q3: row(3),
    }
}

/// # Safety
/// As [`super::EdgeFn`]; writes rows `-n..n`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_h(a: *mut u8, b: *mut u8, stride: usize, s: &Seg, n: isize) {
    let rows = [s.p2, s.p1, s.p0, s.q0, s.q1, s.q2];
    for k in -n..n {
        let off = k * stride as isize;
        // SAFETY: rows p3..q3 are valid by contract.
        unsafe {
            store_pair(
                a.wrapping_offset(off),
                b.wrapping_offset(off),
                rows[(k + 3) as usize],
            )
        };
    }
}

/// Transposes 8 vectors, each holding an 8x8 byte block in each half, in
/// both halves at once (the transpose is its own inverse).
#[inline]
#[target_feature(enable = "neon")]
fn transpose8(q: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
    let x01 = vtrnq_u8(q[0], q[1]);
    let x23 = vtrnq_u8(q[2], q[3]);
    let x45 = vtrnq_u8(q[4], q[5]);
    let x67 = vtrnq_u8(q[6], q[7]);
    let u16 = vreinterpretq_u16_u8;
    let y02 = vtrnq_u16(u16(x01.0), u16(x23.0));
    let y13 = vtrnq_u16(u16(x01.1), u16(x23.1));
    let y46 = vtrnq_u16(u16(x45.0), u16(x67.0));
    let y57 = vtrnq_u16(u16(x45.1), u16(x67.1));
    let u32 = vreinterpretq_u32_u16;
    let z04 = vtrnq_u32(u32(y02.0), u32(y46.0));
    let z15 = vtrnq_u32(u32(y13.0), u32(y57.0));
    let z26 = vtrnq_u32(u32(y02.1), u32(y46.1));
    let z37 = vtrnq_u32(u32(y13.1), u32(y57.1));
    let u8 = vreinterpretq_u8_u32;
    [
        u8(z04.0),
        u8(z15.0),
        u8(z26.0),
        u8(z37.0),
        u8(z04.1),
        u8(z15.1),
        u8(z26.1),
        u8(z37.1),
    ]
}

/// The 16 lines of a vertical edge (lines 0-7 at `a`, 8-15 at `b`).
#[inline]
#[target_feature(enable = "neon")]
fn line(a: *mut u8, b: *mut u8, stride: usize, k: usize) -> *mut u8 {
    if k < 8 {
        a.wrapping_add(k * stride)
    } else {
        b.wrapping_add((k - 8) * stride)
    }
    .wrapping_sub(4)
}

/// # Safety
/// As [`super::EdgeFn`] for a vertical edge.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_v(a: *mut u8, b: *mut u8, stride: usize) -> Seg {
    // SAFETY: p3..q3 of every line are valid by contract.
    let q: [uint8x16_t; 8] = std::array::from_fn(|k| unsafe {
        load_pair(line(a, b, stride, k), line(a, b, stride, k + 8))
    });
    let c = transpose8(q);
    Seg {
        p3: c[0],
        p2: c[1],
        p1: c[2],
        p0: c[3],
        q0: c[4],
        q1: c[5],
        q2: c[6],
        q3: c[7],
    }
}

/// # Safety
/// As [`super::EdgeFn`] for a vertical edge.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_v(a: *mut u8, b: *mut u8, stride: usize, s: &Seg) {
    let r = transpose8([s.p3, s.p2, s.p1, s.p0, s.q0, s.q1, s.q2, s.q3]);
    for (k, &v) in r.iter().enumerate() {
        // SAFETY: p3..q3 of every line are valid by contract.
        unsafe { store_pair(line(a, b, stride, k), line(a, b, stride, k + 8), v) };
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_mb_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_h(a, b, stride);
        filter_mb(&mut s, lim);
        store_h(a, b, stride, &s, 3);
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_mb_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_v(a, b, stride);
        filter_mb(&mut s, lim);
        store_v(a, b, stride, &s);
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_sub_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_h(a, b, stride);
        filter_sub(&mut s, lim);
        store_h(a, b, stride, &s, 2);
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_sub_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_v(a, b, stride);
        filter_sub(&mut s, lim);
        store_v(a, b, stride, &s);
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_simple_h(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_h(a, b, stride);
        filter_simple(&mut s, lim);
        store_h(a, b, stride, &s, 1);
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "neon")]
unsafe fn lf_simple_v(a: *mut u8, b: *mut u8, stride: usize, lim: EdgeLimits) {
    // SAFETY: forwarded contract.
    unsafe {
        let mut s = load_v(a, b, stride);
        filter_simple(&mut s, lim);
        store_v(a, b, stride, &s);
    }
}

// ---------------------------------------------------------------------------
// Subpixel interpolation

/// The magnitudes of the six taps (taps 1 and 4 are never positive, the
/// others never negative).
#[inline]
#[target_feature(enable = "neon")]
fn taps(t: &[i32; 6]) -> [uint8x8_t; 6] {
    std::array::from_fn(|i| vdup_n_u8(t[i].unsigned_abs() as u8))
}

/// Eight outputs from the samples `s[k]` (k to the right of each output's
/// first tap).
#[inline]
#[target_feature(enable = "neon")]
fn filter8(s: [uint8x8_t; 6], k: &[uint8x8_t; 6]) -> uint8x8_t {
    let pos = vmull_u8(s[2], k[2]);
    let pos = vmlal_u8(pos, s[3], k[3]);
    let pos = vmlal_u8(pos, s[0], k[0]);
    let pos = vmlal_u8(pos, s[5], k[5]);
    let neg = vmull_u8(s[1], k[1]);
    let neg = vmlal_u8(neg, s[4], k[4]);
    vqrshrn_n_u16::<7>(vqsubq_u16(pos, neg))
}

/// # Safety
/// `w` bytes writable at `p` (4, 8 or 16).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_w(p: *mut u8, lo: uint8x8_t, hi: uint8x8_t, w: usize) {
    // SAFETY: the caller guarantees `w` writable bytes.
    unsafe {
        match w {
            16 => vst1q_u8(p, vcombine_u8(lo, hi)),
            8 => vst1_u8(p, lo),
            _ => (p as *mut u32).write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(lo))),
        }
    }
}

/// # Safety
/// 32 bytes readable at each source row; `w` writable at each output row.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hpass(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    rows: usize,
    t: &[i32; 6],
) {
    let k = taps(t);
    for r in 0..rows {
        // SAFETY: the caller guarantees the reads and writes.
        unsafe {
            let p = src.add(r * sstride);
            let lo = vld1q_u8(p);
            let hi = vld1q_u8(p.add(16));
            let s = [
                lo,
                vextq_u8::<1>(lo, hi),
                vextq_u8::<2>(lo, hi),
                vextq_u8::<3>(lo, hi),
                vextq_u8::<4>(lo, hi),
                vextq_u8::<5>(lo, hi),
            ];
            let a = filter8(s.map(|v| vget_low_u8(v)), &k);
            let b = if w == 16 {
                filter8(s.map(|v| vget_high_u8(v)), &k)
            } else {
                a
            };
            store_w(dst.add(r * dstride), a, b, w);
        }
    }
}

/// # Safety
/// `w` (at least 8) readable bytes at each of `rows + 5` source rows; `w`
/// writable at each output row.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn vpass(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    rows: usize,
    t: &[i32; 6],
) {
    let k = taps(t);
    // SAFETY (loads): source rows below rows + 5, readable by contract.
    let load = |i: usize| unsafe {
        let p = src.add(i * sstride);
        if w == 16 {
            vld1q_u8(p)
        } else {
            vcombine_u8(vld1_u8(p), vdup_n_u8(0))
        }
    };
    let mut s = [load(0), load(1), load(2), load(3), load(4), load(5)];
    for r in 0..rows {
        if r > 0 {
            s = [s[1], s[2], s[3], s[4], s[5], load(r + 5)];
        }
        let a = filter8(s.map(|v| vget_low_u8(v)), &k);
        let b = if w == 16 {
            filter8(s.map(|v| vget_high_u8(v)), &k)
        } else {
            a
        };
        // SAFETY: `w` writable bytes on each output row.
        unsafe { store_w(dst.add(r * dstride), a, b, w) };
    }
}

/// # Safety
/// See [`super::SubpelFn`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn subpel(
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
    // A zero index is the identity filter; a pass with it is skipped.
    // SAFETY (all calls): the window is readable as SubpelFn requires;
    // `mid` holds 21 rows of 16.
    unsafe {
        match (fx, fy) {
            (0, 0) => {
                for r in 0..h {
                    std::ptr::copy_nonoverlapping(
                        src.add((r + 2) * sstride + 2),
                        dst.add(r * dstride),
                        w,
                    );
                }
            }
            (_, 0) => hpass(
                src.add(2 * sstride),
                sstride,
                dst,
                dstride,
                w,
                h,
                &filters[fx],
            ),
            (0, _) => vpass(src.add(2), sstride, dst, dstride, w, h, &filters[fy]),
            _ => {
                let mut mid = [0u8; 21 * 16];
                hpass(src, sstride, mid.as_mut_ptr(), 16, w, h + 5, &filters[fx]);
                vpass(mid.as_ptr(), 16, dst, dstride, w, h, &filters[fy]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse DCT

#[inline]
#[target_feature(enable = "neon")]
fn mul_sin(x: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<16>(vmulq_n_s32(x, scalar::SIN_PI8_SQRT2))
}

#[inline]
#[target_feature(enable = "neon")]
fn mul_cos(x: int32x4_t) -> int32x4_t {
    vaddq_s32(
        x,
        vshrq_n_s32::<16>(vmulq_n_s32(x, scalar::COS_PI8_SQRT2_MINUS1)),
    )
}

/// One pass of section 14.4 over four vectors (lanes independent):
/// returns (out0, out1, out2, out3) before any rounding.
#[inline]
#[target_feature(enable = "neon")]
fn pass(i0: int32x4_t, i1: int32x4_t, i2: int32x4_t, i3: int32x4_t) -> [int32x4_t; 4] {
    let a1 = vaddq_s32(i0, i2);
    let b1 = vsubq_s32(i0, i2);
    let c1 = vsubq_s32(mul_sin(i1), mul_cos(i3));
    let d1 = vaddq_s32(mul_cos(i1), mul_sin(i3));
    [
        vaddq_s32(a1, d1),
        vaddq_s32(b1, c1),
        vsubq_s32(b1, c1),
        vsubq_s32(a1, d1),
    ]
}

/// Transposes a 4x4 block of 16-bit values.
#[inline]
#[target_feature(enable = "neon")]
fn transpose4(r: [int16x4_t; 4]) -> [int16x4_t; 4] {
    let t01 = vtrn_s16(r[0], r[1]);
    let t23 = vtrn_s16(r[2], r[3]);
    let s02 = vtrn_s32(vreinterpret_s32_s16(t01.0), vreinterpret_s32_s16(t23.0));
    let s13 = vtrn_s32(vreinterpret_s32_s16(t01.1), vreinterpret_s32_s16(t23.1));
    [
        vreinterpret_s16_s32(s02.0),
        vreinterpret_s16_s32(s13.0),
        vreinterpret_s16_s32(s02.1),
        vreinterpret_s16_s32(s13.1),
    ]
}

/// # Safety
/// See [`super::IdctAddFn`].
#[target_feature(enable = "neon")]
unsafe fn idct_add(coeffs: &[i16; 16], dst: *mut u8, stride: usize) {
    // SAFETY: `coeffs` holds 16 values.
    let rows: [int16x4_t; 4] =
        std::array::from_fn(|r| unsafe { vld1_s16(coeffs.as_ptr().add(4 * r)) });
    // Vertical pass: rows as vectors, columns in lanes; the results are
    // stored in 16 bits (narrowing keeps the low 16, as `as i16` does).
    let t = pass(
        vmovl_s16(rows[0]),
        vmovl_s16(rows[1]),
        vmovl_s16(rows[2]),
        vmovl_s16(rows[3]),
    )
    .map(|v| vmovn_s32(v));
    // Horizontal pass on the transpose, then the rounding shift.
    let c = transpose4(t);
    let o = pass(
        vmovl_s16(c[0]),
        vmovl_s16(c[1]),
        vmovl_s16(c[2]),
        vmovl_s16(c[3]),
    )
    .map(|v| vmovn_s32(vshrq_n_s32::<3>(vaddq_s32(v, vdupq_n_s32(4)))));
    // o[c] holds column c of the output by row; back to rows.
    let res = transpose4(o);
    for (r, &v) in res.iter().enumerate() {
        // SAFETY: 4 bytes on each of 4 rows by the caller's guarantee.
        unsafe {
            let p = dst.add(r * stride);
            let px = vreinterpret_u8_u32(vdup_n_u32((p as *const u32).read_unaligned()));
            let sum = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(px)), vcombine_s16(v, v));
            let out = vqmovun_s16(sum);
            (p as *mut u32).write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(out)));
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder kernels

/// Loads `w` (4, 8 or 16) bytes, zero-padded.
///
/// # Safety
/// `w` bytes readable at `p`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_w(p: *const u8, w: usize) -> uint8x16_t {
    // SAFETY: the caller guarantees `w` readable bytes.
    unsafe {
        match w {
            16 => vld1q_u8(p),
            8 => vcombine_u8(vld1_u8(p), vdup_n_u8(0)),
            _ => vreinterpretq_u8_u32(vsetq_lane_u32::<0>(
                (p as *const u32).read_unaligned(),
                vdupq_n_u32(0),
            )),
        }
    }
}

/// # Safety
/// See [`super::SadFn`].
#[target_feature(enable = "neon")]
unsafe fn sad(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if !matches!(w, 4 | 8 | 16) || h > 64 {
        // SAFETY: forwarded.
        return unsafe { scalar::sad(a, astride, b, bstride, w, h) };
    }
    let mut acc = vdupq_n_u16(0);
    for r in 0..h {
        // SAFETY: `w` bytes on each row.
        unsafe {
            let x = load_w(a.add(r * astride), w);
            let y = load_w(b.add(r * bstride), w);
            acc = vpadalq_u8(acc, vabdq_u8(x, y));
        }
    }
    vaddlvq_u16(acc)
}

/// # Safety
/// See [`super::SseFn`].
#[target_feature(enable = "neon")]
unsafe fn sse(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    if !matches!(w, 4 | 8 | 16) {
        // SAFETY: forwarded.
        return unsafe { scalar::sse(a, astride, b, bstride, w, h) };
    }
    let mut acc = vdupq_n_u32(0);
    for r in 0..h {
        // SAFETY: `w` bytes on each row.
        unsafe {
            let x = load_w(a.add(r * astride), w);
            let y = load_w(b.add(r * bstride), w);
            let d = vabdq_u8(x, y);
            acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
            acc = vpadalq_u16(acc, vmull_u8(vget_high_u8(d), vget_high_u8(d)));
        }
    }
    vaddlvq_u32(acc)
}

/// The quantiser: the scalar arithmetic, two 64-bit products at a time.
fn quant(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]) {
    // SAFETY: NEON is part of every aarch64 target this module builds for.
    unsafe { quant_neon(c, q, skip_dc) }
}

#[target_feature(enable = "neon")]
fn quant_neon(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]) {
    let mut lv = [0i16; 16];
    let mut deq = [0i16; 16];
    let max = vdupq_n_u32(2048);
    for h in 0..2 {
        // SAFETY: `c` holds 16 values; 8 * h + 8 <= 16.
        let v = unsafe { vld1q_s16(c.as_ptr().add(8 * h)) };
        // |c| as unsigned 16 bits (|-32768| = 32768).
        let a = vreinterpretq_u16_s16(vabsq_s16(v));
        let mut round = [q.round[1] as u32; 8];
        let mut recip = [q.recip[1]; 8];
        let mut step = [q.step[1] as i16; 8];
        if h == 0 {
            round[0] = q.round[0] as u32;
            recip[0] = q.recip[0];
            step[0] = q.step[0] as i16;
        }
        // SAFETY: the arrays hold 8 values each.
        let (r, m, st) = unsafe {
            (
                [vld1q_u32(round.as_ptr()), vld1q_u32(round.as_ptr().add(4))],
                [vld1q_u32(recip.as_ptr()), vld1q_u32(recip.as_ptr().add(4))],
                vld1q_s16(step.as_ptr()),
            )
        };
        let n = [
            vaddq_u32(vmovl_u16(vget_low_u16(a)), r[0]),
            vaddq_u32(vmovl_u16(vget_high_u16(a)), r[1]),
        ];
        let l: [uint32x4_t; 2] = std::array::from_fn(|k| {
            let lo = vshrn_n_u64::<32>(vmull_u32(vget_low_u32(n[k]), vget_low_u32(m[k])));
            let hi = vshrn_n_u64::<32>(vmull_u32(vget_high_u32(n[k]), vget_high_u32(m[k])));
            vminq_u32(vcombine_u32(lo, hi), max)
        });
        let l = vreinterpretq_s16_u16(vcombine_u16(vmovn_u32(l[0]), vmovn_u32(l[1])));
        let mut l = vbslq_s16(vcltzq_s16(v), vnegq_s16(l), l);
        if h == 0 && skip_dc {
            l = vsetq_lane_s16::<0>(0, l);
        }
        // The low 16 bits of each product: `(l * step) as i16`.
        let dq = vmulq_s16(l, st);
        // SAFETY: 8 values from 8 * h in each output.
        unsafe {
            vst1q_s16(lv.as_mut_ptr().add(8 * h), l);
            vst1q_s16(deq.as_mut_ptr().add(8 * h), dq);
        }
    }
    (lv, deq)
}
