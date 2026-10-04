//! SSE4.1 and AVX2 kernels (x86-64). Each reproduces its scalar reference
//! bit for bit; `dsp::tests` checks that on random and extreme inputs.
//!
//! The loop filter works on 16 segments at once in signed 8-bit lanes with
//! saturating arithmetic, which is exactly section 15's `c()` clamping: a
//! sum such as `c(c(p1 - q1) + 3 * (q0 - p0))` is built by saturating adds
//! of `q0 - p0` (itself saturated) three times, and once a partial sum
//! saturates every later term pushes the same way, as the exact sum would.
//! The subpixel filters use `pmaddubsw` on (pixel, tap) pairs ordered so no
//! pair overflows 16 bits, and `pmulhrsw` by 256 for `(x + 64) >> 7`.

use super::{Dsp, EdgeLimits, QuantParams, scalar};
use std::arch::x86_64::*;

/// The SSE4.1 set, if the CPU has SSE4.1.
pub(crate) fn sse41() -> Option<&'static Dsp> {
    static SSE41: Dsp = Dsp {
        name: "sse4.1",
        subpel: subpel_sse41,
        idct_add: idct_add_sse41,
        lf_mb_h,
        lf_mb_v,
        lf_sub_h,
        lf_sub_v,
        lf_simple_h,
        lf_simple_v,
        sad: sad_sse41,
        sse: sse_sse41,
        fdct: fdct_entry_sse41,
        quant: quant_entry_sse41,
    };
    std::is_x86_feature_detected!("sse4.1").then_some(&SSE41)
}

/// The AVX2 set (the SSE4.1 kernels where AVX2 adds nothing), if the CPU
/// has AVX2.
pub(crate) fn avx2() -> Option<&'static Dsp> {
    static AVX2: Dsp = Dsp {
        name: "avx2",
        subpel: subpel_avx2,
        idct_add: idct_add_sse41,
        lf_mb_h,
        lf_mb_v,
        lf_sub_h,
        lf_sub_v,
        lf_simple_h,
        lf_simple_v,
        sad: sad_avx2,
        sse: sse_avx2,
        fdct: fdct_entry_sse41,
        quant: quant_entry_sse41,
    };
    (std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("sse4.1"))
        .then_some(&AVX2)
}

// ---------------------------------------------------------------------------
// Small helpers

#[inline]
#[target_feature(enable = "sse4.1")]
fn abd(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

/// Arithmetic right shift of signed bytes by 3.
#[inline]
#[target_feature(enable = "sse4.1")]
fn sra3_epi8(x: __m128i) -> __m128i {
    // Each byte doubled into a 16-bit lane is x * 257, whose high byte is
    // x; shifting by 8 + n leaves floor(x / 2^n) (the low byte adds less
    // than one step).
    let lo = _mm_srai_epi16::<11>(_mm_unpacklo_epi8(x, x));
    let hi = _mm_srai_epi16::<11>(_mm_unpackhi_epi8(x, x));
    _mm_packs_epi16(lo, hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sra1_epi8(x: __m128i) -> __m128i {
    let lo = _mm_srai_epi16::<9>(_mm_unpacklo_epi8(x, x));
    let hi = _mm_srai_epi16::<9>(_mm_unpackhi_epi8(x, x));
    _mm_packs_epi16(lo, hi)
}

/// `c((k * w + 63) >> 7)` for signed bytes `w`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn tap_epi8(w: __m128i, k: i16) -> __m128i {
    let lo = _mm_srai_epi16::<8>(_mm_unpacklo_epi8(w, w));
    let hi = _mm_srai_epi16::<8>(_mm_unpackhi_epi8(w, w));
    let (kk, r) = (_mm_set1_epi16(k), _mm_set1_epi16(63));
    let lo = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(lo, kk), r));
    let hi = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(hi, kk), r));
    _mm_packs_epi16(lo, hi)
}

/// The 8 pixels p3..q3 of 16 segments, one vector per position.
struct Seg {
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
}

/// Lanes (0xff) where `|p0 - q0| * 2 + |p1 - q1| / 2 <= e`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn edge_mask(s: &Seg, e: u8) -> __m128i {
    let a = abd(s.p0, s.q0);
    let a = _mm_adds_epu8(a, a);
    let b = _mm_and_si128(_mm_srli_epi16::<1>(abd(s.p1, s.q1)), _mm_set1_epi8(0x7f));
    let sum = _mm_adds_epu8(a, b);
    _mm_cmpeq_epi8(
        _mm_subs_epu8(sum, _mm_set1_epi8(e as i8)),
        _mm_setzero_si128(),
    )
}

/// The normal filter's mask (`filter_yes`) and the high edge variance mask.
#[inline]
#[target_feature(enable = "sse4.1")]
fn normal_masks(s: &Seg, lim: EdgeLimits) -> (__m128i, __m128i) {
    let d10 = abd(s.p1, s.p0);
    let e10 = abd(s.q1, s.q0);
    let m = _mm_max_epu8(abd(s.p3, s.p2), abd(s.p2, s.p1));
    let m = _mm_max_epu8(m, _mm_max_epu8(d10, e10));
    let m = _mm_max_epu8(m, _mm_max_epu8(abd(s.q3, s.q2), abd(s.q2, s.q1)));
    let zero = _mm_setzero_si128();
    let interior = _mm_cmpeq_epi8(_mm_subs_epu8(m, _mm_set1_epi8(lim.interior as i8)), zero);
    let mask = _mm_and_si128(interior, edge_mask(s, lim.edge));
    let h = _mm_max_epu8(d10, e10);
    let not_hev = _mm_cmpeq_epi8(_mm_subs_epu8(h, _mm_set1_epi8(lim.hev as i8)), zero);
    (mask, _mm_xor_si128(not_hev, _mm_set1_epi8(-1)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn flip(x: __m128i) -> __m128i {
    _mm_xor_si128(x, _mm_set1_epi8(-128))
}

/// `c(base + 3 * (q0 - p0))` in signed lanes.
#[inline]
#[target_feature(enable = "sse4.1")]
fn plus_3d(base: __m128i, ps0: __m128i, qs0: __m128i) -> __m128i {
    let d = _mm_subs_epi8(qs0, ps0);
    _mm_adds_epi8(_mm_adds_epi8(_mm_adds_epi8(base, d), d), d)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn filter_mb(s: &mut Seg, lim: EdgeLimits) {
    let (mask, hev) = normal_masks(s, lim);
    let (ps2, ps1, ps0) = (flip(s.p2), flip(s.p1), flip(s.p0));
    let (qs0, qs1, qs2) = (flip(s.q0), flip(s.q1), flip(s.q2));
    let w = _mm_and_si128(plus_3d(_mm_subs_epi8(ps1, qs1), ps0, qs0), mask);
    // High edge variance: section 15.2's common_adjust on p0 and q0.
    let wh = _mm_and_si128(w, hev);
    let f1 = sra3_epi8(_mm_adds_epi8(wh, _mm_set1_epi8(4)));
    let f2 = sra3_epi8(_mm_adds_epi8(wh, _mm_set1_epi8(3)));
    let qs0 = _mm_subs_epi8(qs0, f1);
    let ps0 = _mm_adds_epi8(ps0, f2);
    // Otherwise the three taps of 27, 18 and 9 sevenths... of 128ths.
    let wn = _mm_andnot_si128(hev, w);
    let a = tap_epi8(wn, 27);
    let qs0 = _mm_subs_epi8(qs0, a);
    let ps0 = _mm_adds_epi8(ps0, a);
    let a = tap_epi8(wn, 18);
    let qs1 = _mm_subs_epi8(qs1, a);
    let ps1 = _mm_adds_epi8(ps1, a);
    let a = tap_epi8(wn, 9);
    let qs2 = _mm_subs_epi8(qs2, a);
    let ps2 = _mm_adds_epi8(ps2, a);
    s.p2 = flip(ps2);
    s.p1 = flip(ps1);
    s.p0 = flip(ps0);
    s.q0 = flip(qs0);
    s.q1 = flip(qs1);
    s.q2 = flip(qs2);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn filter_sub(s: &mut Seg, lim: EdgeLimits) {
    let (mask, hev) = normal_masks(s, lim);
    let (ps1, ps0, qs0, qs1) = (flip(s.p1), flip(s.p0), flip(s.q0), flip(s.q1));
    let outer = _mm_and_si128(_mm_subs_epi8(ps1, qs1), hev);
    let a = _mm_and_si128(plus_3d(outer, ps0, qs0), mask);
    let f1 = sra3_epi8(_mm_adds_epi8(a, _mm_set1_epi8(4)));
    let f2 = sra3_epi8(_mm_adds_epi8(a, _mm_set1_epi8(3)));
    let qs0 = _mm_subs_epi8(qs0, f1);
    let ps0 = _mm_adds_epi8(ps0, f2);
    let a2 = _mm_andnot_si128(hev, sra1_epi8(_mm_adds_epi8(f1, _mm_set1_epi8(1))));
    let qs1 = _mm_subs_epi8(qs1, a2);
    let ps1 = _mm_adds_epi8(ps1, a2);
    s.p1 = flip(ps1);
    s.p0 = flip(ps0);
    s.q0 = flip(qs0);
    s.q1 = flip(qs1);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn filter_simple(s: &mut Seg, lim: EdgeLimits) {
    let mask = edge_mask(s, lim.edge);
    let (ps1, ps0, qs0, qs1) = (flip(s.p1), flip(s.p0), flip(s.q0), flip(s.q1));
    let a = _mm_and_si128(plus_3d(_mm_subs_epi8(ps1, qs1), ps0, qs0), mask);
    let f1 = sra3_epi8(_mm_adds_epi8(a, _mm_set1_epi8(4)));
    let f2 = sra3_epi8(_mm_adds_epi8(a, _mm_set1_epi8(3)));
    s.q0 = flip(_mm_subs_epi8(qs0, f1));
    s.p0 = flip(_mm_adds_epi8(ps0, f2));
}

/// Loads 16 lanes: 8 bytes at `a`, 8 at `b`.
///
/// # Safety
/// 8 bytes at `a` and at `b` must be readable.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load_pair(a: *const u8, b: *const u8) -> __m128i {
    // SAFETY: the caller guarantees both 8-byte reads.
    unsafe {
        if b == a.wrapping_add(8) {
            _mm_loadu_si128(a as *const __m128i)
        } else {
            _mm_unpacklo_epi64(
                _mm_loadl_epi64(a as *const __m128i),
                _mm_loadl_epi64(b as *const __m128i),
            )
        }
    }
}

/// Stores 16 lanes: 8 bytes to `a`, 8 to `b`.
///
/// # Safety
/// 8 bytes at `a` and at `b` must be writable.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn store_pair(a: *mut u8, b: *mut u8, v: __m128i) {
    // SAFETY: the caller guarantees both 8-byte writes.
    unsafe {
        _mm_storel_epi64(a as *mut __m128i, v);
        _mm_storel_epi64(b as *mut __m128i, _mm_unpackhi_epi64(v, v));
    }
}

/// Horizontal edge: rows p3..q3 across, 16 columns along.
///
/// # Safety
/// As [`super::EdgeFn`].
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load_h(a: *mut u8, b: *mut u8, stride: usize) -> Seg {
    let row = |k: isize| {
        let off = k * stride as isize;
        // SAFETY: rows p3 (k = -4) to q3 (k = 3) are valid by contract.
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
/// As [`super::EdgeFn`]; writes rows `-n..n` (n = 3, 2 or 1).
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn store_h(a: *mut u8, b: *mut u8, stride: usize, s: &Seg, n: isize) {
    let rows = [s.p2, s.p1, s.p0, s.q0, s.q1, s.q2];
    for k in -n..n {
        let off = k * stride as isize;
        // SAFETY: rows p3..q3 are valid by contract; k is within -3..3.
        unsafe {
            store_pair(
                a.wrapping_offset(off),
                b.wrapping_offset(off),
                rows[(k + 3) as usize],
            )
        };
    }
}

/// Transposes 16 rows of 8 bytes (rows 0-7 at `a`, 8-15 at `b`, starting
/// at p3) into 8 vectors of 16 lanes.
///
/// # Safety
/// As [`super::EdgeFn`] for a vertical edge.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load_v(a: *mut u8, b: *mut u8, stride: usize) -> Seg {
    let r = |k: usize| {
        let p = if k < 8 {
            a.wrapping_add(k * stride)
        } else {
            b.wrapping_add((k - 8) * stride)
        };
        // SAFETY: p3..q3 of every line are valid by contract.
        unsafe { _mm_loadl_epi64(p.wrapping_sub(4) as *const __m128i) }
    };
    let b0 = _mm_unpacklo_epi8(r(0), r(1));
    let b1 = _mm_unpacklo_epi8(r(2), r(3));
    let b2 = _mm_unpacklo_epi8(r(4), r(5));
    let b3 = _mm_unpacklo_epi8(r(6), r(7));
    let b4 = _mm_unpacklo_epi8(r(8), r(9));
    let b5 = _mm_unpacklo_epi8(r(10), r(11));
    let b6 = _mm_unpacklo_epi8(r(12), r(13));
    let b7 = _mm_unpacklo_epi8(r(14), r(15));
    // Columns 0-3 / 4-7 of rows 0-3, 4-7, 8-11, 12-15.
    let c0 = _mm_unpacklo_epi16(b0, b1);
    let c1 = _mm_unpackhi_epi16(b0, b1);
    let c2 = _mm_unpacklo_epi16(b2, b3);
    let c3 = _mm_unpackhi_epi16(b2, b3);
    let c4 = _mm_unpacklo_epi16(b4, b5);
    let c5 = _mm_unpackhi_epi16(b4, b5);
    let c6 = _mm_unpacklo_epi16(b6, b7);
    let c7 = _mm_unpackhi_epi16(b6, b7);
    // Column pairs of rows 0-7 and 8-15.
    let d0 = _mm_unpacklo_epi32(c0, c2);
    let d1 = _mm_unpackhi_epi32(c0, c2);
    let d2 = _mm_unpacklo_epi32(c1, c3);
    let d3 = _mm_unpackhi_epi32(c1, c3);
    let e0 = _mm_unpacklo_epi32(c4, c6);
    let e1 = _mm_unpackhi_epi32(c4, c6);
    let e2 = _mm_unpacklo_epi32(c5, c7);
    let e3 = _mm_unpackhi_epi32(c5, c7);
    Seg {
        p3: _mm_unpacklo_epi64(d0, e0),
        p2: _mm_unpackhi_epi64(d0, e0),
        p1: _mm_unpacklo_epi64(d1, e1),
        p0: _mm_unpackhi_epi64(d1, e1),
        q0: _mm_unpacklo_epi64(d2, e2),
        q1: _mm_unpackhi_epi64(d2, e2),
        q2: _mm_unpacklo_epi64(d3, e3),
        q3: _mm_unpackhi_epi64(d3, e3),
    }
}

/// The inverse of [`load_v`]: writes all 8 bytes of every line.
///
/// # Safety
/// As [`super::EdgeFn`] for a vertical edge.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn store_v(a: *mut u8, b: *mut u8, stride: usize, s: &Seg) {
    let f0 = _mm_unpacklo_epi8(s.p3, s.p2);
    let f1 = _mm_unpackhi_epi8(s.p3, s.p2);
    let f2 = _mm_unpacklo_epi8(s.p1, s.p0);
    let f3 = _mm_unpackhi_epi8(s.p1, s.p0);
    let f4 = _mm_unpacklo_epi8(s.q0, s.q1);
    let f5 = _mm_unpackhi_epi8(s.q0, s.q1);
    let f6 = _mm_unpacklo_epi8(s.q2, s.q3);
    let f7 = _mm_unpackhi_epi8(s.q2, s.q3);
    // Rows 0-3 / 4-7 (columns 0-3, then 4-7), and 8-15 likewise.
    let g0 = _mm_unpacklo_epi16(f0, f2);
    let g1 = _mm_unpackhi_epi16(f0, f2);
    let g2 = _mm_unpacklo_epi16(f4, f6);
    let g3 = _mm_unpackhi_epi16(f4, f6);
    let g4 = _mm_unpacklo_epi16(f1, f3);
    let g5 = _mm_unpackhi_epi16(f1, f3);
    let g6 = _mm_unpacklo_epi16(f5, f7);
    let g7 = _mm_unpackhi_epi16(f5, f7);
    let rows = [
        _mm_unpacklo_epi32(g0, g2), // rows 0, 1
        _mm_unpackhi_epi32(g0, g2), // rows 2, 3
        _mm_unpacklo_epi32(g1, g3), // rows 4, 5
        _mm_unpackhi_epi32(g1, g3), // rows 6, 7
        _mm_unpacklo_epi32(g4, g6), // rows 8, 9
        _mm_unpackhi_epi32(g4, g6),
        _mm_unpacklo_epi32(g5, g7),
        _mm_unpackhi_epi32(g5, g7),
    ];
    for (i, &v) in rows.iter().enumerate() {
        for half in 0..2 {
            let k = 2 * i + half;
            let p = if k < 8 {
                a.wrapping_add(k * stride)
            } else {
                b.wrapping_add((k - 8) * stride)
            };
            let x = if half == 0 {
                v
            } else {
                _mm_unpackhi_epi64(v, v)
            };
            // SAFETY: p3..q3 of every line are valid by contract.
            unsafe { _mm_storel_epi64(p.wrapping_sub(4) as *mut __m128i, x) };
        }
    }
}

/// # Safety
/// See [`super::EdgeFn`].
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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

/// The six taps as three `pmaddubsw` operands: (t2, t1), (t3, t4), (t0,
/// t5), each pair low byte first. Paired this way no pair's products
/// exceed 16 bits (|t2|, |t3| <= 123 with t1, t4 <= 0), and the running
/// sum, saturated, still rounds to the right byte: t0 and t5 are never
/// negative, so a sum that saturates high only grows.
#[inline]
fn tap_pairs(t: &[i32; 6]) -> [i16; 3] {
    let pair = |lo: i32, hi: i32| ((hi as i8 as u8 as u16) << 8 | lo as i8 as u8 as u16) as i16;
    [pair(t[2], t[1]), pair(t[3], t[4]), pair(t[0], t[5])]
}

/// Six-tap sums of 8 or 16 outputs: `s[k]` holds the samples k to the
/// right of each output's first tap.
#[inline]
#[target_feature(enable = "sse4.1")]
fn taps_lo(s: &[__m128i; 6], k: &[__m128i; 3]) -> __m128i {
    let a = _mm_maddubs_epi16(_mm_unpacklo_epi8(s[2], s[1]), k[0]);
    let b = _mm_maddubs_epi16(_mm_unpacklo_epi8(s[3], s[4]), k[1]);
    let c = _mm_maddubs_epi16(_mm_unpacklo_epi8(s[0], s[5]), k[2]);
    _mm_mulhrs_epi16(_mm_adds_epi16(_mm_adds_epi16(a, b), c), _mm_set1_epi16(256))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn taps_hi(s: &[__m128i; 6], k: &[__m128i; 3]) -> __m128i {
    let a = _mm_maddubs_epi16(_mm_unpackhi_epi8(s[2], s[1]), k[0]);
    let b = _mm_maddubs_epi16(_mm_unpackhi_epi8(s[3], s[4]), k[1]);
    let c = _mm_maddubs_epi16(_mm_unpackhi_epi8(s[0], s[5]), k[2]);
    _mm_mulhrs_epi16(_mm_adds_epi16(_mm_adds_epi16(a, b), c), _mm_set1_epi16(256))
}

/// Stores the low `w` (4, 8 or 16) bytes of `v`.
///
/// # Safety
/// `w` bytes at `p` must be writable.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn store_w(p: *mut u8, v: __m128i, w: usize) {
    // SAFETY: the caller guarantees `w` writable bytes.
    unsafe {
        match w {
            16 => _mm_storeu_si128(p as *mut __m128i, v),
            8 => _mm_storel_epi64(p as *mut __m128i, v),
            _ => (p as *mut i32).write_unaligned(_mm_cvtsi128_si32(v)),
        }
    }
}

/// The horizontal pass over `rows` rows: `w` outputs per row from `w + 5`
/// samples starting at `src`.
///
/// # Safety
/// 32 bytes (16 for `w` < 16) readable at each source row; `w` writable at
/// each destination row.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn hpass_sse41(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    rows: usize,
    taps: &[i32; 6],
) {
    let k = tap_pairs(taps).map(|x| _mm_set1_epi16(x));
    for r in 0..rows {
        let p = src.wrapping_add(r * sstride);
        // SAFETY: the caller guarantees the reads and writes.
        unsafe {
            let lo = _mm_loadu_si128(p as *const __m128i);
            if w == 16 {
                let hi = _mm_loadu_si128(p.add(16) as *const __m128i);
                let s = [
                    lo,
                    _mm_alignr_epi8::<1>(hi, lo),
                    _mm_alignr_epi8::<2>(hi, lo),
                    _mm_alignr_epi8::<3>(hi, lo),
                    _mm_alignr_epi8::<4>(hi, lo),
                    _mm_alignr_epi8::<5>(hi, lo),
                ];
                let v = _mm_packus_epi16(taps_lo(&s, &k), taps_hi(&s, &k));
                _mm_storeu_si128(dst.add(r * dstride) as *mut __m128i, v);
            } else {
                let s = [
                    lo,
                    _mm_srli_si128::<1>(lo),
                    _mm_srli_si128::<2>(lo),
                    _mm_srli_si128::<3>(lo),
                    _mm_srli_si128::<4>(lo),
                    _mm_srli_si128::<5>(lo),
                ];
                let t = taps_lo(&s, &k);
                store_w(dst.add(r * dstride), _mm_packus_epi16(t, t), w);
            }
        }
    }
}

/// The vertical pass: `rows` outputs per column from `rows + 5` source
/// rows starting at `src`.
///
/// # Safety
/// `w` (8 when `w` is 4) readable bytes at each of the `rows + 5` source
/// rows; `w` writable at each destination row.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn vpass_sse41(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    rows: usize,
    taps: &[i32; 6],
) {
    let k = tap_pairs(taps).map(|x| _mm_set1_epi16(x));
    // SAFETY (for every load below): row i of the source is readable for
    // `w` (or 8) bytes by the caller's guarantee, i < rows + 5.
    let load = |i: usize| unsafe {
        let p = src.add(i * sstride) as *const __m128i;
        if w == 16 {
            _mm_loadu_si128(p)
        } else {
            _mm_loadl_epi64(p)
        }
    };
    let mut s = [load(0), load(1), load(2), load(3), load(4), load(5)];
    for r in 0..rows {
        if r > 0 {
            s = [s[1], s[2], s[3], s[4], s[5], load(r + 5)];
        }
        let lo = taps_lo(&s, &k);
        let hi = if w == 16 { taps_hi(&s, &k) } else { lo };
        // SAFETY: the caller guarantees `w` writable bytes on each row.
        unsafe { store_w(dst.add(r * dstride), _mm_packus_epi16(lo, hi), w) };
    }
}

/// # Safety
/// `w` bytes on each of `h` rows of both.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn copy_block(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    w: usize,
    h: usize,
) {
    for r in 0..h {
        // SAFETY: the caller guarantees the rows.
        unsafe { std::ptr::copy_nonoverlapping(src.add(r * sstride), dst.add(r * dstride), w) };
    }
}

/// # Safety
/// See [`super::SubpelFn`].
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn subpel_sse41(
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
    // A zero index is the identity filter (128 at the centre tap), so a
    // pass with it is skipped; the result is the same bytes.
    // SAFETY (all calls): the window is readable as SubpelFn requires;
    // `mid` holds 21 rows of 16.
    unsafe {
        match (fx, fy) {
            (0, 0) => copy_block(src.add(2 * sstride + 2), sstride, dst, dstride, w, h),
            (_, 0) => hpass_sse41(
                src.add(2 * sstride),
                sstride,
                dst,
                dstride,
                w,
                h,
                &filters[fx],
            ),
            (0, _) => vpass_sse41(src.add(2), sstride, dst, dstride, w, h, &filters[fy]),
            _ => {
                let mut mid = [0u8; 21 * 16];
                hpass_sse41(src, sstride, mid.as_mut_ptr(), 16, w, h + 5, &filters[fx]);
                vpass_sse41(mid.as_ptr(), 16, dst, dstride, w, h, &filters[fy]);
            }
        }
    }
}

/// Six-tap sums of two 16-output rows (one per 128-bit lane).
#[inline]
#[target_feature(enable = "avx2")]
fn taps_avx2(s: &[__m256i; 6], k: &[__m256i; 3]) -> __m256i {
    let r = _mm256_set1_epi16(256);
    let a = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s[2], s[1]), k[0]);
    let b = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s[3], s[4]), k[1]);
    let c = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(s[0], s[5]), k[2]);
    let lo = _mm256_mulhrs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(a, b), c), r);
    let a = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s[2], s[1]), k[0]);
    let b = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s[3], s[4]), k[1]);
    let c = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(s[0], s[5]), k[2]);
    let hi = _mm256_mulhrs_epi16(_mm256_adds_epi16(_mm256_adds_epi16(a, b), c), r);
    _mm256_packus_epi16(lo, hi)
}

/// [`hpass_sse41`] for `w` = 16, two rows at a time.
///
/// # Safety
/// As [`hpass_sse41`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hpass16_avx2(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    rows: usize,
    taps: &[i32; 6],
) {
    let k = tap_pairs(taps).map(|x| _mm256_set1_epi16(x));
    let mut r = 0;
    while r + 2 <= rows {
        let p = src.wrapping_add(r * sstride);
        // SAFETY: rows r and r + 1 are readable for 32 bytes and the
        // destination rows writable for 16, by the caller's guarantee.
        unsafe {
            let ld = |q: *const u8| {
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(q as *const __m128i)),
                    _mm_loadu_si128(q.add(sstride) as *const __m128i),
                )
            };
            let lo = ld(p);
            let hi = ld(p.add(16));
            let s = [
                lo,
                _mm256_alignr_epi8::<1>(hi, lo),
                _mm256_alignr_epi8::<2>(hi, lo),
                _mm256_alignr_epi8::<3>(hi, lo),
                _mm256_alignr_epi8::<4>(hi, lo),
                _mm256_alignr_epi8::<5>(hi, lo),
            ];
            let v = taps_avx2(&s, &k);
            let d = dst.add(r * dstride);
            _mm_storeu_si128(d as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(
                d.add(dstride) as *mut __m128i,
                _mm256_extracti128_si256::<1>(v),
            );
        }
        r += 2;
    }
    if r < rows {
        // SAFETY: forwarded.
        unsafe {
            hpass_sse41(
                src.add(r * sstride),
                sstride,
                dst.add(r * dstride),
                dstride,
                16,
                rows - r,
                taps,
            )
        };
    }
}

/// [`vpass_sse41`] for `w` = 16, two rows at a time.
///
/// # Safety
/// As [`vpass_sse41`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn vpass16_avx2(
    src: *const u8,
    sstride: usize,
    dst: *mut u8,
    dstride: usize,
    rows: usize,
    taps: &[i32; 6],
) {
    let k = tap_pairs(taps).map(|x| _mm256_set1_epi16(x));
    // SAFETY (loads): source rows i and i + 1 with i + 1 < rows + 5.
    let ld = |i: usize| unsafe {
        let p = src.add(i * sstride);
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p as *const __m128i)),
            _mm_loadu_si128(p.add(sstride) as *const __m128i),
        )
    };
    let mut r = 0;
    while r + 2 <= rows {
        let s = [ld(r), ld(r + 1), ld(r + 2), ld(r + 3), ld(r + 4), ld(r + 5)];
        let v = taps_avx2(&s, &k);
        // SAFETY: destination rows r, r + 1 are writable for 16 bytes.
        unsafe {
            let d = dst.add(r * dstride);
            _mm_storeu_si128(d as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeu_si128(
                d.add(dstride) as *mut __m128i,
                _mm256_extracti128_si256::<1>(v),
            );
        }
        r += 2;
    }
    if r < rows {
        // SAFETY: forwarded.
        unsafe {
            vpass_sse41(
                src.add(r * sstride),
                sstride,
                dst.add(r * dstride),
                dstride,
                16,
                rows - r,
                taps,
            )
        };
    }
}

/// # Safety
/// See [`super::SubpelFn`].
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn subpel_avx2(
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
    // SAFETY (all calls): as in subpel_sse41.
    unsafe {
        if w != 16 {
            return subpel_sse41(src, sstride, dst, dstride, w, h, fx, fy, filters);
        }
        match (fx, fy) {
            (0, 0) => copy_block(src.add(2 * sstride + 2), sstride, dst, dstride, w, h),
            (_, 0) => hpass16_avx2(src.add(2 * sstride), sstride, dst, dstride, h, &filters[fx]),
            (0, _) => vpass16_avx2(src.add(2), sstride, dst, dstride, h, &filters[fy]),
            _ => {
                let mut mid = [0u8; 21 * 16];
                hpass16_avx2(src, sstride, mid.as_mut_ptr(), 16, h + 5, &filters[fx]);
                vpass16_avx2(mid.as_ptr(), 16, dst, dstride, h, &filters[fy]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse DCT

/// `(x * k) >> 16` for the two constants of section 14.4, exactly: 35468
/// is above `i16::MAX`, so `x * 35468 >> 16` is computed as `x + (x *
/// (35468 - 65536) >> 16)`, which is the same integer.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mulhi(x: __m128i, k: i32) -> __m128i {
    _mm_mulhi_epi16(x, _mm_set1_epi16(k as i16))
}

/// # Safety
/// See [`super::IdctAddFn`].
#[target_feature(enable = "sse4.1")]
unsafe fn idct_add_sse41(coeffs: &[i16; 16], dst: *mut u8, stride: usize) {
    const SIN_LO: i32 = scalar::SIN_PI8_SQRT2 - 65536;
    const COS: i32 = scalar::COS_PI8_SQRT2_MINUS1;
    // SAFETY: `coeffs` is 32 bytes.
    let (r01, r23) = unsafe {
        let p = coeffs.as_ptr() as *const __m128i;
        (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)))
    };
    // Vertical pass on rows as vectors (columns in lanes), 16-bit wrapping
    // like section 14's 16-bit stores.
    let (i0, i4) = (r01, _mm_unpackhi_epi64(r01, r01));
    let (i8, i12) = (r23, _mm_unpackhi_epi64(r23, r23));
    let a1 = _mm_add_epi16(i0, i8);
    let b1 = _mm_sub_epi16(i0, i8);
    let sin4 = _mm_add_epi16(mulhi(i4, SIN_LO), i4);
    let cos4 = _mm_add_epi16(mulhi(i4, COS), i4);
    let sin12 = _mm_add_epi16(mulhi(i12, SIN_LO), i12);
    let cos12 = _mm_add_epi16(mulhi(i12, COS), i12);
    let c1 = _mm_sub_epi16(sin4, cos12);
    let d1 = _mm_add_epi16(cos4, sin12);
    let t0 = _mm_add_epi16(a1, d1);
    let t3 = _mm_sub_epi16(a1, d1);
    let t1 = _mm_add_epi16(b1, c1);
    let t2 = _mm_sub_epi16(b1, c1);
    // Transpose: columns of tmp become rows (one per vector).
    let u01 = _mm_unpacklo_epi16(t0, t1); // t00 t10 t01 t11 t02 t12 t03 t13
    let u23 = _mm_unpacklo_epi16(t2, t3);
    let c01 = _mm_unpacklo_epi32(u01, u23); // col 0 | col 1
    let c23 = _mm_unpackhi_epi32(u01, u23); // col 2 | col 3
    // Horizontal pass in 32 bits (the sums before the final shift may
    // exceed 16 bits; section 14.4 shifts them unwrapped).
    let ext = |v: __m128i| _mm_cvtepi16_epi32(v);
    let hi = |v: __m128i| _mm_unpackhi_epi64(v, v);
    let (x0, x1, x2, x3) = (ext(c01), ext(hi(c01)), ext(c23), ext(hi(c23)));
    let sin = |v: __m128i, x: __m128i| _mm_add_epi32(ext(mulhi(v, SIN_LO)), x);
    let cos = |v: __m128i, x: __m128i| _mm_add_epi32(ext(mulhi(v, COS)), x);
    let a1 = _mm_add_epi32(x0, x2);
    let b1 = _mm_sub_epi32(x0, x2);
    let c1 = _mm_sub_epi32(sin(hi(c01), x1), cos(hi(c23), x3));
    let d1 = _mm_add_epi32(cos(hi(c01), x1), sin(hi(c23), x3));
    let four = _mm_set1_epi32(4);
    let o0 = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_add_epi32(a1, d1), four));
    let o3 = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_sub_epi32(a1, d1), four));
    let o1 = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_add_epi32(b1, c1), four));
    let o2 = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_sub_epi32(b1, c1), four));
    // Each output fits 16 bits (|out| < 2^14), so packing saturates nothing.
    let x = _mm_packs_epi32(o0, o1); // column 0 | column 1, by row
    let y = _mm_packs_epi32(o2, o3);
    let lo = _mm_unpacklo_epi16(x, y); // r0c0 r0c2 r1c0 r1c2 ...
    let hi = _mm_unpackhi_epi16(x, y); // r0c1 r0c3 ...
    let rows01 = _mm_unpacklo_epi16(lo, hi);
    let rows23 = _mm_unpackhi_epi16(lo, hi);
    // SAFETY: 4 bytes on each of 4 rows by the caller's guarantee.
    unsafe {
        let ld = |r: usize| (dst.add(r * stride) as *const i32).read_unaligned();
        let px01 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(
            _mm_cvtsi32_si128(ld(0)),
            _mm_cvtsi32_si128(ld(1)),
        ));
        let px23 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(
            _mm_cvtsi32_si128(ld(2)),
            _mm_cvtsi32_si128(ld(3)),
        ));
        let v = _mm_packus_epi16(_mm_add_epi16(px01, rows01), _mm_add_epi16(px23, rows23));
        for r in 0..4 {
            let word = _mm_cvtsi128_si32(match r {
                0 => v,
                1 => _mm_srli_si128::<4>(v),
                2 => _mm_srli_si128::<8>(v),
                _ => _mm_srli_si128::<12>(v),
            });
            (dst.add(r * stride) as *mut i32).write_unaligned(word);
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder kernels

/// # Safety
/// See [`super::SadFn`].
#[target_feature(enable = "sse4.1")]
unsafe fn sad_sse41(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut acc = _mm_setzero_si128();
    for r in 0..h {
        // SAFETY: `w` bytes on each row by the caller's guarantee.
        let (x, y) = unsafe {
            let (pa, pb) = (a.add(r * astride), b.add(r * bstride));
            match w {
                16 => (
                    _mm_loadu_si128(pa as *const __m128i),
                    _mm_loadu_si128(pb as *const __m128i),
                ),
                8 => (
                    _mm_loadl_epi64(pa as *const __m128i),
                    _mm_loadl_epi64(pb as *const __m128i),
                ),
                4 => (
                    _mm_cvtsi32_si128((pa as *const i32).read_unaligned()),
                    _mm_cvtsi32_si128((pb as *const i32).read_unaligned()),
                ),
                _ => return scalar::sad(a, astride, b, bstride, w, h),
            }
        };
        acc = _mm_add_epi64(acc, _mm_sad_epu8(x, y));
    }
    (_mm_cvtsi128_si64(acc) + _mm_extract_epi64::<1>(acc)) as u32
}

/// # Safety
/// See [`super::SadFn`].
#[target_feature(enable = "avx2")]
unsafe fn sad_avx2(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if w != 16 || !h.is_multiple_of(2) {
        // SAFETY: forwarded.
        return unsafe { sad_sse41(a, astride, b, bstride, w, h) };
    }
    let mut acc = _mm256_setzero_si256();
    for r in (0..h).step_by(2) {
        // SAFETY: rows r and r + 1, 16 bytes each.
        unsafe {
            let ld = |p: *const u8, s: usize| {
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(p as *const __m128i)),
                    _mm_loadu_si128(p.add(s) as *const __m128i),
                )
            };
            let x = ld(a.add(r * astride), astride);
            let y = ld(b.add(r * bstride), bstride);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(x, y));
        }
    }
    let s = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    (_mm_cvtsi128_si64(s) + _mm_extract_epi64::<1>(s)) as u32
}

/// # Safety
/// See [`super::SseFn`].
#[target_feature(enable = "sse4.1")]
unsafe fn sse_sse41(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut acc = _mm_setzero_si128();
    let sq = |x: __m128i, y: __m128i| {
        let d = _mm_sub_epi16(_mm_cvtepu8_epi16(x), _mm_cvtepu8_epi16(y));
        _mm_madd_epi16(d, d)
    };
    for r in 0..h {
        // SAFETY: `w` bytes on each row by the caller's guarantee.
        unsafe {
            let (pa, pb) = (a.add(r * astride), b.add(r * bstride));
            match w {
                16 => {
                    let x = _mm_loadu_si128(pa as *const __m128i);
                    let y = _mm_loadu_si128(pb as *const __m128i);
                    acc = _mm_add_epi32(acc, sq(x, y));
                    acc =
                        _mm_add_epi32(acc, sq(_mm_unpackhi_epi64(x, x), _mm_unpackhi_epi64(y, y)));
                }
                8 => {
                    let x = _mm_loadl_epi64(pa as *const __m128i);
                    let y = _mm_loadl_epi64(pb as *const __m128i);
                    acc = _mm_add_epi32(acc, sq(x, y));
                }
                4 => {
                    let x = _mm_cvtsi32_si128((pa as *const i32).read_unaligned());
                    let y = _mm_cvtsi32_si128((pb as *const i32).read_unaligned());
                    acc = _mm_add_epi32(acc, sq(x, y));
                }
                _ => return scalar::sse(a, astride, b, bstride, w, h),
            }
        }
    }
    // At most 16 x 16 squares of 255^2 spread over 4 lanes: no overflow.
    let s = _mm_add_epi32(acc, _mm_unpackhi_epi64(acc, acc));
    let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
    _mm_cvtsi128_si32(s) as u32 as u64
}

/// # Safety
/// See [`super::SseFn`].
#[target_feature(enable = "avx2")]
unsafe fn sse_avx2(
    a: *const u8,
    astride: usize,
    b: *const u8,
    bstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    if w != 16 {
        // SAFETY: forwarded.
        return unsafe { sse_sse41(a, astride, b, bstride, w, h) };
    }
    let mut acc = _mm256_setzero_si256();
    for r in 0..h {
        // SAFETY: 16 bytes on each row.
        unsafe {
            let x = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(r * astride) as *const __m128i));
            let y = _mm256_cvtepu8_epi16(_mm_loadu_si128(b.add(r * bstride) as *const __m128i));
            let d = _mm256_sub_epi16(x, y);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
        }
    }
    let s = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
    _mm_cvtsi128_si32(s) as u32 as u64
}

/// The forward DCT: the scalar arithmetic with `pmaddwd` dot products.
#[target_feature(enable = "sse4.1")]
fn fdct_sse41(input: &[i16; 16]) -> [i16; 16] {
    let basis = &scalar::FDCT_BASIS;
    // Row v of the basis, twice: dots with two 4-sample rows at once.
    let k = |v: usize| {
        let b = basis[v];
        _mm_setr_epi16(
            b[0] as i16,
            b[1] as i16,
            b[2] as i16,
            b[3] as i16,
            b[0] as i16,
            b[1] as i16,
            b[2] as i16,
            b[3] as i16,
        )
    };
    let ks = [k(0), k(1), k(2), k(3)];
    // SAFETY: `input` is 32 bytes.
    let (r01, r23) = unsafe {
        let p = input.as_ptr() as *const __m128i;
        (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)))
    };
    // dots(x)[v] = [row a . basis v, row b . basis v] for the two rows in x,
    // as T[a][0..4] | T[b][0..4] after the horizontal adds.
    let rows = |x: __m128i| {
        let m: [__m128i; 4] = std::array::from_fn(|v| _mm_madd_epi16(x, ks[v]));
        // hadd(m0, m1) = [a.b0, b.b0, a.b1, b.b1]
        let h01 = _mm_hadd_epi32(m[0], m[1]);
        let h23 = _mm_hadd_epi32(m[2], m[3]);
        // Row a: (a.b0, a.b1, a.b2, a.b3), row b likewise.
        let a = _mm_unpacklo_epi64(
            _mm_shuffle_epi32::<0b10_00_10_00>(h01),
            _mm_shuffle_epi32::<0b10_00_10_00>(h23),
        );
        let b = _mm_unpacklo_epi64(
            _mm_shuffle_epi32::<0b11_01_11_01>(h01),
            _mm_shuffle_epi32::<0b11_01_11_01>(h23),
        );
        let r = _mm_set1_epi32(1 << 7);
        (
            _mm_srai_epi32::<8>(_mm_add_epi32(a, r)),
            _mm_srai_epi32::<8>(_mm_add_epi32(b, r)),
        )
    };
    let (t0, t1) = rows(r01);
    let (t2, t3) = rows(r23);
    // Column pass: out[u][v] = sum_y basis[u][y] * t[y][v]. Transpose t
    // (each |t| < 2^15) to columns and take the same dot products.
    let t01 = _mm_packs_epi32(t0, t1); // t row 0 | row 1
    let t23 = _mm_packs_epi32(t2, t3);
    let lo = _mm_unpacklo_epi16(t01, t23); // t00 t20 t01 t21 t02 t22 t03 t23
    let hi = _mm_unpackhi_epi16(t01, t23); // t10 t30 ...
    let c01 = _mm_unpacklo_epi16(lo, hi); // t00 t10 t20 t30 | t01 t11 t21 t31
    let c23 = _mm_unpackhi_epi16(lo, hi);
    let cols = |x: __m128i| {
        let m: [__m128i; 4] = std::array::from_fn(|u| _mm_madd_epi16(x, ks[u]));
        let h01 = _mm_hadd_epi32(m[0], m[1]); // [c.b0, d.b0, c.b1, d.b1]
        let h23 = _mm_hadd_epi32(m[2], m[3]);
        let a = _mm_unpacklo_epi64(
            _mm_shuffle_epi32::<0b10_00_10_00>(h01),
            _mm_shuffle_epi32::<0b10_00_10_00>(h23),
        );
        let b = _mm_unpacklo_epi64(
            _mm_shuffle_epi32::<0b11_01_11_01>(h01),
            _mm_shuffle_epi32::<0b11_01_11_01>(h23),
        );
        let r = _mm_set1_epi32(1 << 16);
        (
            _mm_srai_epi32::<17>(_mm_add_epi32(a, r)),
            _mm_srai_epi32::<17>(_mm_add_epi32(b, r)),
        )
    };
    // out column v = (out[0][v], out[1][v], out[2][v], out[3][v]).
    let (o0, o1) = cols(c01);
    let (o2, o3) = cols(c23);
    let x = _mm_packs_epi32(o0, o1);
    let y = _mm_packs_epi32(o2, o3);
    let lo = _mm_unpacklo_epi16(x, y);
    let hi = _mm_unpackhi_epi16(x, y);
    let mut out = [0i16; 16];
    // SAFETY: `out` is 32 bytes.
    unsafe {
        let p = out.as_mut_ptr() as *mut __m128i;
        _mm_storeu_si128(p, _mm_unpacklo_epi16(lo, hi));
        _mm_storeu_si128(p.add(1), _mm_unpackhi_epi16(lo, hi));
    }
    out
}

fn fdct_entry_sse41(input: &[i16; 16]) -> [i16; 16] {
    // SAFETY: this entry is only in the SSE4.1 and AVX2 sets, chosen after
    // detecting SSE4.1.
    unsafe { fdct_sse41(input) }
}

/// The quantiser: the scalar arithmetic, four coefficients per pair of
/// 64-bit multiplies.
#[target_feature(enable = "sse4.1")]
fn quant_sse41(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]) {
    // SAFETY: `c` is 32 bytes.
    let (c0, c1) = unsafe {
        let p = c.as_ptr() as *const __m128i;
        (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)))
    };
    let r_ac = _mm_set1_epi32(q.round[1]);
    let r_dc = _mm_insert_epi32::<0>(r_ac, q.round[0]);
    let m_ac = _mm_set1_epi32(q.recip[1] as i32);
    let m_dc = _mm_insert_epi32::<0>(m_ac, q.recip[0] as i32);
    let max = _mm_set1_epi32(2048);
    // (n * m) >> 32 for four unsigned 32-bit lanes.
    let mulhi = |n: __m128i, m: __m128i| {
        let even = _mm_srli_epi64::<32>(_mm_mul_epu32(n, m));
        let odd = _mm_mul_epu32(_mm_srli_epi64::<32>(n), _mm_srli_epi64::<32>(m));
        _mm_blend_epi16::<0b1100_1100>(even, odd)
    };
    let levels = |v: __m128i, dc: bool| {
        // |c| as unsigned 16 bits (|-32768| = 32768), widened.
        let a = _mm_abs_epi16(v);
        let lo = _mm_cvtepu16_epi32(a);
        let hi = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(a, a));
        let (r0, m0) = if dc { (r_dc, m_dc) } else { (r_ac, m_ac) };
        let l0 = _mm_min_epi32(mulhi(_mm_add_epi32(lo, r0), m0), max);
        let l1 = _mm_min_epi32(mulhi(_mm_add_epi32(hi, r_ac), m_ac), max);
        // Levels are at most 2048; the sign is the coefficient's (a zero
        // coefficient has level 0, as round < step).
        _mm_sign_epi16(_mm_packs_epi32(l0, l1), v)
    };
    let mut l0 = levels(c0, true);
    let l1 = levels(c1, false);
    if skip_dc {
        l0 = _mm_insert_epi16::<0>(l0, 0);
    }
    let s_ac = _mm_set1_epi16(q.step[1] as i16);
    let s_dc = _mm_insert_epi16::<0>(s_ac, q.step[0]);
    // The low 16 bits of each product: `(l * step) as i16`.
    let d0 = _mm_mullo_epi16(l0, s_dc);
    let d1 = _mm_mullo_epi16(l1, s_ac);
    let mut lv = [0i16; 16];
    let mut deq = [0i16; 16];
    // SAFETY: both arrays are 32 bytes.
    unsafe {
        let (pl, pd) = (
            lv.as_mut_ptr() as *mut __m128i,
            deq.as_mut_ptr() as *mut __m128i,
        );
        _mm_storeu_si128(pl, l0);
        _mm_storeu_si128(pl.add(1), l1);
        _mm_storeu_si128(pd, d0);
        _mm_storeu_si128(pd.add(1), d1);
    }
    (lv, deq)
}

fn quant_entry_sse41(c: &[i16; 16], q: &QuantParams, skip_dc: bool) -> ([i16; 16], [i16; 16]) {
    // SAFETY: only in the SSE4.1 and AVX2 sets, chosen after detecting
    // SSE4.1.
    unsafe { quant_sse41(c, q, skip_dc) }
}
