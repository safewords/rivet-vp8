//! The loop filter (RFC 6386 section 15).
//!
//! Each function filters every segment straddling one edge: `pos` is the
//! first pixel after the edge in the first segment, `step` the distance
//! across the edge (1 for a vertical edge, the stride for a horizontal one)
//! and `along` the distance between segments.

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

/// Thresholds for one macroblock's edges (section 15.4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Params {
    /// Edge limit for macroblock edges.
    pub mb_limit: i32,
    /// Edge limit for subblock edges.
    pub sub_limit: i32,
    /// Interior difference limit.
    pub interior: i32,
    /// High edge variance threshold.
    pub hev: i32,
}

impl Params {
    /// The thresholds for `level` (1..=63) at `sharpness` (0..=7).
    pub(crate) fn new(level: u8, sharpness: u8, key_frame: bool) -> Params {
        let level = level as i32;
        let mut interior = level;
        if sharpness > 0 {
            interior >>= if sharpness > 4 { 2 } else { 1 };
            interior = interior.min(9 - sharpness as i32);
        }
        interior = interior.max(1);
        let hev = match (key_frame, level) {
            (true, 40..) => 2,
            (true, 15..) => 1,
            (false, 40..) => 3,
            (false, 20..) => 2,
            (false, 15..) => 1,
            _ => 0,
        };
        Params { mb_limit: (level + 2) * 2 + interior, sub_limit: level * 2 + interior, interior, hev }
    }
}

/// Section 15.2's `common_adjust`: moves p0 and q0 towards each other and
/// returns the adjustment `a`.
#[inline]
fn common_adjust(buf: &mut [u8], at: usize, step: usize, use_outer_taps: bool) -> i32 {
    let p1 = s(buf[at - 2 * step]);
    let p0 = s(buf[at - step]);
    let q0 = s(buf[at]);
    let q1 = s(buf[at + step]);
    let mut a = c8(if use_outer_taps { c8(p1 - q1) } else { 0 } + 3 * (q0 - p0));
    let b = c8(a + 3) >> 3;
    a = c8(a + 4) >> 3;
    buf[at] = u(q0 - a);
    buf[at - step] = u(p0 + b);
    a
}

#[inline]
fn edge_ok(buf: &[u8], at: usize, step: usize, limit: i32) -> bool {
    let p1 = buf[at - 2 * step] as i32;
    let p0 = buf[at - step] as i32;
    let q0 = buf[at] as i32;
    let q1 = buf[at + step] as i32;
    (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= limit
}

/// Section 15.3's `filter_yes`: the edge test plus every interior
/// difference within `interior`.
#[inline]
fn filter_yes(buf: &[u8], at: usize, step: usize, e: i32, i: i32) -> bool {
    let px = |k: isize| buf[(at as isize + k * step as isize) as usize] as i32;
    let (p3, p2, p1, p0, q0, q1, q2, q3) = (px(-4), px(-3), px(-2), px(-1), px(0), px(1), px(2), px(3));
    (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= e
        && (p3 - p2).abs() <= i
        && (p2 - p1).abs() <= i
        && (p1 - p0).abs() <= i
        && (q3 - q2).abs() <= i
        && (q2 - q1).abs() <= i
        && (q1 - q0).abs() <= i
}

#[inline]
fn hev(buf: &[u8], at: usize, step: usize, t: i32) -> bool {
    let p1 = buf[at - 2 * step] as i32;
    let p0 = buf[at - step] as i32;
    let q0 = buf[at] as i32;
    let q1 = buf[at + step] as i32;
    (p1 - p0).abs() > t || (q1 - q0).abs() > t
}

/// The simple filter across one edge of `n` segments (section 15.2).
pub(crate) fn simple_edge(buf: &mut [u8], pos: usize, step: usize, along: usize, n: usize, limit: i32) {
    for k in 0..n {
        let at = pos + k * along;
        if edge_ok(buf, at, step, limit) {
            common_adjust(buf, at, step, true);
        }
    }
}

/// The normal filter's subblock-edge variant (section 15.3).
pub(crate) fn subblock_edge(buf: &mut [u8], pos: usize, step: usize, along: usize, n: usize, limit: i32, p: &Params) {
    for k in 0..n {
        let at = pos + k * along;
        if filter_yes(buf, at, step, limit, p.interior) {
            let hv = hev(buf, at, step, p.hev);
            let p1 = s(buf[at - 2 * step]);
            let q1 = s(buf[at + step]);
            let a = (common_adjust(buf, at, step, hv) + 1) >> 1;
            if !hv {
                buf[at + step] = u(q1 - a);
                buf[at - 2 * step] = u(p1 + a);
            }
        }
    }
}

/// The normal filter's macroblock-edge variant (section 15.3).
pub(crate) fn mb_edge(buf: &mut [u8], pos: usize, step: usize, along: usize, n: usize, limit: i32, p: &Params) {
    for k in 0..n {
        let at = pos + k * along;
        if !filter_yes(buf, at, step, limit, p.interior) {
            continue;
        }
        if hev(buf, at, step, p.hev) {
            common_adjust(buf, at, step, true);
            continue;
        }
        let p2 = s(buf[at - 3 * step]);
        let p1 = s(buf[at - 2 * step]);
        let p0 = s(buf[at - step]);
        let q0 = s(buf[at]);
        let q1 = s(buf[at + step]);
        let q2 = s(buf[at + 2 * step]);
        let w = c8(c8(p1 - q1) + 3 * (q0 - p0));
        let a = c8((27 * w + 63) >> 7);
        buf[at] = u(q0 - a);
        buf[at - step] = u(p0 + a);
        let a = c8((18 * w + 63) >> 7);
        buf[at + step] = u(q1 - a);
        buf[at - 2 * step] = u(p1 + a);
        let a = c8((9 * w + 63) >> 7);
        buf[at + 2 * step] = u(q2 - a);
        buf[at - 3 * step] = u(p2 + a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_follow_section_15_4() {
        let p = Params::new(32, 0, true);
        assert_eq!((p.interior, p.mb_limit, p.sub_limit, p.hev), (32, 100, 96, 1));
        let p = Params::new(32, 5, false);
        assert_eq!(p.interior, 4); // 32 >> 2 = 8, capped at 9 - 5
        assert_eq!(p.hev, 2);
        assert_eq!(Params::new(1, 7, false).interior, 1);
        assert_eq!(Params::new(63, 0, false).hev, 3);
    }

    #[test]
    fn a_small_step_is_smoothed_and_a_large_one_kept() {
        // A vertical edge in one row: 8 pixels, step 1.
        let mut row = [100, 100, 100, 100, 108, 108, 108, 108];
        let p = Params::new(20, 0, true);
        mb_edge(&mut row, 4, 1, 8, 1, p.mb_limit, &p);
        assert!(row[3] > 100 && row[4] < 108, "{row:?}");
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        let mut row = [10, 10, 10, 10, 200, 200, 200, 200];
        let before = row;
        mb_edge(&mut row, 4, 1, 8, 1, p.mb_limit, &p);
        subblock_edge(&mut row, 4, 1, 8, 1, p.sub_limit, &p);
        simple_edge(&mut row, 4, 1, 8, 1, p.sub_limit);
        assert_eq!(row, before);
    }

    #[test]
    fn simple_filter_rounding() {
        // p1 p0 | q0 q1 = 100 100 | 104 104: a = c(c(p1 - q1) + 3 (q0 - p0))
        // = -4 + 12 = 8, b = (8 + 3) >> 3 = 1, a = (8 + 4) >> 3 = 1.
        let mut row = [100, 100, 104, 104];
        simple_edge(&mut row, 2, 1, 4, 1, 40);
        assert_eq!(row, [100, 101, 103, 104]);
        // When (a + 3) >> 3 and (a + 4) >> 3 differ, p0 moves one less
        // than q0: 100 100 | 106 106 gives a = 18 - 6 = 12, b = 1, a = 2.
        let mut row = [100, 100, 106, 106];
        simple_edge(&mut row, 2, 1, 4, 1, 40);
        assert_eq!(row, [100, 101, 104, 106]);
    }
}
