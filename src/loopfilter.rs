//! The loop filter's thresholds (RFC 6386 section 15.4). The filters
//! themselves are kernels in `dsp` (`dsp::scalar::filter_segment` is the
//! reference); `recon::filter_mb` applies them to a macroblock.

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
        Params {
            mb_limit: (level + 2) * 2 + interior,
            sub_limit: level * 2 + interior,
            interior,
            hev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::EdgeLimits;
    use crate::dsp::scalar::{EdgeKind, filter_segment};

    #[test]
    fn params_follow_section_15_4() {
        let p = Params::new(32, 0, true);
        assert_eq!(
            (p.interior, p.mb_limit, p.sub_limit, p.hev),
            (32, 100, 96, 1)
        );
        let p = Params::new(32, 5, false);
        assert_eq!(p.interior, 4); // 32 >> 2 = 8, capped at 9 - 5
        assert_eq!(p.hev, 2);
        assert_eq!(Params::new(1, 7, false).interior, 1);
        assert_eq!(Params::new(63, 0, false).hev, 3);
    }

    fn limits(p: &Params, edge: i32) -> EdgeLimits {
        EdgeLimits {
            edge: edge as u8,
            interior: p.interior as u8,
            hev: p.hev as u8,
        }
    }

    #[test]
    fn a_small_step_is_smoothed_and_a_large_one_kept() {
        // One segment across an edge: p3 p2 p1 p0 | q0 q1 q2 q3.
        let mut x = [100, 100, 100, 100, 108, 108, 108, 108];
        let p = Params::new(20, 0, true);
        filter_segment(EdgeKind::Mb, &mut x, limits(&p, p.mb_limit));
        assert!(x[3] > 100 && x[4] < 108, "{x:?}");
        assert!(x.windows(2).all(|w| w[0] <= w[1]), "{x:?}");
        let mut x = [10, 10, 10, 10, 200, 200, 200, 200];
        let before = x;
        filter_segment(EdgeKind::Mb, &mut x, limits(&p, p.mb_limit));
        filter_segment(EdgeKind::Sub, &mut x, limits(&p, p.sub_limit));
        filter_segment(EdgeKind::Simple, &mut x, limits(&p, p.sub_limit));
        assert_eq!(x, before);
    }

    #[test]
    fn simple_filter_rounding() {
        // p1 p0 | q0 q1 = 100 100 | 104 104: a = c(c(p1 - q1) + 3 (q0 - p0))
        // = -4 + 12 = 8, b = (8 + 3) >> 3 = 1, a = (8 + 4) >> 3 = 1.
        let lim = EdgeLimits {
            edge: 40,
            interior: 1,
            hev: 0,
        };
        let mut x = [0, 0, 100, 100, 104, 104, 0, 0];
        filter_segment(EdgeKind::Simple, &mut x, lim);
        assert_eq!(x[2..6], [100, 101, 103, 104]);
        // When (a + 3) >> 3 and (a + 4) >> 3 differ, p0 moves one less
        // than q0: 100 100 | 106 106 gives a = 18 - 6 = 12, b = 1, a = 2.
        let mut x = [0, 0, 100, 100, 106, 106, 0, 0];
        filter_segment(EdgeKind::Simple, &mut x, lim);
        assert_eq!(x[2..6], [100, 101, 104, 106]);
    }
}
