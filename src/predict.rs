//! Intra prediction (RFC 6386 section 12) and inter prediction's subpixel
//! interpolation (section 18).

use crate::tables::*;

/// The edge of an intra-predicted block: the row above (`above[..n]`, and
/// for subblocks the four pixels above and to the right after it), the
/// column to the left, and the pixel above-left. Pixels outside the frame
/// have already been replaced by section 12's 127 (above) and 129 (left).
pub(crate) struct Edge<'a> {
    pub above: &'a [u8],
    pub left: &'a [u8],
    pub top_left: u8,
    /// Whether the row above / column to the left lie inside the frame —
    /// DC_PRED averages only real pixels (section 12.2).
    pub have_above: bool,
    pub have_left: bool,
}

/// Predicts an `n`x`n` block (16 for luma, 8 for chroma) with one of the
/// whole-block modes DC_PRED, V_PRED, H_PRED, TM_PRED (sections 12.2-12.3).
pub(crate) fn predict_block(dst: &mut [u8], off: usize, stride: usize, n: usize, mode: u8, e: &Edge) {
    match mode {
        DC_PRED => {
            let shift = n.trailing_zeros();
            let sa: u32 = e.above[..n].iter().map(|&p| p as u32).sum();
            let sl: u32 = e.left[..n].iter().map(|&p| p as u32).sum();
            let v = match (e.have_above, e.have_left) {
                (true, true) => (sa + sl + (1 << shift)) >> (shift + 1),
                (true, false) => (sa + (1 << (shift - 1))) >> shift,
                (false, true) => (sl + (1 << (shift - 1))) >> shift,
                (false, false) => 128,
            } as u8;
            for r in 0..n {
                dst[off + r * stride..off + r * stride + n].fill(v);
            }
        }
        V_PRED => {
            for r in 0..n {
                dst[off + r * stride..off + r * stride + n].copy_from_slice(&e.above[..n]);
            }
        }
        H_PRED => {
            for r in 0..n {
                dst[off + r * stride..off + r * stride + n].fill(e.left[r]);
            }
        }
        _ => {
            // TM_PRED: X[r][c] = clamp(L[r] + A[c] - P).
            for r in 0..n {
                let d = e.left[r] as i32 - e.top_left as i32;
                for c in 0..n {
                    dst[off + r * stride + c] = (d + e.above[c] as i32).clamp(0, 255) as u8;
                }
            }
        }
    }
}

#[inline]
fn avg2(x: u8, y: u8) -> u8 {
    ((x as u32 + y as u32 + 1) >> 1) as u8
}

#[inline]
fn avg3(x: u8, y: u8, z: u8) -> u8 {
    ((x as u32 + 2 * y as u32 + z as u32 + 2) >> 2) as u8
}

/// Predicts one 4x4 luma subblock with a subblock mode (section 12.3).
/// `e.above` holds 8 pixels: the 4 above and the 4 above-right.
pub(crate) fn predict_subblock(dst: &mut [u8], off: usize, stride: usize, mode: u8, e: &Edge) {
    let a = e.above;
    let l = e.left;
    let p = e.top_left;
    // The edge as one run, from the bottom of the left column up through
    // the corner and along the row above: L3 L2 L1 L0 P A0 .. A7.
    let edge: [u8; 13] = [l[3], l[2], l[1], l[0], p, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
    let s3 = |i: usize| avg3(edge[i - 1], edge[i], edge[i + 1]);
    let s2 = |i: usize| avg2(edge[i], edge[i + 1]);
    let mut b = [[0u8; 4]; 4];
    match mode {
        B_DC_PRED => {
            let s: u32 = a[..4].iter().chain(&l[..4]).map(|&v| v as u32).sum();
            b = [[((s + 4) >> 3) as u8; 4]; 4];
        }
        B_TM_PRED => {
            for (r, row) in b.iter_mut().enumerate() {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = (l[r] as i32 + a[c] as i32 - p as i32).clamp(0, 255) as u8;
                }
            }
        }
        B_VE_PRED => {
            // Each column is the smoothed pixel above it (A[-1] is P).
            for c in 0..4 {
                let v = s3(5 + c);
                for row in b.iter_mut() {
                    row[c] = v;
                }
            }
        }
        B_HE_PRED => {
            // Each row is the smoothed pixel to its left; the bottom row
            // has no pixel below it and repeats L3.
            b[0] = [s3(3); 4];
            b[1] = [s3(2); 4];
            b[2] = [s3(1); 4];
            b[3] = [avg3(l[2], l[3], l[3]); 4];
        }
        B_LD_PRED => {
            // Down-left diagonals, from the smoothed row above (and right).
            for (r, row) in b.iter_mut().enumerate() {
                for (c, v) in row.iter_mut().enumerate() {
                    let k = r + c;
                    *v = if k < 6 { s3(6 + k) } else { avg3(a[6], a[7], a[7]) };
                }
            }
        }
        B_RD_PRED => {
            // Down-right diagonals through the corner.
            for (r, row) in b.iter_mut().enumerate() {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = s3(4 + c - r);
                }
            }
        }
        B_VR_PRED => {
            b[3][0] = s3(2);
            b[2][0] = s3(3);
            b[3][1] = s3(4);
            b[1][0] = s3(4);
            b[2][1] = s2(4);
            b[0][0] = s2(4);
            b[3][2] = s3(5);
            b[1][1] = s3(5);
            b[2][2] = s2(5);
            b[0][1] = s2(5);
            b[3][3] = s3(6);
            b[1][2] = s3(6);
            b[2][3] = s2(6);
            b[0][2] = s2(6);
            b[1][3] = s3(7);
            b[0][3] = s2(7);
        }
        B_VL_PRED => {
            // In terms of the row above: A[j] is edge[5 + j].
            b[0][0] = s2(5);
            b[1][0] = s3(6);
            b[2][0] = s2(6);
            b[0][1] = s2(6);
            b[1][1] = s3(7);
            b[3][0] = s3(7);
            b[2][1] = s2(7);
            b[0][2] = s2(7);
            b[3][1] = s3(8);
            b[1][2] = s3(8);
            b[2][2] = s2(8);
            b[0][3] = s2(8);
            b[3][2] = s3(9);
            b[1][3] = s3(9);
            // The last two break the pattern (section 12.3).
            b[2][3] = s3(10);
            b[3][3] = s3(11);
        }
        B_HD_PRED => {
            b[3][0] = s2(0);
            b[3][1] = s3(1);
            b[2][0] = s2(1);
            b[3][2] = s2(1);
            b[2][1] = s3(2);
            b[3][3] = s3(2);
            b[2][2] = s2(2);
            b[1][0] = s2(2);
            b[2][3] = s3(3);
            b[1][1] = s3(3);
            b[1][2] = s2(3);
            b[0][0] = s2(3);
            b[1][3] = s3(4);
            b[0][1] = s3(4);
            b[0][2] = s3(5);
            b[0][3] = s3(6);
        }
        _ => {
            // B_HU_PRED: up-right from the left column, which runs out at
            // L3 for most of the bottom.
            b[0][0] = avg2(l[0], l[1]);
            b[0][1] = avg3(l[0], l[1], l[2]);
            b[0][2] = avg2(l[1], l[2]);
            b[1][0] = b[0][2];
            b[0][3] = avg3(l[1], l[2], l[3]);
            b[1][1] = b[0][3];
            b[1][2] = avg2(l[2], l[3]);
            b[2][0] = b[1][2];
            b[1][3] = avg3(l[2], l[3], l[3]);
            b[2][1] = b[1][3];
            b[2][2] = l[3];
            b[2][3] = l[3];
            b[3] = [l[3]; 4];
        }
    }
    for (r, row) in b.iter().enumerate() {
        dst[off + r * stride..off + r * stride + 4].copy_from_slice(row);
    }
}

/// A reference plane for inter prediction: the decoded area, every
/// macroblock of it, with no border. Reads outside it take the nearest edge
/// pixel — the buffer extension of RFC 6386 section 5, carried to any
/// distance.
#[derive(Clone, Copy)]
pub(crate) struct RefPlane<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
}

/// Predicts a `w`x`h` block at (`x`, `y`) of a plane from `src` displaced
/// by (`mvx`, `mvy`) in eighths of a sample of that plane (section 18), into
/// `dst[off..]`. `filters` is the six-tap or the bilinear set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_inter(
    src: RefPlane,
    dst: &mut [u8],
    off: usize,
    stride: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mvx: i32,
    mvy: i32,
    filters: &[[i32; 6]; 8],
) {
    let ix = x + (mvx >> 3);
    let iy = y + (mvy >> 3);
    let fx = (mvx & 7) as usize;
    let fy = (mvy & 7) as usize;
    // The source window: 2 samples before and 3 after in each direction,
    // the reach of a six-tap filter.
    const MAXW: usize = 16 + 5;
    let mut win = [0u8; MAXW * MAXW];
    let (ww, wh) = (w + 5, h + 5);
    let x0 = ix - 2;
    let y0 = iy - 2;
    let inside = x0 >= 0 && y0 >= 0 && (x0 as usize + ww) <= src.width && (y0 as usize + wh) <= src.height;
    if inside {
        for r in 0..wh {
            let s = (y0 as usize + r) * src.width + x0 as usize;
            win[r * MAXW..r * MAXW + ww].copy_from_slice(&src.data[s..s + ww]);
        }
    } else {
        let maxx = src.width as i32 - 1;
        let maxy = src.height as i32 - 1;
        for r in 0..wh {
            let sy = (y0 + r as i32).clamp(0, maxy) as usize * src.width;
            for c in 0..ww {
                let sx = (x0 + c as i32).clamp(0, maxx) as usize;
                win[r * MAXW + c] = src.data[sy + sx];
            }
        }
    }
    if fx == 0 && fy == 0 {
        for r in 0..h {
            dst[off + r * stride..off + r * stride + w].copy_from_slice(&win[(r + 2) * MAXW + 2..(r + 2) * MAXW + 2 + w]);
        }
        return;
    }
    // Horizontal pass over every window row, then the vertical pass
    // (section 18.3); each rounds and saturates to 8 bits.
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
            dst[off + r * stride + c] = ((s + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge<'a>(above: &'a [u8], left: &'a [u8], p: u8) -> Edge<'a> {
        Edge { above, left, top_left: p, have_above: true, have_left: true }
    }

    #[test]
    fn whole_block_modes() {
        let above: Vec<u8> = (0..16).map(|i| 10 * i as u8).collect();
        let left: Vec<u8> = (0..16).map(|i| 200 - 5 * i as u8).collect();
        let mut buf = vec![0u8; 16 * 16];
        predict_block(&mut buf, 0, 16, 16, V_PRED, &edge(&above, &left, 50));
        assert_eq!(&buf[15 * 16..], &above[..]);
        predict_block(&mut buf, 0, 16, 16, H_PRED, &edge(&above, &left, 50));
        assert!(buf[5 * 16..6 * 16].iter().all(|&v| v == left[5]));
        predict_block(&mut buf, 0, 16, 16, TM_PRED, &edge(&above, &left, 50));
        assert_eq!(buf[3 * 16 + 4], (left[3] as i32 + above[4] as i32 - 50).clamp(0, 255) as u8);
        predict_block(&mut buf, 0, 16, 16, DC_PRED, &edge(&above, &left, 50));
        let s: u32 = above.iter().chain(&left).map(|&v| v as u32).sum();
        assert!(buf.iter().all(|&v| v as u32 == (s + 16) >> 5));
        // Top-left macroblock: no neighbours, 128.
        let e = Edge { above: &above, left: &left, top_left: 0, have_above: false, have_left: false };
        predict_block(&mut buf, 0, 16, 8, DC_PRED, &e);
        assert_eq!(buf[7 * 16 + 7], 128);
    }

    /// Every diagonal mode against the defining property of section 12.3:
    /// all pixels on one line share a value taken from the edge.
    #[test]
    fn subblock_modes_follow_their_diagonals() {
        let above = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [15, 25, 35, 45];
        let mut buf = [0u8; 16];
        let e = edge(&above, &left, 5);
        predict_subblock(&mut buf, 0, 4, B_LD_PRED, &e);
        for r in 0..4 {
            for c in 0..4 {
                // The diagonal's first pixel: on the top row or right column.
                let k = r + c;
                let (r0, c0) = if k <= 3 { (0, k) } else { (k - 3, 3) };
                assert_eq!(buf[r * 4 + c], buf[r0 * 4 + c0]);
            }
        }
        assert_eq!(buf[0], avg3(10, 20, 30));
        predict_subblock(&mut buf, 0, 4, B_RD_PRED, &e);
        assert_eq!(buf[0], avg3(left[0], 5, above[0]));
        assert_eq!(buf[5], buf[0]);
        assert_eq!(buf[15], buf[0]);
        assert_eq!(buf[3], avg3(above[1], above[2], above[3]));
        predict_subblock(&mut buf, 0, 4, B_VE_PRED, &e);
        assert_eq!(buf[0], avg3(5, 10, 20));
        assert_eq!(buf[12 + 3], avg3(30, 40, 50));
        predict_subblock(&mut buf, 0, 4, B_HE_PRED, &e);
        assert_eq!(buf[0], avg3(5, 15, 25));
        assert_eq!(buf[15], avg3(35, 45, 45));
        predict_subblock(&mut buf, 0, 4, B_HU_PRED, &e);
        assert_eq!(buf[15], 45);
        assert_eq!(buf[0], avg2(15, 25));
        predict_subblock(&mut buf, 0, 4, B_VL_PRED, &e);
        assert_eq!(buf[0], avg2(10, 20));
        assert_eq!(buf[15], avg3(60, 70, 80));
        predict_subblock(&mut buf, 0, 4, B_HD_PRED, &e);
        assert_eq!(buf[12], avg2(45, 35));
        assert_eq!(buf[3], avg3(10, 20, 30));
        predict_subblock(&mut buf, 0, 4, B_VR_PRED, &e);
        assert_eq!(buf[0], avg2(5, 10));
        assert_eq!(buf[3], avg2(30, 40));
        predict_subblock(&mut buf, 0, 4, B_DC_PRED, &e);
        assert!(buf.iter().all(|&v| v == ((100 + 120 + 4) >> 3) as u8));
    }

    #[test]
    fn whole_pel_inter_is_a_copy_and_edges_extend() {
        let data: Vec<u8> = (0..32 * 32).map(|i| (i % 251) as u8).collect();
        let src = RefPlane { data: &data, width: 32, height: 32 };
        let mut out = [0u8; 16];
        predict_inter(src, &mut out, 0, 4, 8, 8, 4, 4, 16, -8, &SIXTAP_FILTERS);
        for r in 0..4 {
            assert_eq!(&out[r * 4..r * 4 + 4], &data[(7 + r) * 32 + 10..(7 + r) * 32 + 14]);
        }
        // Far outside: every sample is the corner.
        predict_inter(src, &mut out, 0, 4, 0, 0, 4, 4, -8000, -8000, &SIXTAP_FILTERS);
        assert!(out.iter().all(|&v| v == data[0]));
    }

    #[test]
    fn subpel_inter_on_a_ramp() {
        // A horizontal ramp is reproduced by every filter at half-pel
        // (symmetric taps): the value midway between two samples.
        let data: Vec<u8> = (0..32 * 32).map(|i| (4 * (i % 32)) as u8).collect();
        let src = RefPlane { data: &data, width: 32, height: 32 };
        let mut out = [0u8; 16];
        for f in [&SIXTAP_FILTERS, &BILINEAR_FILTERS] {
            predict_inter(src, &mut out, 0, 4, 8, 8, 4, 4, 4, 0, f);
            assert_eq!(out[0], 34);
            assert_eq!(out[3], 46);
        }
    }
}
