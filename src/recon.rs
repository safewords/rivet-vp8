//! Macroblock reconstruction and loop filtering on a picture that several
//! threads work on at once.
//!
//! A macroblock is predicted into a small work buffer that also holds the
//! pixels its prediction reads (the row above, with the four above-right,
//! and the column to the left, with section 12's 127 and 129 outside the
//! frame); its residue is added there, and the result is copied into the
//! picture. The loop filter runs on the picture in place. Rows of
//! macroblocks are reconstructed and filtered by different threads, so the
//! picture is shared through [`SharedFrame`], which hands out raw pointers;
//! the decoder's scheduling (see `decoder::Schedule`) guarantees that no
//! two threads touch the same pixel at the same time, and every access
//! here is an `unsafe fn` whose contract says so.

use crate::decoder::{FrameBuf, MbInfo, PlaneBuf};
use crate::dsp::{EdgeLimits, dsp};
use crate::loopfilter::Params;
use crate::predict::{Edge, predict_block, predict_subblock};
use crate::tables::*;
use crate::transform::add_residue;
use std::marker::PhantomData;

/// One plane of a [`SharedFrame`].
#[derive(Clone, Copy)]
pub(crate) struct SharedPlane {
    ptr: *mut u8,
    len: usize,
    pub width: usize,
    pub height: usize,
}

// SAFETY: a SharedPlane is a pointer into a plane that outlives it (the
// SharedFrame's borrow); all access through it is by `unsafe fn`s whose
// callers guarantee that threads never touch the same bytes concurrently.
unsafe impl Send for SharedPlane {}
// SAFETY: as above.
unsafe impl Sync for SharedPlane {}

impl SharedPlane {
    /// A pointer to the sample at (`x`, `y`), which must be in the plane.
    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> *mut u8 {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) outside the plane"
        );
        // In bounds: y * width + x < len.
        self.ptr.wrapping_add(y * self.width + x)
    }

    /// Copies the `w`x`h` rectangle at (`x`, `y`) into `out` (rows
    /// `ostride` apart).
    ///
    /// # Safety
    ///
    /// No other thread may write the rectangle during the call.
    #[inline]
    pub(crate) unsafe fn read(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        out: &mut [u8],
        ostride: usize,
    ) {
        assert!(x + w <= self.width && y + h <= self.height);
        assert!(h == 0 || (h - 1) * ostride + w <= out.len());
        for r in 0..h {
            let from = (y + r) * self.width + x;
            debug_assert!(from + w <= self.len);
            // SAFETY: the rectangle is inside the plane (asserted above)
            // and, by the caller's guarantee, not written concurrently.
            let row = unsafe { std::slice::from_raw_parts(self.ptr.add(from), w) };
            out[r * ostride..r * ostride + w].copy_from_slice(row);
        }
    }

    /// Copies `src` (rows `sstride` apart) into the `w`x`h` rectangle at
    /// (`x`, `y`).
    ///
    /// # Safety
    ///
    /// No other thread may read or write the rectangle during the call.
    #[inline]
    pub(crate) unsafe fn write(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        src: &[u8],
        sstride: usize,
    ) {
        assert!(x + w <= self.width && y + h <= self.height);
        assert!(h == 0 || (h - 1) * sstride + w <= src.len());
        for r in 0..h {
            let to = (y + r) * self.width + x;
            debug_assert!(to + w <= self.len);
            // SAFETY: inside the plane (asserted above); not accessed
            // concurrently, by the caller's guarantee.
            let row = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(to), w) };
            row.copy_from_slice(&src[r * sstride..r * sstride + w]);
        }
    }

    /// One sample.
    ///
    /// # Safety
    ///
    /// As [`Self::read`].
    #[inline]
    unsafe fn get(&self, x: usize, y: usize) -> u8 {
        // SAFETY: `at` checks the bounds; the caller excludes writers.
        unsafe { *self.at(x, y) }
    }
}

/// A picture whose planes several threads read and write, each its own
/// pixels (see the module documentation).
pub(crate) struct SharedFrame<'a> {
    pub planes: [SharedPlane; 3],
    _frame: PhantomData<&'a mut FrameBuf>,
}

impl<'a> SharedFrame<'a> {
    pub(crate) fn new(frame: &'a mut FrameBuf) -> Self {
        let plane = |p: &mut PlaneBuf| SharedPlane {
            ptr: p.data.as_mut_ptr(),
            len: p.data.len(),
            width: p.width,
            height: p.height,
        };
        let [y, u, v] = &mut frame.planes;
        SharedFrame {
            planes: [plane(y), plane(u), plane(v)],
            _frame: PhantomData,
        }
    }
}

/// Stride of [`MbWork::luma`].
pub(crate) const LS: usize = 32;
/// Stride of [`MbWork::u`] and [`MbWork::v`].
pub(crate) const CS: usize = 16;

/// A macroblock being reconstructed, with the pixels its intra prediction
/// reads around it. Luma: row 0 holds the pixel above-left (column 0), the
/// 16 above (1-16) and the 4 above-right (17-20); rows 1-16 hold the pixel
/// to the left (column 0) and the macroblock (1-16). Chroma likewise, 8
/// wide, without the above-right.
pub(crate) struct MbWork {
    pub luma: [u8; LS * 17],
    pub u: [u8; CS * 9],
    pub v: [u8; CS * 9],
}

/// Offset of the macroblock's first luma pixel in [`MbWork::luma`].
pub(crate) const LUMA0: usize = LS + 1;
/// Offset of the first chroma pixel in [`MbWork::u`] / [`MbWork::v`].
pub(crate) const CHROMA0: usize = CS + 1;

impl MbWork {
    pub(crate) fn new() -> Self {
        MbWork {
            luma: [0; LS * 17],
            u: [0; CS * 9],
            v: [0; CS * 9],
        }
    }

    /// Fills in the edge pixels of the macroblock at (`mbx`, `mby`) from
    /// the picture: the reconstruction (before loop filtering) of the
    /// macroblocks above, above-right and to the left, or section 12's
    /// stand-ins outside the frame (127 above, 129 to the left, 127 for the
    /// above-left corner of the top row). The above-right pixels of the
    /// last macroblock in a row repeat the last pixel above (section 12.3).
    ///
    /// # Safety
    ///
    /// No other thread may write those pixels during the call.
    pub(crate) unsafe fn load_edges(
        &mut self,
        frame: &SharedFrame,
        mbx: usize,
        mby: usize,
        mbw: usize,
    ) {
        let y = &frame.planes[0];
        let (x0, y0) = (mbx * 16, mby * 16);
        let top = &mut self.luma[..21];
        if y0 == 0 {
            top.fill(127);
        } else {
            // SAFETY (all reads here): the caller excludes writers.
            unsafe {
                top[0] = if x0 == 0 { 129 } else { y.get(x0 - 1, y0 - 1) };
                y.read(x0, y0 - 1, 16, 1, &mut top[1..17], 16);
                if mbx + 1 < mbw {
                    y.read(x0 + 16, y0 - 1, 4, 1, &mut top[17..21], 4);
                } else {
                    let last = top[16];
                    top[17..21].fill(last);
                }
            }
        }
        for r in 0..16 {
            self.luma[(r + 1) * LS] = if x0 == 0 {
                129
            } else {
                // SAFETY: as above.
                unsafe { y.get(x0 - 1, y0 + r) }
            };
        }
        for (p, buf) in [
            (&frame.planes[1], &mut self.u),
            (&frame.planes[2], &mut self.v),
        ] {
            let (x0, y0) = (mbx * 8, mby * 8);
            if y0 == 0 {
                buf[..9].fill(127);
            } else {
                // SAFETY: as above.
                unsafe {
                    buf[0] = if x0 == 0 { 129 } else { p.get(x0 - 1, y0 - 1) };
                    p.read(x0, y0 - 1, 8, 1, &mut buf[1..9], 8);
                }
            }
            for r in 0..8 {
                buf[(r + 1) * CS] = if x0 == 0 {
                    129
                } else {
                    // SAFETY: as above.
                    unsafe { p.get(x0 - 1, y0 + r) }
                };
            }
        }
    }

    /// Copies the reconstructed macroblock into the picture.
    ///
    /// # Safety
    ///
    /// No other thread may access the macroblock's pixels during the call.
    pub(crate) unsafe fn store(&self, frame: &SharedFrame, mbx: usize, mby: usize) {
        // SAFETY: forwarded.
        unsafe {
            frame.planes[0].write(mbx * 16, mby * 16, 16, 16, &self.luma[LUMA0..], LS);
            frame.planes[1].write(mbx * 8, mby * 8, 8, 8, &self.u[CHROMA0..], CS);
            frame.planes[2].write(mbx * 8, mby * 8, 8, 8, &self.v[CHROMA0..], CS);
        }
    }

    /// The edge of a whole-block mode from a work buffer: the `n` pixels
    /// above, the `n` to the left, the one above-left.
    fn whole_edge(buf: &[u8], s: usize, n: usize) -> ([u8; 16], [u8; 16], u8) {
        let mut above = [0u8; 16];
        above[..n].copy_from_slice(&buf[1..1 + n]);
        let mut left = [0u8; 16];
        for (r, l) in left[..n].iter_mut().enumerate() {
            *l = buf[(r + 1) * s];
        }
        (above, left, buf[0])
    }

    /// The luma edge of the whole-block modes: above, left, above-left.
    pub(crate) fn luma_edge(&self) -> ([u8; 16], [u8; 16], u8) {
        Self::whole_edge(&self.luma, LS, 16)
    }

    /// The edge of chroma plane 1 (U) or 2 (V).
    pub(crate) fn chroma_edge(&self, plane: usize) -> ([u8; 8], [u8; 8], u8) {
        let buf = if plane == 1 { &self.u } else { &self.v };
        let (a, l, tl) = Self::whole_edge(buf, CS, 8);
        let mut above = [0u8; 8];
        let mut left = [0u8; 8];
        above.copy_from_slice(&a[..8]);
        left.copy_from_slice(&l[..8]);
        (above, left, tl)
    }

    /// Sets the macroblock's luma to a 16x16 block (stride 16).
    pub(crate) fn put_luma(&mut self, b: &[u8; 256]) {
        for r in 0..16 {
            let o = LUMA0 + r * LS;
            self.luma[o..o + 16].copy_from_slice(&b[r * 16..r * 16 + 16]);
        }
    }

    /// Sets chroma plane 1 (U) or 2 (V) to an 8x8 block (stride 8).
    pub(crate) fn put_chroma(&mut self, plane: usize, b: &[u8; 64]) {
        let buf = if plane == 1 { &mut self.u } else { &mut self.v };
        for r in 0..8 {
            let o = CHROMA0 + r * CS;
            buf[o..o + 8].copy_from_slice(&b[r * 8..r * 8 + 8]);
        }
    }

    /// Predicts the whole luma block with one of the 16x16 modes.
    pub(crate) fn predict_luma(&mut self, mode: u8, have_above: bool, have_left: bool) {
        let (above, left, top_left) = Self::whole_edge(&self.luma, LS, 16);
        let e = Edge {
            above: &above,
            left: &left,
            top_left,
            have_above,
            have_left,
        };
        predict_block(&mut self.luma, LUMA0, LS, 16, mode, &e);
    }

    /// Predicts both chroma blocks with one of the whole-block modes.
    pub(crate) fn predict_chroma(&mut self, mode: u8, have_above: bool, have_left: bool) {
        for buf in [&mut self.u, &mut self.v] {
            let (above, left, top_left) = Self::whole_edge(&buf[..], CS, 8);
            let e = Edge {
                above: &above,
                left: &left,
                top_left,
                have_above,
                have_left,
            };
            predict_block(buf, CHROMA0, CS, 8, mode, &e);
        }
    }

    /// The edge of luma subblock `b` (section 12.3): above (4) and
    /// above-right (4), left (4), above-left. The right column's
    /// above-right pixels come from the row above the macroblock.
    pub(crate) fn subblock_edge(&self, b: usize) -> ([u8; 8], [u8; 4], u8) {
        let (bx, by) = (b & 3, b >> 2);
        // The work row above the subblock, and its first column.
        let row = 4 * by * LS;
        let col = 4 * bx;
        let mut above = [0u8; 8];
        above[..4].copy_from_slice(&self.luma[row + col + 1..row + col + 5]);
        if bx == 3 {
            above[4..].copy_from_slice(&self.luma[17..21]);
        } else {
            above[4..].copy_from_slice(&self.luma[row + col + 5..row + col + 9]);
        }
        let left = std::array::from_fn(|r| self.luma[row + (r + 1) * LS + col]);
        (above, left, self.luma[row + col])
    }

    /// The offset of luma subblock `b` in [`Self::luma`].
    #[inline]
    pub(crate) fn sub_off(b: usize) -> usize {
        LUMA0 + 4 * (b >> 2) * LS + 4 * (b & 3)
    }

    /// Predicts luma subblock `b` with subblock mode `mode`.
    pub(crate) fn predict_sub(&mut self, b: usize, mode: u8) {
        let (above, left, top_left) = self.subblock_edge(b);
        let e = Edge {
            above: &above,
            left: &left,
            top_left,
            have_above: true,
            have_left: true,
        };
        predict_subblock(&mut self.luma, Self::sub_off(b), LS, mode, &e);
    }

    /// Adds the residue of the 25 blocks (Y 0-15, U 16-19, V 20-23) whose
    /// `nonzero` flag is set; Y blocks for `y_too` only (B_PRED adds each
    /// as it goes).
    pub(crate) fn add_residues(
        &mut self,
        blocks: &[[i16; 16]; 25],
        nonzero: &[bool; 25],
        y_too: bool,
    ) {
        if y_too {
            for b in 0..16 {
                if nonzero[b] || blocks[b][0] != 0 {
                    add_residue(&blocks[b], &mut self.luma, Self::sub_off(b), LS);
                }
            }
        }
        for b in 0..4 {
            let off = CHROMA0 + 4 * (b >> 1) * CS + 4 * (b & 1);
            if nonzero[16 + b] {
                add_residue(&blocks[16 + b], &mut self.u, off, CS);
            }
            if nonzero[20 + b] {
                add_residue(&blocks[20 + b], &mut self.v, off, CS);
            }
        }
    }
}

/// Intra prediction and residue of one macroblock into `w` (sections 12,
/// 14), its edges already loaded.
pub(crate) fn reconstruct_intra(
    w: &mut MbWork,
    info: &MbInfo,
    mbx: usize,
    mby: usize,
    blocks: &[[i16; 16]; 25],
    nonzero: &[bool; 25],
) {
    if info.ymode == B_PRED {
        for b in 0..16 {
            w.predict_sub(b, info.bmodes[b]);
            if nonzero[b] {
                add_residue(&blocks[b], &mut w.luma, MbWork::sub_off(b), LS);
            }
        }
    } else {
        w.predict_luma(info.ymode, mby > 0, mbx > 0);
    }
    w.predict_chroma(info.uvmode, mby > 0, mbx > 0);
    w.add_residues(blocks, nonzero, info.ymode != B_PRED);
}

/// The loop filter thresholds for each level 0-63 at a sharpness.
pub(crate) fn filter_params(sharpness: u8, key_frame: bool) -> [Params; 64] {
    let mut params = [Params::default(); 64];
    for (l, p) in params.iter_mut().enumerate().skip(1) {
        *p = Params::new(l as u8, sharpness, key_frame);
    }
    params
}

/// Filters the edges of the macroblock at (`mbx`, `mby`) in place
/// (section 15): its left edge, its inner vertical edges, its top edge, its
/// inner horizontal edges, in that order; luma only for the simple filter.
///
/// # Safety
///
/// The filter reads and writes up to four pixels on either side of each
/// edge: no other thread may access the macroblock, the three columns to
/// its left or the three rows above it (four, for reads) during the call.
pub(crate) unsafe fn filter_mb(
    frame: &SharedFrame,
    mbx: usize,
    mby: usize,
    p: &Params,
    inner: bool,
    simple: bool,
) {
    let d = dsp();
    let lim = |edge: i32| EdgeLimits {
        edge: edge as u8,
        interior: p.interior as u8,
        hev: p.hev as u8,
    };
    let (mb, sub) = (lim(p.mb_limit), lim(p.sub_limit));
    let y = &frame.planes[0];
    let s = y.width;
    let at = y.at(mbx * 16, mby * 16);
    // Lines (vertical edges) or columns (horizontal ones) 8-15.
    let (v8, h8) = (at.wrapping_add(8 * s), at.wrapping_add(8));
    let off = |p: *mut u8, k: usize| p.wrapping_add(k);
    // SAFETY: every edge filtered lies inside the plane with four pixels on
    // each side: the left and top edges only when a macroblock lies there,
    // the inner ones inside this macroblock. The caller excludes other
    // threads from all of them.
    unsafe {
        if simple {
            if mbx > 0 {
                (d.lf_simple_v)(at, v8, s, mb);
            }
            if inner {
                for x in [4, 8, 12] {
                    (d.lf_simple_v)(off(at, x), off(v8, x), s, sub);
                }
            }
            if mby > 0 {
                (d.lf_simple_h)(at, h8, s, mb);
            }
            if inner {
                for r in [4, 8, 12] {
                    (d.lf_simple_h)(off(at, r * s), off(h8, r * s), s, sub);
                }
            }
            return;
        }
        if mbx > 0 {
            (d.lf_mb_v)(at, v8, s, mb);
        }
        if inner {
            for x in [4, 8, 12] {
                (d.lf_sub_v)(off(at, x), off(v8, x), s, sub);
            }
        }
        if mby > 0 {
            (d.lf_mb_h)(at, h8, s, mb);
        }
        if inner {
            for r in [4, 8, 12] {
                (d.lf_sub_h)(off(at, r * s), off(h8, r * s), s, sub);
            }
        }
        // Chroma: U is segments 0-7, V 8-15 of each edge.
        let (pu, pv) = (&frame.planes[1], &frame.planes[2]);
        let cs = pu.width;
        let (u, v) = (pu.at(mbx * 8, mby * 8), pv.at(mbx * 8, mby * 8));
        if mbx > 0 {
            (d.lf_mb_v)(u, v, cs, mb);
        }
        if inner {
            (d.lf_sub_v)(off(u, 4), off(v, 4), cs, sub);
        }
        if mby > 0 {
            (d.lf_mb_h)(u, v, cs, mb);
        }
        if inner {
            (d.lf_sub_h)(off(u, 4 * cs), off(v, 4 * cs), cs, sub);
        }
    }
}
