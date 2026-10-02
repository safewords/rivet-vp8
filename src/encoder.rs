//! The encoder: key frames and inter frames at a fixed quantiser.
//!
//! Each macroblock is decided, quantised and reconstructed in raster order
//! (intra prediction reads the reconstruction, as the decoder's does), then
//! the frame is written: header, modes, tokens. Token probabilities are
//! re-estimated from the frame's own statistics and sent as updates where
//! that saves bits. Finally the frame is decoded by an internal
//! [`Decoder`], whose loop-filtered output is the reference for the next
//! frame — so the encoder's references are the decoder's by construction.
//!
//! Decisions:
//! - key frames: each macroblock tries the four 16x16 luma modes and B_PRED
//!   (every subblock trying all ten subblock modes), picking the lowest
//!   rate-distortion cost; chroma takes the mode with the least prediction
//!   error.
//! - inter frames: a whole-sample motion search (coarse grid, then
//!   refinement) around the zero vector and the neighbours' vectors, then a
//!   quarter-sample refinement, against the last frame; the macroblock takes
//!   ZEROMV, NEARESTMV, NEARMV, NEWMV or a 16x16 intra mode, whichever has
//!   the least prediction error plus rate. SPLITMV, golden and altref
//!   references are not used.

use crate::boolcoder::{BoolEncoder, cost, tree_cost, tree_path};
use crate::decoder::{
    Decoder, Dequant, FrameBuf, INTRA, LAST, MbInfo, Mv, PlaneBuf, Probs, QuantIndices, chroma_mvs,
    clamp_mv, find_near_mvs, implied_bmode, subblock_edge, whole_edge,
};
use crate::error::{Result, config};
use crate::frame::Frame;
use crate::predict::{Edge, predict_block, predict_inter, predict_subblock};
use crate::tables::*;
use crate::transform::{add_residue, forward_dct, forward_wht, inverse_wht};

/// Encoder settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// Frame width, 1 to 16383.
    pub width: u32,
    /// Frame height, 1 to 16383.
    pub height: u32,
    /// Quantiser index, 0 (finest) to 127 (coarsest): the rate control. The
    /// index selects the dequantisation factors of RFC 6386 section 14.1;
    /// every frame uses it.
    pub quantizer: u8,
    /// A key frame every this many frames; 1 makes every frame a key frame,
    /// 0 only the first.
    pub keyframe_interval: u32,
    /// Loop filter level, 0 (off) to 63; `None` derives one from the
    /// quantiser.
    pub loop_filter_level: Option<u8>,
    /// Loop filter sharpness, 0 to 7.
    pub sharpness: u8,
    /// Motion search range in whole luma samples (1 to 64).
    pub search_range: u8,
}

impl Default for Config {
    /// 0x0 (set the size), quantiser 40, a key frame every 120 frames,
    /// automatic loop filter, search range 16.
    fn default() -> Self {
        Config {
            width: 0,
            height: 0,
            quantizer: 40,
            keyframe_interval: 120,
            loop_filter_level: None,
            sharpness: 0,
            search_range: 16,
        }
    }
}

/// A VP8 encoder: [`Frame`]s in, compressed frames out (one per input,
/// every one shown), ready for an IVF file ([`crate::ivf::IvfWriter`]) or
/// any other container.
pub struct Encoder {
    cfg: Config,
    mbw: usize,
    mbh: usize,
    frames: u64,
    force_key: bool,
    /// Decodes what the encoder writes, so the references match.
    dec: Decoder,
}

/// What the encoder decided for one macroblock.
#[derive(Clone)]
struct MbCode {
    info: MbInfo,
    /// Quantised levels, raster order: Y 0-15, U 16-19, V 20-23, Y2 24.
    levels: [[i16; 16]; 25],
    /// The base vector of NEWMV.
    best: Mv,
    /// The mode probabilities (from the neighbour census) of an inter
    /// macroblock.
    mode_probs: [u8; 4],
}

/// Token counts by [plane type][band][context][node][bit].
type Counts = [[[[[u32; 2]; 11]; 3]; 8]; 4];

/// Where a token bool's probability comes from.
#[derive(Clone, Copy)]
enum Slot {
    /// coeff_probs[type][band][ctx][node].
    Tree(usize, usize, usize),
    /// A fixed probability (extra bits, sign).
    Fixed(u8),
}

impl Encoder {
    /// An encoder for frames of `cfg.width` x `cfg.height`.
    pub fn new(cfg: Config) -> Result<Encoder> {
        if cfg.width == 0 || cfg.height == 0 || cfg.width > 16383 || cfg.height > 16383 {
            return Err(config(format!(
                "frame size {}x{}: VP8 codes 1 to 16383 samples each way",
                cfg.width, cfg.height
            )));
        }
        if cfg.quantizer > 127 {
            return Err(config(format!("quantizer {} (0 to 127)", cfg.quantizer)));
        }
        if cfg.loop_filter_level.is_some_and(|l| l > 63) {
            return Err(config("loop filter level above 63"));
        }
        if cfg.sharpness > 7 {
            return Err(config("sharpness above 7"));
        }
        if cfg.search_range == 0 || cfg.search_range > 64 {
            return Err(config("search range must be 1 to 64"));
        }
        let mbw = (cfg.width as usize).div_ceil(16);
        let mbh = (cfg.height as usize).div_ceil(16);
        Ok(Encoder {
            cfg,
            mbw,
            mbh,
            frames: 0,
            force_key: false,
            dec: Decoder::new(),
        })
    }

    /// The settings.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Makes the next frame a key frame.
    pub fn force_key_frame(&mut self) {
        self.force_key = true;
    }

    /// The decoder's reconstruction of the last frame encoded — what a
    /// decoder will show for it.
    pub fn reconstruction(&self) -> Option<Frame> {
        self.dec.last_frame()
    }

    /// Encodes one frame; returns the compressed frame.
    pub fn encode(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(config(format!(
                "frame is {}x{}, the encoder was configured for {}x{}",
                frame.width, frame.height, self.cfg.width, self.cfg.height
            )));
        }
        let interval = self.cfg.keyframe_interval as u64;
        let key = self.frames == 0
            || self.force_key
            || (interval > 0 && self.frames.is_multiple_of(interval));
        self.force_key = false;
        let src = pad_source(frame, self.mbw, self.mbh);
        let bytes = self.encode_frame(&src, key)?;
        // Decode it: the decoder's reconstruction is the next reference.
        self.dec.decode_frame(&bytes)?;
        self.frames += 1;
        Ok(bytes)
    }

    fn encode_frame(&mut self, src: &FrameBuf, key: bool) -> Result<Vec<u8>> {
        let (mbw, mbh) = (self.mbw, self.mbh);
        let q = self.cfg.quantizer as i32;
        let dq = Dequant::new(
            &QuantIndices {
                y_ac: q,
                ..Default::default()
            },
            q,
        );
        let lambda = (dq.y[1] * dq.y[1]) as f64 / 32.0;
        let probs = if key {
            Probs::default()
        } else {
            self.dec.probs().clone()
        };

        let mut recon = FrameBuf {
            planes: [
                PlaneBuf {
                    data: vec![0; mbw * 16 * mbh * 16],
                    width: mbw * 16,
                    height: mbh * 16,
                },
                PlaneBuf {
                    data: vec![0; mbw * 8 * mbh * 8],
                    width: mbw * 8,
                    height: mbh * 8,
                },
                PlaneBuf {
                    data: vec![0; mbw * 8 * mbh * 8],
                    width: mbw * 8,
                    height: mbh * 8,
                },
            ],
        };
        let stride = mbw + 1;
        let mut mbs = vec![MbInfo::default(); stride * (mbh + 1)];
        let mut codes = Vec::with_capacity(mbw * mbh);
        let reference = if key {
            None
        } else {
            Some(self.dec.last_reference().clone())
        };
        for mby in 0..mbh {
            for mbx in 0..mbw {
                let idx = (mby + 1) * stride + mbx + 1;
                let code = match &reference {
                    None => self.code_intra_mb(
                        src, &mut recon, mbx, mby, &dq, lambda, &mbs, idx, true, &probs,
                    ),
                    Some(r) => self.code_inter_mb(
                        src, &mut recon, r, mbx, mby, &dq, lambda, &mbs, idx, &probs,
                    ),
                };
                mbs[idx] = code.info;
                codes.push(code);
            }
        }
        self.write_frame(key, &codes, &mbs, probs)
    }

    /// Codes a macroblock with intra prediction: on key frames trying the
    /// 16x16 modes and B_PRED, in inter frames the 16x16 modes.
    #[allow(clippy::too_many_arguments)]
    fn code_intra_mb(
        &self,
        src: &FrameBuf,
        recon: &mut FrameBuf,
        mbx: usize,
        mby: usize,
        dq: &Dequant,
        lambda: f64,
        mbs: &[MbInfo],
        idx: usize,
        key: bool,
        probs: &Probs,
    ) -> MbCode {
        let (x0, y0) = (mbx * 16, mby * 16);
        let s = block_of(&src.planes[0], x0, y0, 16);
        let mut info = MbInfo::default();
        let mut levels = [[0i16; 16]; 25];

        // The 16x16 modes, each coded in full.
        let (above, left, top_left) = whole_edge(&recon.planes[0], x0, y0, 16);
        let e = Edge {
            above: &above,
            left: &left,
            top_left,
            have_above: y0 > 0,
            have_left: x0 > 0,
        };
        let mode_cost = |m: u8| {
            if key {
                tree_cost(&KF_YMODE_TREE, &KF_YMODE_PROBS, 0, m)
            } else {
                tree_cost(&YMODE_TREE, &probs.ymode, 0, m)
            }
        };
        // (cost, mode, levels, reconstruction)
        type Candidate = (f64, u8, [[i16; 16]; 25], [u8; 256]);
        let mut best16: Option<Candidate> = None;
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            let mut pred = [0u8; 256];
            predict_block(&mut pred, 0, 16, 16, mode, &e);
            let mut lv = [[0i16; 16]; 25];
            let (rec, rate) = code_luma_y2(&s, &pred, dq, &mut lv);
            let j = sse(&s, &rec) as f64 + lambda * (rate + mode_cost(mode)) as f64 / 256.0;
            if best16.as_ref().is_none_or(|b| j < b.0) {
                best16 = Some((j, mode, lv, rec));
            }
        }
        let (j16, mode16, lv16, rec16) = best16.expect("four modes tried");

        // B_PRED, on key frames: subblock by subblock in the frame itself,
        // since each subblock predicts from the ones before it.
        let mut use_bpred = false;
        if key {
            let saved: [u8; 256] = block_of(&recon.planes[0], x0, y0, 16);
            let above_modes = mbs[idx - (self.mbw + 1)].bmodes;
            let left_modes = mbs[idx - 1].bmodes;
            let mut bmodes = [B_DC_PRED; 16];
            let mut lvb = [[0i16; 16]; 25];
            let mut jb = lambda * mode_cost(B_PRED) as f64 / 256.0;
            let p = &mut recon.planes[0];
            let stride = p.width;
            for b in 0..16 {
                let (bx, by) = (b & 3, b >> 2);
                let off = (y0 + 4 * by) * stride + x0 + 4 * bx;
                let sb: [u8; 16] =
                    std::array::from_fn(|i| s[(4 * by + i / 4) * 16 + 4 * bx + i % 4]);
                let (ab, lb, tl) = subblock_edge(p, mbx, mby, self.mbw, b);
                let ctx_a = if b < 4 {
                    above_modes[b + 12]
                } else {
                    bmodes[b - 4]
                };
                let ctx_l = if b & 3 == 0 {
                    left_modes[b + 3]
                } else {
                    bmodes[b - 1]
                };
                let mprobs = &KF_BMODE_PROBS[ctx_a as usize][ctx_l as usize];
                let e = Edge {
                    above: &ab,
                    left: &lb,
                    top_left: tl,
                    have_above: true,
                    have_left: true,
                };
                let mut best: Option<(f64, u8, [i16; 16], [i16; 16])> = None;
                for m in 0..10u8 {
                    let mut pred = [0u8; 16];
                    predict_subblock(&mut pred, 0, 4, m, &e);
                    let (lv, deq) = code_4x4(&sb, &pred, dq.y, 0);
                    let mut rec = pred;
                    add_residue(&deq, &mut rec, 0, 4);
                    let rate = block_rate(&lv, 0) + tree_cost(&BMODE_TREE, mprobs, 0, m);
                    let j = sse(&sb, &rec) as f64 + lambda * rate as f64 / 256.0;
                    if best.as_ref().is_none_or(|x| j < x.0) {
                        best = Some((j, m, lv, deq));
                    }
                }
                let (j, m, lv, deq) = best.expect("ten modes tried");
                jb += j;
                bmodes[b] = m;
                lvb[b] = lv;
                predict_subblock(&mut p.data, off, stride, m, &e);
                add_residue(&deq, &mut p.data, off, stride);
                if jb > j16 {
                    break;
                }
            }
            if jb < j16 {
                use_bpred = true;
                info.ymode = B_PRED;
                info.bmodes = bmodes;
                levels[..16].copy_from_slice(&lvb[..16]);
            } else {
                put_block(&mut recon.planes[0], x0, y0, 16, &saved);
            }
        }
        if !use_bpred {
            info.ymode = mode16;
            info.bmodes = [implied_bmode(mode16); 16];
            levels[..16].copy_from_slice(&lv16[..16]);
            levels[24] = lv16[24];
            put_block(&mut recon.planes[0], x0, y0, 16, &rec16);
        }

        // Chroma: the mode with the least prediction error.
        let su = block_of(&src.planes[1], mbx * 8, mby * 8, 8);
        let sv = block_of(&src.planes[2], mbx * 8, mby * 8, 8);
        let mut best_uv = (u64::MAX, DC_PRED, [0u8; 64], [0u8; 64]);
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            let mut preds = [[0u8; 64]; 2];
            for (k, pred) in preds.iter_mut().enumerate() {
                let p = &recon.planes[1 + k];
                let (above, left, top_left) = whole_edge(p, mbx * 8, mby * 8, 8);
                let e = Edge {
                    above: &above,
                    left: &left,
                    top_left,
                    have_above: mby > 0,
                    have_left: mbx > 0,
                };
                predict_block(pred, 0, 8, 8, mode, &e);
            }
            let d = sse(&su, &preds[0]) + sse(&sv, &preds[1]);
            if d < best_uv.0 {
                best_uv = (d, mode, preds[0], preds[1]);
            }
        }
        info.uvmode = best_uv.1;
        code_chroma(
            recon,
            mbx,
            mby,
            &su,
            &sv,
            &best_uv.2,
            &best_uv.3,
            dq,
            &mut levels,
        );
        info.skip = levels.iter().all(|b| b.iter().all(|&l| l == 0));
        MbCode {
            info,
            levels,
            best: Mv::ZERO,
            mode_probs: [0; 4],
        }
    }

    /// Codes a macroblock of an inter frame: a motion search against the
    /// last frame, then the cheapest of the inter modes and the 16x16 intra
    /// modes.
    #[allow(clippy::too_many_arguments)]
    fn code_inter_mb(
        &self,
        src: &FrameBuf,
        recon: &mut FrameBuf,
        reference: &FrameBuf,
        mbx: usize,
        mby: usize,
        dq: &Dequant,
        lambda: f64,
        mbs: &[MbInfo],
        idx: usize,
        probs: &Probs,
    ) -> MbCode {
        let (mbw, mbh) = (self.mbw, self.mbh);
        let (x0, y0) = (mbx * 16, mby * 16);
        let s = block_of(&src.planes[0], x0, y0, 16);
        let refy = reference.planes[0].as_ref();
        let near = find_near_mvs(mbs, idx, mbw + 1, LAST, &[false; 4]);
        let mode_probs: [u8; 4] = std::array::from_fn(|i| MODE_CONTEXTS[near.cnt[i] as usize][i]);
        let clamp = |mv: Mv| clamp_mv(mv, mbx, mby, mbw, mbh);
        let best = clamp(near.best);
        let nearest = clamp(near.nearest);
        let nearv = clamp(near.near);
        // SAD weighs a bit at about sqrt(lambda).
        let lambda_sad = lambda.sqrt();
        // The intra/inter flag, about a bit before the frame's statistics
        // are known.
        let inter_flag = 256;
        let predict = |mv: Mv, out: &mut [u8; 256]| {
            predict_inter(
                refy,
                out,
                0,
                16,
                x0 as i32,
                y0 as i32,
                16,
                16,
                2 * mv.col as i32,
                2 * mv.row as i32,
                &SIXTAP_FILTERS,
            );
        };

        // Motion search.
        let lim = clamp_mv(
            Mv {
                row: i16::MIN,
                col: i16::MIN,
            },
            mbx,
            mby,
            mbw,
            mbh,
        );
        let lim_hi = clamp_mv(
            Mv {
                row: i16::MAX,
                col: i16::MAX,
            },
            mbx,
            mby,
            mbw,
            mbh,
        );
        let in_range = |mv: Mv| {
            mv.row >= lim.row
                && mv.row <= lim_hi.row
                && mv.col >= lim.col
                && mv.col <= lim_hi.col
                && (mv.row as i32 - best.row as i32).abs() <= 1023
                && (mv.col as i32 - best.col as i32).abs() <= 1023
        };
        let mv_cost = |mv: Mv| {
            mv_bits(
                &probs.mv,
                Mv {
                    row: mv.row - best.row,
                    col: mv.col - best.col,
                },
            ) + tree_cost(&MV_REF_TREE, &mode_probs, 0, NEWMV)
        };
        let full_sad = |mv: Mv| -> f64 {
            let mut pred = [0u8; 256];
            predict(mv, &mut pred);
            sad(&s, &pred) as f64 + lambda_sad * mv_cost(mv) as f64 / 256.0
        };
        let range = self.cfg.search_range as i32 * 4;
        let mut best_mv = Mv::ZERO;
        let mut best_j = f64::MAX;
        let try_mv = |mv: Mv, best_mv: &mut Mv, best_j: &mut f64| {
            if in_range(mv) {
                let j = full_sad(mv);
                if j < *best_j {
                    *best_j = j;
                    *best_mv = mv;
                }
            }
        };
        let round_px = |mv: Mv| Mv {
            row: (mv.row as i32 & !3) as i16,
            col: (mv.col as i32 & !3) as i16,
        };
        for c in [Mv::ZERO, round_px(best), round_px(nearest), round_px(nearv)] {
            try_mv(c, &mut best_mv, &mut best_j);
        }
        // Coarse grid around the zero vector, then a descending diamond.
        for dy in (-range..=range).step_by(16) {
            for dx in (-range..=range).step_by(16) {
                try_mv(
                    Mv {
                        row: dy as i16,
                        col: dx as i16,
                    },
                    &mut best_mv,
                    &mut best_j,
                );
            }
        }
        // Steps in quarter samples: 2 and 1 sample, then half and quarter.
        for step in [8, 4, 2, 1] {
            for _ in 0..16 {
                let centre = best_mv;
                for (dy, dx) in [
                    (-step, 0),
                    (step, 0),
                    (0, -step),
                    (0, step),
                    (-step, -step),
                    (-step, step),
                    (step, -step),
                    (step, step),
                ] {
                    let c = Mv {
                        row: (centre.row as i32 + dy) as i16,
                        col: (centre.col as i32 + dx) as i16,
                    };
                    try_mv(c, &mut best_mv, &mut best_j);
                }
                if best_mv == centre {
                    break;
                }
            }
        }

        // Candidate modes, by prediction error plus rate.
        let mut cands: Vec<(u8, Mv, u32)> = vec![
            (
                ZEROMV,
                Mv::ZERO,
                tree_cost(&MV_REF_TREE, &mode_probs, 0, ZEROMV),
            ),
            (
                NEARESTMV,
                nearest,
                tree_cost(&MV_REF_TREE, &mode_probs, 0, NEARESTMV),
            ),
            (
                NEARMV,
                nearv,
                tree_cost(&MV_REF_TREE, &mode_probs, 0, NEARMV),
            ),
        ];
        if !cands.iter().any(|c| c.1 == best_mv) {
            cands.push((NEWMV, best_mv, mv_cost(best_mv)));
        }
        let mut choice: Option<(f64, u8, Mv, [u8; 256])> = None;
        for (mode, mv, bits) in cands {
            let mut pred = [0u8; 256];
            predict(mv, &mut pred);
            let j = sad(&s, &pred) as f64 + lambda_sad * (bits + inter_flag) as f64 / 256.0;
            if choice.as_ref().is_none_or(|c| j < c.0) {
                choice = Some((j, mode, mv, pred));
            }
        }
        let (j_inter, mode, mv, pred) = choice.expect("three candidates at least");

        // A 16x16 intra mode instead, if its prediction is better.
        let (above, left, top_left) = whole_edge(&recon.planes[0], x0, y0, 16);
        let e = Edge {
            above: &above,
            left: &left,
            top_left,
            have_above: y0 > 0,
            have_left: x0 > 0,
        };
        let mut intra_j = f64::MAX;
        for m in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            let mut p = [0u8; 256];
            predict_block(&mut p, 0, 16, 16, m, &e);
            let j = sad(&s, &p) as f64
                + lambda_sad * (tree_cost(&YMODE_TREE, &probs.ymode, 0, m) + 2 * inter_flag) as f64
                    / 256.0;
            intra_j = intra_j.min(j);
        }
        if intra_j < j_inter {
            let mut c =
                self.code_intra_mb(src, recon, mbx, mby, dq, lambda, mbs, idx, false, probs);
            c.mode_probs = mode_probs;
            return c;
        }

        let mut info = MbInfo {
            ymode: mode,
            ref_frame: LAST,
            mv,
            mvs: [mv; 16],
            ..MbInfo::default()
        };
        info.bmodes = [B_DC_PRED; 16];
        let mut levels = [[0i16; 16]; 25];
        let (rec, _) = code_luma_y2(&s, &pred, dq, &mut levels);
        put_block(&mut recon.planes[0], x0, y0, 16, &rec);
        // Chroma, with the vectors the decoder derives.
        let cmv = chroma_mvs(&info, false);
        let su = block_of(&src.planes[1], mbx * 8, mby * 8, 8);
        let sv = block_of(&src.planes[2], mbx * 8, mby * 8, 8);
        let mut preds = [[0u8; 64]; 2];
        for (k, pred) in preds.iter_mut().enumerate() {
            let r = reference.planes[1 + k].as_ref();
            let (mx, my) = cmv[0];
            predict_inter(
                r,
                pred,
                0,
                8,
                (mbx * 8) as i32,
                (mby * 8) as i32,
                8,
                8,
                mx,
                my,
                &SIXTAP_FILTERS,
            );
        }
        code_chroma(
            recon,
            mbx,
            mby,
            &su,
            &sv,
            &preds[0],
            &preds[1],
            dq,
            &mut levels,
        );
        info.skip = levels.iter().all(|b| b.iter().all(|&l| l == 0));
        MbCode {
            info,
            levels,
            best,
            mode_probs,
        }
    }

    /// Writes the frame: uncompressed chunk, first partition (header and
    /// modes), one token partition.
    fn write_frame(
        &self,
        key: bool,
        codes: &[MbCode],
        mbs: &[MbInfo],
        mut probs: Probs,
    ) -> Result<Vec<u8>> {
        let (mbw, mbh) = (self.mbw, self.mbh);
        let q = self.cfg.quantizer as u32;
        let level = self
            .cfg
            .loop_filter_level
            .unwrap_or_else(|| auto_filter_level(q as i32)) as u32;

        // Token statistics, for the probability updates.
        let mut counts: Counts = [[[[[0; 2]; 11]; 3]; 8]; 4];
        walk_tokens(codes, mbw, mbh, |t, slot, bit| {
            if let Slot::Tree(b, c, n) = slot {
                counts[t][b][c][n][bit as usize] += 1;
            }
        });
        let mut updates = Vec::new();
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let [n0, n1] = counts[t][b][c][n];
                        let old = probs.coeff[t][b][c][n];
                        let up = COEFF_UPDATE_PROBS[t][b][c][n];
                        let total = n0 + n1;
                        let mut new = old;
                        if total > 0 {
                            let p = ((256 * n0 as u64 + total as u64 / 2) / total as u64)
                                .clamp(1, 255) as u8;
                            let old_cost = n0 as u64 * cost(old, false) as u64
                                + n1 as u64 * cost(old, true) as u64
                                + cost(up, false) as u64;
                            let new_cost = n0 as u64 * cost(p, false) as u64
                                + n1 as u64 * cost(p, true) as u64
                                + cost(up, true) as u64
                                + 8 * 256;
                            if new_cost < old_cost {
                                new = p;
                            }
                        }
                        updates.push(new != old);
                        probs.coeff[t][b][c][n] = new;
                    }
                }
            }
        }

        let skips = codes.iter().filter(|c| c.info.skip).count();
        let total = codes.len();
        let no_skip_coeff = skips > 0;
        let prob_skip = ((256 * (total - skips) + total / 2) / total).clamp(1, 255) as u8;
        let intra = codes.iter().filter(|c| c.info.ref_frame == INTRA).count();
        let prob_intra = ((256 * intra + total / 2) / total).clamp(1, 255) as u8;

        // First partition: the frame header (section 19.2).
        let mut h = BoolEncoder::new();
        if key {
            h.flag(false); // colour space
            h.flag(false); // clamping required
        }
        h.flag(false); // no segmentation
        h.flag(false); // normal loop filter
        h.literal(6, level);
        h.literal(3, self.cfg.sharpness as u32);
        h.flag(false); // no mode / reference filter deltas
        h.literal(2, 0); // one token partition
        h.literal(7, q);
        for _ in 0..5 {
            h.flag(false); // no quantiser deltas
        }
        if key {
            h.flag(true); // refresh_entropy_probs
        } else {
            h.flag(false); // refresh golden
            h.flag(false); // refresh altref
            h.literal(2, 0); // no copy to golden
            h.literal(2, 0); // no copy to altref
            h.flag(false); // sign bias golden
            h.flag(false); // sign bias altref
            h.flag(true); // refresh_entropy_probs
            h.flag(true); // refresh last
        }
        let mut u = updates.iter();
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let up = *u.next().expect("one flag per probability");
                        h.write(COEFF_UPDATE_PROBS[t][b][c][n], up);
                        if up {
                            h.literal(8, probs.coeff[t][b][c][n] as u32);
                        }
                    }
                }
            }
        }
        h.flag(no_skip_coeff);
        if no_skip_coeff {
            h.literal(8, prob_skip as u32);
        }
        if !key {
            h.literal(8, prob_intra as u32);
            h.literal(8, 255); // prob_last: always the last frame
            h.literal(8, 128); // prob_gf (unused)
            h.flag(false); // no luma mode probability update
            h.flag(false); // no chroma mode probability update
            for i in 0..2 {
                for j in 0..MVP_COUNT {
                    h.write(MV_UPDATE_PROBS[i][j], false);
                }
            }
        }

        // Modes (section 19.3).
        let stride = mbw + 1;
        for (i, code) in codes.iter().enumerate() {
            let (mbx, mby) = (i % mbw, i / mbw);
            let idx = (mby + 1) * stride + mbx + 1;
            let info = &code.info;
            if no_skip_coeff {
                h.write(prob_skip, info.skip);
            }
            if key {
                h.tree(&KF_YMODE_TREE, &KF_YMODE_PROBS, 0, info.ymode);
                if info.ymode == B_PRED {
                    let above = mbs[idx - stride].bmodes;
                    let left = mbs[idx - 1].bmodes;
                    for b in 0..16 {
                        let a = if b < 4 {
                            above[b + 12]
                        } else {
                            info.bmodes[b - 4]
                        };
                        let l = if b & 3 == 0 {
                            left[b + 3]
                        } else {
                            info.bmodes[b - 1]
                        };
                        h.tree(
                            &BMODE_TREE,
                            &KF_BMODE_PROBS[a as usize][l as usize],
                            0,
                            info.bmodes[b],
                        );
                    }
                }
                h.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, 0, info.uvmode);
                continue;
            }
            h.write(prob_intra, info.ref_frame != INTRA);
            if info.ref_frame == INTRA {
                h.tree(&YMODE_TREE, &probs.ymode, 0, info.ymode);
                if info.ymode == B_PRED {
                    for b in 0..16 {
                        h.tree(&BMODE_TREE, &BMODE_PROBS, 0, info.bmodes[b]);
                    }
                }
                h.tree(&UV_MODE_TREE, &probs.uvmode, 0, info.uvmode);
                continue;
            }
            debug_assert_eq!(info.ref_frame, LAST);
            h.write(255, false); // last frame
            h.tree(&MV_REF_TREE, &code.mode_probs, 0, info.ymode);
            if info.ymode == NEWMV {
                write_mv(
                    &mut h,
                    &probs.mv,
                    Mv {
                        row: info.mv.row - code.best.row,
                        col: info.mv.col - code.best.col,
                    },
                );
            }
        }
        let first = h.finish();

        // The token partition.
        let mut tk = BoolEncoder::new();
        walk_tokens(codes, mbw, mbh, |t, slot, bit| match slot {
            Slot::Tree(b, c, n) => tk.write(probs.coeff[t][b][c][n], bit),
            Slot::Fixed(p) => tk.write(p, bit),
        });
        let tokens = tk.finish();

        if first.len() >= 1 << 19 {
            return Err(config("first partition above 512 KiB: raise the quantiser"));
        }
        let mut out = Vec::with_capacity(10 + first.len() + tokens.len());
        let tag = (!key as u32) | (1 << 4) | ((first.len() as u32) << 5);
        out.extend_from_slice(&tag.to_le_bytes()[..3]);
        if key {
            out.extend_from_slice(&[0x9d, 0x01, 0x2a]);
            out.extend_from_slice(&(self.cfg.width as u16).to_le_bytes());
            out.extend_from_slice(&(self.cfg.height as u16).to_le_bytes());
        }
        out.extend_from_slice(&first);
        out.extend_from_slice(&tokens);
        Ok(out)
    }
}

/// A loop filter level for a quantiser index: stronger as quantisation
/// coarsens.
fn auto_filter_level(q: i32) -> u8 {
    (AC_QLOOKUP[q.clamp(0, 127) as usize] as i32 / 4).clamp(0, 63) as u8
}

/// The input, padded to whole macroblocks by repeating its last row and
/// column.
fn pad_source(frame: &Frame, mbw: usize, mbh: usize) -> FrameBuf {
    let mut planes: [PlaneBuf; 3] = Default::default();
    for (i, p) in planes.iter_mut().enumerate() {
        let n = if i == 0 { 16 } else { 8 };
        let (w, h) = (mbw * n, mbh * n);
        let fp = frame.planes[i];
        let (fw, fh) = (fp.width as usize, fp.height as usize);
        let data = frame.plane(i);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            let sy = y.min(fh - 1);
            for x in 0..w {
                out[y * w + x] = data[sy * fw + x.min(fw - 1)];
            }
        }
        *p = PlaneBuf {
            data: out,
            width: w,
            height: h,
        };
    }
    FrameBuf { planes }
}

/// The `n`x`n` block at (`x0`, `y0`) of a plane, row by row.
fn block_of<const N: usize>(p: &PlaneBuf, x0: usize, y0: usize, n: usize) -> [u8; N] {
    let mut out = [0u8; N];
    for r in 0..n {
        out[r * n..r * n + n].copy_from_slice(&p.data[(y0 + r) * p.width + x0..][..n]);
    }
    out
}

fn put_block(p: &mut PlaneBuf, x0: usize, y0: usize, n: usize, b: &[u8]) {
    for r in 0..n {
        let w = p.width;
        p.data[(y0 + r) * w + x0..][..n].copy_from_slice(&b[r * n..r * n + n]);
    }
}

fn sse(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((x as i32 - y as i32) * (x as i32 - y as i32)) as u64)
        .sum()
}

fn sad(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

/// Quantises `c` (a forward DCT or WHT, raster order) from coefficient
/// `first` on; returns the levels and the dequantised values the decoder
/// will reconstruct from them.
fn quantise(c: &[i16; 16], q: [i32; 2], first: usize) -> ([i16; 16], [i16; 16]) {
    let mut lv = [0i16; 16];
    let mut deq = [0i16; 16];
    for &r in &ZIGZAG[first..] {
        let step = q[(r > 0) as usize];
        let v = c[r] as i32;
        // A dead zone below half a step for the AC coefficients.
        let round = if r == 0 { step / 2 } else { step * 3 / 8 };
        let l = ((v.abs() + round) / step).min(2048);
        let l = if v < 0 { -l } else { l };
        lv[r] = l as i16;
        deq[r] = (l * step) as i16;
    }
    (lv, deq)
}

/// Transforms and quantises the residue of one 4x4 block.
fn code_4x4(src: &[u8; 16], pred: &[u8; 16], q: [i32; 2], first: usize) -> ([i16; 16], [i16; 16]) {
    let res: [i16; 16] = std::array::from_fn(|i| src[i] as i16 - pred[i] as i16);
    quantise(&forward_dct(&res), q, first)
}

/// Codes a 16x16 luma residue with a Y2 block: returns the reconstruction
/// and an estimate of the rate (1/256 bits); `levels` receives Y 0-15 and
/// Y2 (24).
fn code_luma_y2(
    src: &[u8; 256],
    pred: &[u8; 256],
    dq: &Dequant,
    levels: &mut [[i16; 16]; 25],
) -> ([u8; 256], u32) {
    let mut coefs = [[0i16; 16]; 16];
    let mut dcs = [0i16; 16];
    for b in 0..16 {
        let (bx, by) = (b & 3, b >> 2);
        let res: [i16; 16] = std::array::from_fn(|i| {
            let o = (4 * by + i / 4) * 16 + 4 * bx + i % 4;
            src[o] as i16 - pred[o] as i16
        });
        coefs[b] = forward_dct(&res);
        dcs[b] = coefs[b][0];
    }
    let (y2, y2deq) = quantise(&forward_wht(&dcs), dq.y2, 0);
    levels[24] = y2;
    let dc = inverse_wht(&y2deq);
    let mut rate = block_rate(&y2, 0);
    let mut rec = *pred;
    for b in 0..16 {
        let (lv, mut deq) = quantise(&coefs[b], dq.y, 1);
        levels[b] = lv;
        rate += block_rate(&lv, 1);
        deq[0] = dc[b];
        let off = 4 * (b >> 2) * 16 + 4 * (b & 3);
        add_residue(&deq, &mut rec, off, 16);
    }
    (rec, rate)
}

/// Codes both chroma residues from their predictions into `recon` and
/// `levels` 16-23.
#[allow(clippy::too_many_arguments)]
fn code_chroma(
    recon: &mut FrameBuf,
    mbx: usize,
    mby: usize,
    su: &[u8; 64],
    sv: &[u8; 64],
    pu: &[u8; 64],
    pv: &[u8; 64],
    dq: &Dequant,
    levels: &mut [[i16; 16]; 25],
) {
    for (k, (s, p)) in [(su, pu), (sv, pv)].into_iter().enumerate() {
        let mut rec = *p;
        for b in 0..4 {
            let (bx, by) = (b & 1, b >> 1);
            let sb: [u8; 16] = std::array::from_fn(|i| s[(4 * by + i / 4) * 8 + 4 * bx + i % 4]);
            let pb: [u8; 16] = std::array::from_fn(|i| p[(4 * by + i / 4) * 8 + 4 * bx + i % 4]);
            let (lv, deq) = code_4x4(&sb, &pb, dq.uv, 0);
            levels[16 + 4 * k + b] = lv;
            add_residue(&deq, &mut rec, 4 * by * 8 + 4 * bx, 8);
        }
        put_block(&mut recon.planes[1 + k], mbx * 8, mby * 8, 8, &rec);
    }
}

/// A rough rate for one block's tokens, in 1/256 bits: about two bits per
/// zero before the last non-zero level, and more for larger levels.
fn block_rate(lv: &[i16; 16], first: usize) -> u32 {
    let Some(last) = (first..16).rev().find(|&i| lv[ZIGZAG[i]] != 0) else {
        return 256;
    };
    let mut r = 256; // end of block
    for &z in &ZIGZAG[first..=last] {
        let a = lv[z].unsigned_abs() as u32;
        r += if a == 0 {
            384
        } else {
            768 + 512 * (32 - a.leading_zeros())
        };
    }
    r
}

/// Calls `emit(plane type, slot, bit)` for every bool of every token of
/// the frame, in partition order, with the contexts of section 13.3.
fn walk_tokens(codes: &[MbCode], mbw: usize, mbh: usize, mut emit: impl FnMut(usize, Slot, bool)) {
    let mut above = vec![[0u8; 9]; mbw];
    for mby in 0..mbh {
        let mut left = [0u8; 9];
        for (mbx, a) in above.iter_mut().enumerate() {
            let code = &codes[mby * mbw + mbx];
            let has_y2 = code.info.ymode != B_PRED && code.info.ymode != SPLITMV;
            if code.info.skip {
                let (kl, ka) = (left[8], a[8]);
                left = [0; 9];
                *a = [0; 9];
                if !has_y2 {
                    left[8] = kl;
                    a[8] = ka;
                }
                continue;
            }
            let lv = &code.levels;
            let (ytype, yfirst) = if has_y2 { (0, 1) } else { (3, 0) };
            if has_y2 {
                let nz = block_tokens(&lv[24], 1, 0, (a[8] + left[8]) as usize, &mut emit);
                a[8] = nz as u8;
                left[8] = nz as u8;
            }
            for b in 0..16 {
                let (x, y) = (b & 3, b >> 2);
                let nz = block_tokens(&lv[b], ytype, yfirst, (a[x] + left[y]) as usize, &mut emit);
                a[x] = nz as u8;
                left[y] = nz as u8;
            }
            for (base, off) in [(16, 4), (20, 6)] {
                for b in 0..4 {
                    let (x, y) = (b & 1, b >> 1);
                    let nz = block_tokens(
                        &lv[base + b],
                        2,
                        0,
                        (a[off + x] + left[off + y]) as usize,
                        &mut emit,
                    );
                    a[off + x] = nz as u8;
                    left[off + y] = nz as u8;
                }
            }
        }
    }
}

/// The tokens of one block (section 13.2); returns whether it has a
/// non-zero level.
fn block_tokens(
    lv: &[i16; 16],
    t: usize,
    first: usize,
    ctx: usize,
    emit: &mut impl FnMut(usize, Slot, bool),
) -> bool {
    let mut path = [(0usize, false); 16];
    let mut put = |token: u8,
                   start: usize,
                   band: usize,
                   ctx: usize,
                   emit: &mut dyn FnMut(usize, Slot, bool)| {
        let n = tree_path(&COEFF_TREE, start, token, &mut path).expect("a token");
        for &(node, bit) in &path[..n] {
            emit(t, Slot::Tree(band, ctx, node >> 1), bit);
        }
    };
    let Some(last) = (first..16).rev().find(|&i| lv[ZIGZAG[i]] != 0) else {
        put(DCT_EOB, 0, COEFF_BANDS[first], ctx, emit);
        return false;
    };
    let mut ctx = ctx;
    let mut start = 0;
    for i in first..=last {
        let v = lv[ZIGZAG[i]] as i32;
        let a = v.abs();
        let band = COEFF_BANDS[i];
        if a == 0 {
            put(0, start, band, ctx, emit);
            ctx = 0;
            start = 2;
            continue;
        }
        if a <= 4 {
            put(a as u8, start, band, ctx, emit);
        } else {
            let cat = CAT_BASE.iter().rposition(|&b| a >= b).expect("a >= 5");
            put(DCT_CAT1 + cat as u8, start, band, ctx, emit);
            let extra = a - CAT_BASE[cat];
            let bits = PCAT[cat];
            for (k, &p) in bits.iter().enumerate() {
                emit(t, Slot::Fixed(p), (extra >> (bits.len() - 1 - k)) & 1 == 1);
            }
        }
        emit(t, Slot::Fixed(128), v < 0);
        ctx = if a == 1 { 1 } else { 2 };
        start = 0;
    }
    if last < 15 {
        put(DCT_EOB, 0, COEFF_BANDS[last + 1], ctx, emit);
    }
    true
}

/// Writes a motion vector difference (section 17), the inverse of the
/// decoder's reading.
fn write_mv(h: &mut BoolEncoder, p: &[[u8; MVP_COUNT]; 2], d: Mv) {
    write_mv_component(h, &p[0], d.row);
    write_mv_component(h, &p[1], d.col);
}

fn write_mv_component(h: &mut BoolEncoder, p: &[u8; MVP_COUNT], v: i16) {
    let a = v.unsigned_abs() as u32;
    debug_assert!(a <= 1023);
    if a < 8 {
        h.write(p[MVP_IS_SHORT], false);
        h.tree(&SMALL_MV_TREE, &p[MVP_SHORT..MVP_SHORT + 7], 0, a as u8);
    } else {
        h.write(p[MVP_IS_SHORT], true);
        for i in 0..3 {
            h.write(p[MVP_BITS + i], (a >> i) & 1 == 1);
        }
        for i in (4..10).rev() {
            h.write(p[MVP_BITS + i], (a >> i) & 1 == 1);
        }
        if a & 0xfff0 != 0 {
            h.write(p[MVP_BITS + 3], (a >> 3) & 1 == 1);
        }
    }
    if a != 0 {
        h.write(p[MVP_SIGN], v < 0);
    }
}

/// The cost of a motion vector difference, in 1/256 bits.
fn mv_bits(p: &[[u8; MVP_COUNT]; 2], d: Mv) -> u32 {
    mv_component_bits(&p[0], d.row) + mv_component_bits(&p[1], d.col)
}

fn mv_component_bits(p: &[u8; MVP_COUNT], v: i16) -> u32 {
    let a = v.unsigned_abs() as u32;
    let mut c = if a < 8 {
        cost(p[MVP_IS_SHORT], false)
            + tree_cost(&SMALL_MV_TREE, &p[MVP_SHORT..MVP_SHORT + 7], 0, a as u8)
    } else {
        let mut c = cost(p[MVP_IS_SHORT], true);
        for i in (0..10).filter(|&i| i != 3) {
            c += cost(p[MVP_BITS + i], (a >> i) & 1 == 1);
        }
        if a & 0xfff0 != 0 {
            c += cost(p[MVP_BITS + 3], (a >> 3) & 1 == 1);
        }
        c
    };
    if a != 0 {
        c += cost(p[MVP_SIGN], v < 0);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolcoder::BoolDecoder;

    #[test]
    fn mv_components_round_trip() {
        let p = DEFAULT_MV_PROBS;
        let values: Vec<i16> = (-1023..=1023)
            .step_by(7)
            .chain([0, 1, -1, 7, 8, 15, 16, -8, 1023, -1023])
            .collect();
        let mut h = BoolEncoder::new();
        for &v in &values {
            write_mv_component(&mut h, &p[0], v);
        }
        let data = h.finish();
        let mut d = BoolDecoder::new(&data);
        for &v in &values {
            // The decoder's reader, through a one-component vector.
            let got = crate::decoder::read_mv_component(&mut d, &p[0]);
            assert_eq!(got, v);
        }
    }

    #[test]
    fn quantise_matches_dequantise() {
        let c: [i16; 16] = std::array::from_fn(|i| (i as i16 - 8) * 37);
        let (lv, deq) = quantise(&c, [10, 20], 0);
        for i in 0..16 {
            let step = if i == 0 { 10 } else { 20 };
            assert_eq!(deq[i], lv[i] * step);
        }
    }
}
