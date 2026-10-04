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

use crate::boolcoder::{BoolEncoder, TreePaths, cost};
use crate::decoder::{
    Decoder, Dequant, FrameBuf, INTRA, LAST, MbInfo, Mv, PlaneBuf, Probs, QuantIndices, chroma_mvs,
    clamp_mv, find_near_mvs, implied_bmode,
};
use crate::dsp::{QuantParams, dsp};
use crate::error::{Result, config};
use crate::frame::Frame;
use crate::pool::{Padded, PanicGuard};
use crate::predict::{Edge, predict_block, predict_inter, predict_subblock};
use crate::recon::{LS, MbWork, SharedFrame};
use crate::tables::*;
use crate::transform::{add_residue, forward_dct, forward_wht, inverse_wht};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

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
    /// Token partitions: 1, 2, 4 or 8 (macroblock rows take them in turn,
    /// so a decoder can work on several rows at once).
    pub token_partitions: u8,
    /// Threads to encode each frame on, the caller's included: 0 means one
    /// per CPU, 1 the caller's alone. The output is the same whatever the
    /// count.
    pub threads: usize,
}

impl Default for Config {
    /// 0x0 (set the size), quantiser 40, a key frame every 120 frames,
    /// automatic loop filter, search range 16, one token partition, one
    /// thread.
    fn default() -> Self {
        Config {
            width: 0,
            height: 0,
            quantizer: 40,
            keyframe_interval: 120,
            loop_filter_level: None,
            sharpness: 0,
            search_range: 16,
            token_partitions: 1,
            threads: 1,
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
    /// The token contexts (section 13.3) the macroblock's tokens start
    /// from: those above, those to the left.
    ctx_in: [[u8; 9]; 2],
    /// The contexts it leaves for the macroblock below.
    ctx_below: [u8; 9],
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
        if ![1, 2, 4, 8].contains(&cfg.token_partitions) {
            return Err(config("token partitions must be 1, 2, 4 or 8"));
        }
        if cfg.search_range == 0 || cfg.search_range > 64 {
            return Err(config("search range must be 1 to 64"));
        }
        let mbw = (cfg.width as usize).div_ceil(16);
        let mbh = (cfg.height as usize).div_ceil(16);
        let dec = Decoder::with_threads(cfg.threads);
        Ok(Encoder {
            cfg,
            mbw,
            mbh,
            frames: 0,
            force_key: false,
            dec,
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

    fn encode_frame(&self, src: &FrameBuf, key: bool) -> Result<Vec<u8>> {
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
        let ctx = MbCtx {
            src,
            reference: (!key).then(|| self.dec.last_reference()),
            quant: Quantisers::new(&dq),
            lambda,
            costs: ModeCosts::new(&probs),
            mbw,
            mbh,
            search_range: self.cfg.search_range,
        };

        // Macroblocks are decided in raster order as far as each one's
        // inputs go: its neighbours' modes and vectors (left, above-left,
        // above) and their reconstruction (left, above, above-right, for
        // intra prediction). So rows run in a wavefront, each a macroblock
        // or two behind the row above, on the decoder's threads; the
        // choices do not depend on the thread count.
        let mut recon = FrameBuf::new(mbw, mbh);
        let slots: Vec<OnceLock<MbCode>> = (0..mbw * mbh).map(|_| OnceLock::new()).collect();
        let progress: Vec<Padded> = (0..mbh).map(|_| Padded(AtomicUsize::new(0))).collect();
        let next_row = AtomicUsize::new(0);
        let poisoned = AtomicBool::new(false);
        // Token statistics for the probability updates, gathered as the
        // macroblocks are decided (sums, so the order does not matter).
        let counts: Mutex<Box<Counts>> = Mutex::new(Box::new([[[[[0; 2]; 11]; 3]; 8]; 4]));
        {
            let shared = SharedFrame::new(&mut recon);
            let work = || {
                let _guard = PanicGuard(&poisoned);
                let mut w = MbWork::new();
                let mut mine: Box<Counts> = Box::new([[[[[0; 2]; 11]; 3]; 8]; 4]);
                loop {
                    let y = next_row.fetch_add(1, Ordering::Relaxed);
                    if y >= mbh {
                        break;
                    }
                    let mut left = [0u8; 9];
                    for x in 0..mbw {
                        if y > 0 {
                            wait(&progress[y - 1], (x + 2).min(mbw), &poisoned);
                        }
                        let nb = neighbours(&slots, mbw, x, y);
                        // SAFETY: the macroblocks whose pixels these edges
                        // come from (left; above-left to above-right) are
                        // finished — the left by this thread, the row above
                        // waited for — and nobody writes them again.
                        unsafe { w.load_edges(&shared, x, y, mbw) };
                        let mut code = match ctx.reference {
                            None => ctx.code_intra_mb(&mut w, x, y, &nb, true),
                            Some(r) => ctx.code_inter_mb(&mut w, r, x, y, &nb),
                        };
                        let above = if y > 0 {
                            slots[(y - 1) * mbw + x]
                                .get()
                                .expect("the macroblock above is coded")
                                .ctx_below
                        } else {
                            [0; 9]
                        };
                        code.ctx_in = [above, left];
                        let mut below = above;
                        mb_tokens(&code, &mut below, &mut left, &mut |t, slot, bit| {
                            if let Slot::Tree(b, c, n) = slot {
                                mine[t][b][c][n][bit as usize] += 1;
                            }
                        });
                        code.ctx_below = below;
                        // SAFETY: this macroblock's pixels are this thread's
                        // alone until its progress is published below.
                        unsafe { w.store(&shared, x, y) };
                        let _ = slots[y * mbw + x].set(code);
                        progress[y].store(x + 1, Ordering::Release);
                    }
                }
                let mut all = counts.lock().unwrap_or_else(|e| e.into_inner());
                for (a, m) in all
                    .iter_mut()
                    .flatten()
                    .flatten()
                    .flatten()
                    .zip(mine.iter().flatten().flatten().flatten())
                {
                    a[0] += m[0];
                    a[1] += m[1];
                }
            };
            self.dec.pool().run(&work);
        }
        let counts = counts.into_inner().unwrap_or_else(|e| e.into_inner());
        let codes: Vec<MbCode> = slots
            .into_iter()
            .map(|s| s.into_inner().expect("every macroblock coded"))
            .collect();
        let stride = mbw + 1;
        let mut mbs = vec![MbInfo::default(); stride * (mbh + 1)];
        for (i, c) in codes.iter().enumerate() {
            mbs[(i / mbw + 1) * stride + i % mbw + 1] = c.info;
        }
        self.write_frame(key, &codes, &mbs, probs, &counts)
    }
}

/// The modes and vectors around macroblock (`x`, `y`), in the layout
/// `find_near_mvs` takes (stride 3, the macroblock at 4): above-left (0),
/// above (1), left (3). Outside the frame they are the default (intra,
/// zero vectors, B_DC_PRED), as the decoder's border is.
fn neighbours(slots: &[OnceLock<MbCode>], mbw: usize, x: usize, y: usize) -> [MbInfo; 6] {
    let get = |dx: isize, dy: isize| {
        let (xx, yy) = (x as isize + dx, y as isize + dy);
        if xx < 0 || yy < 0 {
            return MbInfo::default();
        }
        slots[yy as usize * mbw + xx as usize]
            .get()
            .expect("a neighbour coded before")
            .info
    };
    let d = MbInfo::default();
    [get(-1, -1), get(0, -1), d, get(-1, 0), d, d]
}

/// Waits until `counter` reaches `target`.
fn wait(counter: &AtomicUsize, target: usize, poisoned: &AtomicBool) {
    let mut spins = 0u32;
    while counter.load(Ordering::Acquire) < target {
        if poisoned.load(Ordering::Relaxed) {
            panic!("another thread encoding this frame panicked");
        }
        if spins < 200 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
}

/// The dead-zone quantiser of a block type, from its dequantisation
/// factors (DC, AC): rounding by half a step for DC, three eighths (a dead
/// zone below half a step) for AC.
fn quant_params(q: [i32; 2]) -> QuantParams {
    QuantParams::new(q, [q[0] / 2, q[1] * 3 / 8])
}

/// The quantisers of a frame.
struct Quantisers {
    y: QuantParams,
    y2: QuantParams,
    uv: QuantParams,
}

impl Quantisers {
    fn new(dq: &Dequant) -> Quantisers {
        Quantisers {
            y: quant_params(dq.y),
            y2: quant_params(dq.y2),
            uv: quant_params(dq.uv),
        }
    }
}

/// Paths through the trees the encoder prices or codes per macroblock.
struct Trees {
    coeff: [TreePaths; 2],
    mv_ref: TreePaths,
    ymode: TreePaths,
    kf_ymode: TreePaths,
    bmode: TreePaths,
    small_mv: TreePaths,
}

static TREES: LazyLock<Trees> = LazyLock::new(|| Trees {
    coeff: [
        TreePaths::new(&COEFF_TREE, 0),
        TreePaths::new(&COEFF_TREE, 2),
    ],
    mv_ref: TreePaths::new(&MV_REF_TREE, 0),
    ymode: TreePaths::new(&YMODE_TREE, 0),
    kf_ymode: TreePaths::new(&KF_YMODE_TREE, 0),
    bmode: TreePaths::new(&BMODE_TREE, 0),
    small_mv: TreePaths::new(&SMALL_MV_TREE, 0),
});

/// Key-frame subblock mode costs by (above, left) context and mode.
static KF_BMODE_COSTS: LazyLock<[[[u32; 10]; 10]; 10]> = LazyLock::new(|| {
    std::array::from_fn(|a| {
        std::array::from_fn(|l| {
            std::array::from_fn(|m| TREES.bmode.cost(&KF_BMODE_PROBS[a][l], m as u8))
        })
    })
});

/// The mode and vector costs of a frame (its probabilities are fixed while
/// its macroblocks are decided).
struct ModeCosts {
    /// Inter-frame luma modes.
    ymode: [u32; 5],
    /// Key-frame luma modes.
    kf_ymode: [u32; 5],
    /// Each motion vector component's cost, row then column, for values
    /// -1023..=1023.
    mv: [Vec<u32>; 2],
    mv_probs: [[u8; MVP_COUNT]; 2],
}

impl ModeCosts {
    fn new(probs: &Probs) -> ModeCosts {
        let t = &*TREES;
        ModeCosts {
            ymode: std::array::from_fn(|m| t.ymode.cost(&probs.ymode, m as u8)),
            kf_ymode: std::array::from_fn(|m| t.kf_ymode.cost(&KF_YMODE_PROBS, m as u8)),
            mv: std::array::from_fn(|c| {
                (-1023..=1023)
                    .map(|v| mv_component_bits(&probs.mv[c], v as i16))
                    .collect()
            }),
            mv_probs: probs.mv,
        }
    }

    /// The cost of a motion vector difference, in 1/256 bits (as
    /// [`mv_bits`]).
    #[inline]
    fn mv_bits(&self, d: Mv) -> u32 {
        let (r, c) = (d.row as i32 + 1023, d.col as i32 + 1023);
        if (0..2047).contains(&r) && (0..2047).contains(&c) {
            self.mv[0][r as usize] + self.mv[1][c as usize]
        } else {
            mv_bits(&self.mv_probs, d)
        }
    }
}

/// What deciding a macroblock needs of the frame (shared by the threads).
struct MbCtx<'a> {
    src: &'a FrameBuf,
    /// The last frame, for inter frames.
    reference: Option<&'a FrameBuf>,
    quant: Quantisers,
    lambda: f64,
    costs: ModeCosts,
    mbw: usize,
    mbh: usize,
    search_range: u8,
}

impl MbCtx<'_> {
    /// Codes a macroblock with intra prediction: on key frames trying the
    /// 16x16 modes and B_PRED, in inter frames the 16x16 modes. `w` holds
    /// the macroblock's edges and receives its reconstruction.
    fn code_intra_mb(
        &self,
        w: &mut MbWork,
        mbx: usize,
        mby: usize,
        nb: &[MbInfo; 6],
        key: bool,
    ) -> MbCode {
        let (x0, y0) = (mbx * 16, mby * 16);
        let s = block_of(&self.src.planes[0], x0, y0, 16);
        let lambda = self.lambda;
        let mut info = MbInfo::default();
        let mut levels = [[0i16; 16]; 25];

        // The 16x16 modes, each coded in full.
        let (above, left, top_left) = w.luma_edge();
        let e = Edge {
            above: &above,
            left: &left,
            top_left,
            have_above: y0 > 0,
            have_left: x0 > 0,
        };
        let mode_cost = |m: u8| {
            if key {
                self.costs.kf_ymode[m as usize]
            } else {
                self.costs.ymode[m as usize]
            }
        };
        // (cost, mode, levels, reconstruction)
        type Candidate = (f64, u8, [[i16; 16]; 25], [u8; 256]);
        let mut best16: Option<Candidate> = None;
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            let mut pred = [0u8; 256];
            predict_block(&mut pred, 0, 16, 16, mode, &e);
            let mut lv = [[0i16; 16]; 25];
            let (rec, rate) = code_luma_y2(&s, &pred, &self.quant, &mut lv);
            let j = sse(&s, 16, &rec, 16, 16, 16) as f64
                + lambda * (rate + mode_cost(mode)) as f64 / 256.0;
            if best16.as_ref().is_none_or(|b| j < b.0) {
                best16 = Some((j, mode, lv, rec));
            }
        }
        let (j16, mode16, lv16, rec16) = best16.expect("four modes tried");

        // B_PRED, on key frames: subblock by subblock in the work buffer,
        // since each subblock predicts from the ones before it.
        let mut use_bpred = false;
        if key {
            let above_modes = nb[1].bmodes;
            let left_modes = nb[3].bmodes;
            let mut bmodes = [B_DC_PRED; 16];
            let mut lvb = [[0i16; 16]; 25];
            let mut jb = lambda * mode_cost(B_PRED) as f64 / 256.0;
            for b in 0..16 {
                let (bx, by) = (b & 3, b >> 2);
                let sb: [u8; 16] =
                    std::array::from_fn(|i| s[(4 * by + i / 4) * 16 + 4 * bx + i % 4]);
                let (ab, lb, tl) = w.subblock_edge(b);
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
                let mcosts = &KF_BMODE_COSTS[ctx_a as usize][ctx_l as usize];
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
                    let (lv, deq) = code_4x4(&sb, &pred, &self.quant.y, 0);
                    let mut rec = pred;
                    add_residue(&deq, &mut rec, 0, 4);
                    let rate = block_rate(&lv, 0) + mcosts[m as usize];
                    let j = sse(&sb, 4, &rec, 4, 4, 4) as f64 + lambda * rate as f64 / 256.0;
                    if best.as_ref().is_none_or(|x| j < x.0) {
                        best = Some((j, m, lv, deq));
                    }
                }
                let (j, m, lv, deq) = best.expect("ten modes tried");
                jb += j;
                bmodes[b] = m;
                lvb[b] = lv;
                w.predict_sub(b, m);
                add_residue(&deq, &mut w.luma, MbWork::sub_off(b), LS);
                if jb > j16 {
                    break;
                }
            }
            if jb < j16 {
                use_bpred = true;
                info.ymode = B_PRED;
                info.bmodes = bmodes;
                levels[..16].copy_from_slice(&lvb[..16]);
            }
        }
        if !use_bpred {
            info.ymode = mode16;
            info.bmodes = [implied_bmode(mode16); 16];
            levels[..16].copy_from_slice(&lv16[..16]);
            levels[24] = lv16[24];
            w.put_luma(&rec16);
        }

        // Chroma: the mode with the least prediction error.
        let su = block_of(&self.src.planes[1], mbx * 8, mby * 8, 8);
        let sv = block_of(&self.src.planes[2], mbx * 8, mby * 8, 8);
        let mut best_uv = (u64::MAX, DC_PRED, [0u8; 64], [0u8; 64]);
        let edges = [w.chroma_edge(1), w.chroma_edge(2)];
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            let mut preds = [[0u8; 64]; 2];
            for (pred, (above, left, top_left)) in preds.iter_mut().zip(&edges) {
                let e = Edge {
                    above,
                    left,
                    top_left: *top_left,
                    have_above: mby > 0,
                    have_left: mbx > 0,
                };
                predict_block(pred, 0, 8, 8, mode, &e);
            }
            let d = sse(&su, 8, &preds[0], 8, 8, 8) + sse(&sv, 8, &preds[1], 8, 8, 8);
            if d < best_uv.0 {
                best_uv = (d, mode, preds[0], preds[1]);
            }
        }
        info.uvmode = best_uv.1;
        code_chroma(
            w,
            &su,
            &sv,
            &best_uv.2,
            &best_uv.3,
            &self.quant.uv,
            &mut levels,
        );
        info.skip = levels.iter().all(|b| b.iter().all(|&l| l == 0));
        MbCode {
            info,
            levels,
            best: Mv::ZERO,
            mode_probs: [0; 4],
            ctx_in: [[0; 9]; 2],
            ctx_below: [0; 9],
        }
    }

    /// Codes a macroblock of an inter frame: a motion search against the
    /// last frame, then the cheapest of the inter modes and the 16x16 intra
    /// modes.
    fn code_inter_mb(
        &self,
        w: &mut MbWork,
        reference: &FrameBuf,
        mbx: usize,
        mby: usize,
        nb: &[MbInfo; 6],
    ) -> MbCode {
        let (mbw, mbh) = (self.mbw, self.mbh);
        let (x0, y0) = (mbx * 16, mby * 16);
        let s = block_of(&self.src.planes[0], x0, y0, 16);
        let refy = reference.planes[0].as_ref();
        let near = find_near_mvs(nb, 4, 3, LAST, &[false; 4]);
        let mode_probs: [u8; 4] = std::array::from_fn(|i| MODE_CONTEXTS[near.cnt[i] as usize][i]);
        let mode_cost: [u32; 10] = std::array::from_fn(|m| match m as u8 {
            NEARESTMV | NEARMV | ZEROMV | NEWMV | SPLITMV => {
                TREES.mv_ref.cost(&mode_probs, m as u8)
            }
            _ => 0,
        });
        let clamp = |mv: Mv| clamp_mv(mv, mbx, mby, mbw, mbh);
        let best = clamp(near.best);
        let nearest = clamp(near.nearest);
        let nearv = clamp(near.near);
        // SAD weighs a bit at about sqrt(lambda).
        let lambda_sad = self.lambda.sqrt();
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
        // The SAD of the prediction for `mv`: read straight from the
        // reference for a whole-sample vector inside it (the prediction is
        // then a copy), otherwise predicted.
        let sad_of = |mv: Mv| -> u32 {
            if mv.row & 3 == 0 && mv.col & 3 == 0 {
                let (ix, iy) = (
                    x0 as i32 + (mv.col as i32 >> 2),
                    y0 as i32 + (mv.row as i32 >> 2),
                );
                if ix >= 0
                    && iy >= 0
                    && ix as usize + 16 <= refy.width
                    && iy as usize + 16 <= refy.height
                {
                    let off = iy as usize * refy.width + ix as usize;
                    return sad(&s, 16, &refy.data[off..], refy.width, 16, 16);
                }
            }
            let mut pred = [0u8; 256];
            predict(mv, &mut pred);
            sad(&s, 16, &pred, 16, 16, 16)
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
        let newmv_cost = mode_cost[NEWMV as usize];
        let mv_cost = |mv: Mv| {
            self.costs.mv_bits(Mv {
                row: mv.row - best.row,
                col: mv.col - best.col,
            }) + newmv_cost
        };
        let full_sad =
            |mv: Mv| -> f64 { sad_of(mv) as f64 + lambda_sad * mv_cost(mv) as f64 / 256.0 };
        let range = self.search_range as i32 * 4;
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
            (ZEROMV, Mv::ZERO, mode_cost[ZEROMV as usize]),
            (NEARESTMV, nearest, mode_cost[NEARESTMV as usize]),
            (NEARMV, nearv, mode_cost[NEARMV as usize]),
        ];
        if !cands.iter().any(|c| c.1 == best_mv) {
            cands.push((NEWMV, best_mv, mv_cost(best_mv)));
        }
        let mut choice: Option<(f64, u8, Mv)> = None;
        for (mode, mv, bits) in cands {
            let j = sad_of(mv) as f64 + lambda_sad * (bits + inter_flag) as f64 / 256.0;
            if choice.as_ref().is_none_or(|c| j < c.0) {
                choice = Some((j, mode, mv));
            }
        }
        let (j_inter, mode, mv) = choice.expect("three candidates at least");

        // A 16x16 intra mode instead, if its prediction is better.
        let (above, left, top_left) = w.luma_edge();
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
            let j = sad(&s, 16, &p, 16, 16, 16) as f64
                + lambda_sad * (self.costs.ymode[m as usize] + 2 * inter_flag) as f64 / 256.0;
            intra_j = intra_j.min(j);
        }
        if intra_j < j_inter {
            let mut c = self.code_intra_mb(w, mbx, mby, nb, false);
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
        let mut pred = [0u8; 256];
        predict(mv, &mut pred);
        let (rec, _) = code_luma_y2(&s, &pred, &self.quant, &mut levels);
        w.put_luma(&rec);
        // Chroma, with the vectors the decoder derives.
        let cmv = chroma_mvs(&info, false);
        let su = block_of(&self.src.planes[1], mbx * 8, mby * 8, 8);
        let sv = block_of(&self.src.planes[2], mbx * 8, mby * 8, 8);
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
            w,
            &su,
            &sv,
            &preds[0],
            &preds[1],
            &self.quant.uv,
            &mut levels,
        );
        info.skip = levels.iter().all(|b| b.iter().all(|&l| l == 0));
        MbCode {
            info,
            levels,
            best,
            mode_probs,
            ctx_in: [[0; 9]; 2],
            ctx_below: [0; 9],
        }
    }
}

impl Encoder {
    /// Writes the frame: uncompressed chunk, first partition (header and
    /// modes), one token partition.
    fn write_frame(
        &self,
        key: bool,
        codes: &[MbCode],
        mbs: &[MbInfo],
        mut probs: Probs,
        counts: &Counts,
    ) -> Result<Vec<u8>> {
        let (mbw, mbh) = (self.mbw, self.mbh);
        let q = self.cfg.quantizer as u32;
        let level = self
            .cfg
            .loop_filter_level
            .unwrap_or_else(|| auto_filter_level(q as i32)) as u32;

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
        let np = self.cfg.token_partitions as usize;
        h.literal(2, np.trailing_zeros()); // token partitions
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

        // The token partitions: macroblock row r goes to partition r % np.
        // Each macroblock's starting contexts were kept, so the partitions
        // are written independently, on the decoder's threads.
        let parts: Vec<OnceLock<Vec<u8>>> = (0..np).map(|_| OnceLock::new()).collect();
        let next = AtomicUsize::new(0);
        self.dec.pool().run(&|| {
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= np {
                    return;
                }
                let mut enc = BoolEncoder::new();
                for mby in (k..mbh).step_by(np) {
                    for code in &codes[mby * mbw..(mby + 1) * mbw] {
                        let [mut above, mut left] = code.ctx_in;
                        mb_tokens(code, &mut above, &mut left, &mut |t, slot, bit| {
                            let p = match slot {
                                Slot::Tree(b, c, n) => probs.coeff[t][b][c][n],
                                Slot::Fixed(p) => p,
                            };
                            enc.write(p, bit);
                        });
                    }
                }
                let _ = parts[k].set(enc.finish());
            }
        });
        let parts: Vec<Vec<u8>> = parts
            .into_iter()
            .map(|p| p.into_inner().expect("every partition written"))
            .collect();
        let tokens_len: usize = parts.iter().map(Vec::len).sum();

        if first.len() >= 1 << 19 {
            return Err(config("first partition above 512 KiB: raise the quantiser"));
        }
        let mut out = Vec::with_capacity(10 + first.len() + 3 * np + tokens_len);
        let tag = (!key as u32) | (1 << 4) | ((first.len() as u32) << 5);
        out.extend_from_slice(&tag.to_le_bytes()[..3]);
        if key {
            out.extend_from_slice(&[0x9d, 0x01, 0x2a]);
            out.extend_from_slice(&(self.cfg.width as u16).to_le_bytes());
            out.extend_from_slice(&(self.cfg.height as u16).to_le_bytes());
        }
        out.extend_from_slice(&first);
        // Every partition's size but the last, 24 bits each (section 9.5).
        for p in &parts[..np - 1] {
            if p.len() >= 1 << 24 {
                return Err(config("token partition above 16 MiB: raise the quantiser"));
            }
            out.extend_from_slice(&(p.len() as u32).to_le_bytes()[..3]);
        }
        for p in &parts {
            out.extend_from_slice(p);
        }
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
        for (y, row) in out.chunks_exact_mut(w).enumerate() {
            let src = &data[y.min(fh - 1) * fw..][..fw];
            row[..fw].copy_from_slice(src);
            row[fw..].fill(src[fw - 1]);
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

/// Checks that `w`x`h` blocks with these strides fit both slices.
#[inline]
fn check_blocks(a: &[u8], astride: usize, b: &[u8], bstride: usize, w: usize, h: usize) {
    assert!(w > 0 && h > 0);
    assert!((h - 1) * astride + w <= a.len() && (h - 1) * bstride + w <= b.len());
}

/// The sum of squared differences of two `w`x`h` blocks.
#[inline]
fn sse(a: &[u8], astride: usize, b: &[u8], bstride: usize, w: usize, h: usize) -> u64 {
    check_blocks(a, astride, b, bstride, w, h);
    // SAFETY: both blocks lie inside their slices (checked above).
    unsafe { (dsp().sse)(a.as_ptr(), astride, b.as_ptr(), bstride, w, h) }
}

/// The sum of absolute differences of two `w`x`h` blocks.
#[inline]
fn sad(a: &[u8], astride: usize, b: &[u8], bstride: usize, w: usize, h: usize) -> u32 {
    check_blocks(a, astride, b, bstride, w, h);
    // SAFETY: both blocks lie inside their slices (checked above).
    unsafe { (dsp().sad)(a.as_ptr(), astride, b.as_ptr(), bstride, w, h) }
}

/// Quantises `c` (a forward DCT or WHT, raster order) from coefficient
/// `first` on; returns the levels and the dequantised values the decoder
/// will reconstruct from them.
#[inline]
fn quantise(c: &[i16; 16], q: &QuantParams, first: usize) -> ([i16; 16], [i16; 16]) {
    debug_assert!(first <= 1);
    (dsp().quant)(c, q, first == 1)
}

/// Transforms and quantises the residue of one 4x4 block.
fn code_4x4(
    src: &[u8; 16],
    pred: &[u8; 16],
    q: &QuantParams,
    first: usize,
) -> ([i16; 16], [i16; 16]) {
    let res: [i16; 16] = std::array::from_fn(|i| src[i] as i16 - pred[i] as i16);
    quantise(&forward_dct(&res), q, first)
}

/// Codes a 16x16 luma residue with a Y2 block: returns the reconstruction
/// and an estimate of the rate (1/256 bits); `levels` receives Y 0-15 and
/// Y2 (24).
fn code_luma_y2(
    src: &[u8; 256],
    pred: &[u8; 256],
    q: &Quantisers,
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
    let (y2, y2deq) = quantise(&forward_wht(&dcs), &q.y2, 0);
    levels[24] = y2;
    let dc = inverse_wht(&y2deq);
    let mut rate = block_rate(&y2, 0);
    let mut rec = *pred;
    for b in 0..16 {
        let (lv, mut deq) = quantise(&coefs[b], &q.y, 1);
        levels[b] = lv;
        rate += block_rate(&lv, 1);
        deq[0] = dc[b];
        let off = 4 * (b >> 2) * 16 + 4 * (b & 3);
        add_residue(&deq, &mut rec, off, 16);
    }
    (rec, rate)
}

/// Codes both chroma residues from their predictions into the work
/// buffer's chroma and `levels` 16-23.
fn code_chroma(
    w: &mut MbWork,
    su: &[u8; 64],
    sv: &[u8; 64],
    pu: &[u8; 64],
    pv: &[u8; 64],
    q: &QuantParams,
    levels: &mut [[i16; 16]; 25],
) {
    for (k, (s, p)) in [(su, pu), (sv, pv)].into_iter().enumerate() {
        let mut rec = *p;
        for b in 0..4 {
            let (bx, by) = (b & 1, b >> 1);
            let sb: [u8; 16] = std::array::from_fn(|i| s[(4 * by + i / 4) * 8 + 4 * bx + i % 4]);
            let pb: [u8; 16] = std::array::from_fn(|i| p[(4 * by + i / 4) * 8 + 4 * bx + i % 4]);
            let (lv, deq) = code_4x4(&sb, &pb, q, 0);
            levels[16 + 4 * k + b] = lv;
            add_residue(&deq, &mut rec, 4 * by * 8 + 4 * bx, 8);
        }
        w.put_chroma(1 + k, &rec);
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
/// one macroblock, in order, from the contexts above and to the left
/// (section 13.3), which it updates as the decoder does.
fn mb_tokens(
    code: &MbCode,
    a: &mut [u8; 9],
    left: &mut [u8; 9],
    emit: &mut impl FnMut(usize, Slot, bool),
) {
    let has_y2 = code.info.ymode != B_PRED && code.info.ymode != SPLITMV;
    if code.info.skip {
        // No tokens: the contexts become empty; a macroblock without Y2
        // leaves the Y2 context alone.
        let (kl, ka) = (left[8], a[8]);
        *left = [0; 9];
        *a = [0; 9];
        if !has_y2 {
            left[8] = kl;
            a[8] = ka;
        }
        return;
    }
    let lv = &code.levels;
    let (ytype, yfirst) = if has_y2 { (0, 1) } else { (3, 0) };
    if has_y2 {
        let nz = block_tokens(&lv[24], 1, 0, (a[8] + left[8]) as usize, emit);
        a[8] = nz as u8;
        left[8] = nz as u8;
    }
    for b in 0..16 {
        let (x, y) = (b & 3, b >> 2);
        let nz = block_tokens(&lv[b], ytype, yfirst, (a[x] + left[y]) as usize, emit);
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
                emit,
            );
            a[off + x] = nz as u8;
            left[off + y] = nz as u8;
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
    let paths = &TREES.coeff;
    let put = |token: u8,
               start: usize,
               band: usize,
               ctx: usize,
               emit: &mut dyn FnMut(usize, Slot, bool)| {
        for &(node, bit) in paths[start / 2].path(token) {
            emit(t, Slot::Tree(band, ctx, node as usize >> 1), bit);
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
        cost(p[MVP_IS_SHORT], false) + TREES.small_mv.cost(&p[MVP_SHORT..MVP_SHORT + 7], a as u8)
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
        let (lv, deq) = quantise(&c, &quant_params([10, 20]), 0);
        for i in 0..16 {
            let step = if i == 0 { 10 } else { 20 };
            assert_eq!(deq[i], lv[i] * step);
        }
    }
}
