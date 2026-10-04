//! The decoder: frame header (RFC 6386 sections 9 and 19.2), macroblock
//! prediction records (sections 10, 11, 16, 17), residue (sections 13-14),
//! reconstruction (sections 12, 14, 18), the loop filter (section 15) and
//! the reference buffers (sections 9.7-9.8).

use crate::boolcoder::BoolDecoder;
use crate::error::{Result, bitstream, unsupported};
use crate::frame::Frame;
use crate::loopfilter::Params;
use crate::pool::{Padded, PanicGuard, Pool, wait};
use crate::predict::{RefPlane, predict_inter};
use crate::recon::{
    CHROMA0, CS, LS, LUMA0, MbWork, SharedFrame, filter_mb, filter_params, reconstruct_intra,
};
use crate::tables::*;
use crate::transform::inverse_wht;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// A motion vector in quarter samples of luma, as coded (section 17.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    pub(crate) const ZERO: Mv = Mv { row: 0, col: 0 };
    fn is_zero(self) -> bool {
        self == Mv::ZERO
    }
}

// Reference frames (section 9.10's prob_intra/prob_last/prob_gf).
pub(crate) const INTRA: u8 = 0;
pub(crate) const LAST: u8 = 1;
pub(crate) const GOLDEN: u8 = 2;
pub(crate) const ALTREF: u8 = 3;

/// The probabilities a frame can update, persistent between frames
/// (sections 13.4, 16.1, 17.2).
#[derive(Clone)]
pub(crate) struct Probs {
    pub coeff: [[[[u8; 11]; 3]; 8]; 4],
    pub ymode: [u8; 4],
    pub uvmode: [u8; 3],
    pub mv: [[u8; MVP_COUNT]; 2],
}

impl Default for Probs {
    fn default() -> Self {
        Probs {
            coeff: DEFAULT_COEFF_PROBS,
            ymode: DEFAULT_YMODE_PROBS,
            uvmode: DEFAULT_UV_MODE_PROBS,
            mv: DEFAULT_MV_PROBS,
        }
    }
}

/// Segment-based adjustments (sections 9.3, 10).
#[derive(Clone, Debug, Default)]
pub(crate) struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    /// Feature values replace the frame's (true) or adjust them (false).
    pub absolute: bool,
    pub quant: [i8; 4],
    pub lf: [i8; 4],
    pub tree_probs: [u8; 3],
}

/// Loop filter level adjustments by reference frame and mode (section 9.4).
#[derive(Clone, Debug, Default)]
pub(crate) struct LfDeltas {
    pub enabled: bool,
    /// Intra, last, golden, altref.
    pub refs: [i8; 4],
    /// B_PRED, ZEROMV, other whole-macroblock inter modes, SPLITMV.
    pub modes: [i8; 4],
}

impl LfDeltas {
    /// The level of a macroblock whose segment-adjusted level is `level`.
    pub(crate) fn apply(&self, level: i32, ref_frame: u8, mode: u8) -> i32 {
        if !self.enabled {
            return level;
        }
        let mut l = level + self.refs[ref_frame as usize] as i32;
        if ref_frame == INTRA {
            if mode == B_PRED {
                l += self.modes[0] as i32;
            }
        } else if mode == ZEROMV {
            l += self.modes[1] as i32;
        } else if mode == SPLITMV {
            l += self.modes[3] as i32;
        } else {
            l += self.modes[2] as i32;
        }
        l.clamp(0, 63)
    }
}

/// The quantiser indices of section 9.6.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QuantIndices {
    pub y_ac: i32,
    pub y_dc_delta: i32,
    pub y2_dc_delta: i32,
    pub y2_ac_delta: i32,
    pub uv_dc_delta: i32,
    pub uv_ac_delta: i32,
}

/// The six dequantisation factors of one quantiser level.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Dequant {
    pub y: [i32; 2],
    pub y2: [i32; 2],
    pub uv: [i32; 2],
}

impl Dequant {
    /// Section 14.1. The Y2 and chroma adjustments are not spelled out in
    /// the RFC's prose (it defers to the reference decoder); these are the
    /// format's long-standing values, confirmed by the test vectors: Y2 DC
    /// doubled, Y2 AC scaled by 155/100 with a floor of 8, chroma DC capped
    /// at 132.
    pub(crate) fn new(q: &QuantIndices, base: i32) -> Dequant {
        let dc = |d: i32| DC_QLOOKUP[(base + d).clamp(0, 127) as usize] as i32;
        let ac = |d: i32| AC_QLOOKUP[(base + d).clamp(0, 127) as usize] as i32;
        Dequant {
            y: [dc(q.y_dc_delta), ac(0)],
            y2: [
                dc(q.y2_dc_delta) * 2,
                (ac(q.y2_ac_delta) * 155 / 100).max(8),
            ],
            uv: [dc(q.uv_dc_delta).min(132), ac(q.uv_ac_delta)],
        }
    }
}

/// The frame header fields (section 19.2).
#[derive(Clone, Debug, Default)]
pub(crate) struct Header {
    pub key_frame: bool,
    pub version: u8,
    pub show_frame: bool,
    pub first_part_size: usize,
    pub width: u32,
    pub height: u32,
    pub horiz_scale: u8,
    pub vert_scale: u8,
    pub color_space: bool,
    pub clamping_type: bool,
    pub simple_filter: bool,
    pub filter_level: u8,
    pub sharpness: u8,
    pub partitions: usize,
    pub quant: QuantIndices,
    pub refresh_entropy: bool,
    pub refresh_golden: bool,
    pub refresh_altref: bool,
    pub copy_to_golden: u8,
    pub copy_to_altref: u8,
    pub sign_bias_golden: bool,
    pub sign_bias_altref: bool,
    pub refresh_last: bool,
    pub mb_no_skip_coeff: bool,
    pub prob_skip: u8,
    pub prob_intra: u8,
    pub prob_last: u8,
    pub prob_golden: u8,
}

/// One plane of a decoded picture: every macroblock of it (the coded size
/// rounded up to 16 luma samples), stride == width.
#[derive(Clone, Default)]
pub(crate) struct PlaneBuf {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl PlaneBuf {
    fn new(width: usize, height: usize) -> Self {
        PlaneBuf {
            data: vec![0; width * height],
            width,
            height,
        }
    }
    pub(crate) fn as_ref(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.data,
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct FrameBuf {
    pub planes: [PlaneBuf; 3],
}

impl FrameBuf {
    pub(crate) fn new(mbw: usize, mbh: usize) -> Self {
        FrameBuf {
            planes: [
                PlaneBuf::new(mbw * 16, mbh * 16),
                PlaneBuf::new(mbw * 8, mbh * 8),
                PlaneBuf::new(mbw * 8, mbh * 8),
            ],
        }
    }

    /// The visible `width` x `height` picture, cropped from the macroblocks.
    pub(crate) fn to_frame(&self, width: u32, height: u32) -> Frame {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
        for (p, (pw, ph)) in self.planes.iter().zip([(w, h), (cw, ch), (cw, ch)]) {
            for r in 0..ph {
                data.extend_from_slice(&p.data[r * p.width..r * p.width + pw]);
            }
        }
        Frame::from_packed(width, height, data).expect("decoded size is valid")
    }
}

/// What the decoder keeps of each macroblock of the frame being decoded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MbInfo {
    pub ymode: u8,
    pub uvmode: u8,
    pub ref_frame: u8,
    /// The macroblock's vector; for SPLITMV, that of its last subblock.
    pub mv: Mv,
    pub mvs: [Mv; 16],
    /// Subblock intra modes, for the key-frame contexts of section 11.3.
    pub bmodes: [u8; 16],
    pub segment: u8,
    pub skip: bool,
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo {
            ymode: DC_PRED,
            uvmode: DC_PRED,
            ref_frame: INTRA,
            mv: Mv::ZERO,
            mvs: [Mv::ZERO; 16],
            bmodes: [B_DC_PRED; 16],
            segment: 0,
            skip: false,
        }
    }
}

/// A VP8 decoder: compressed frames in, in decode order, pictures out.
///
/// ```no_run
/// # fn main() -> vp8::Result<()> {
/// let data = std::fs::read("video.ivf")?;
/// let mut ivf = vp8::ivf::IvfReader::new(&data[..])?;
/// let mut dec = vp8::Decoder::new();
/// while let Some(frame) = ivf.next_frame()? {
///     if let Some(picture) = dec.decode(&frame.data)? {
///         // picture.plane(0), .plane(1), .plane(2): Y, U, V
///     }
/// }
/// # Ok(()) }
/// ```
pub struct Decoder {
    probs: Probs,
    seg: Segmentation,
    lf_deltas: LfDeltas,
    width: u32,
    height: u32,
    mbw: usize,
    mbh: usize,
    bufs: Vec<FrameBuf>,
    last: usize,
    golden: usize,
    altref: usize,
    sign_bias: [bool; 4],
    seg_map: Vec<u8>,
    /// Macroblock info with a one-macroblock border above and to the left
    /// (stride `mbw + 1`); the border stays at its default (intra, zero
    /// vectors, B_DC_PRED).
    mbs: Vec<MbInfo>,
    have_key_frame: bool,
    /// The key frame's upscaling codes (section 9.1).
    scaling: (u8, u8),
    /// The header of the last frame decoded (for the header dump test).
    pub(crate) last_header: Header,
    /// The threads frames are decoded on (none but the caller's for 1).
    pool: Pool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-macroblock reconstruction state shared by the residue decoder.
pub(crate) struct Coeffs {
    /// 25 blocks of 16 dequantised coefficients in raster order: Y 0-15,
    /// U 16-19, V 20-23, Y2 24.
    pub blocks: [[i16; 16]; 25],
    /// Which blocks have anything to inverse-transform.
    pub nonzero: [bool; 25],
}

/// Section 18.1's clamp, in quarter samples: macroblock vectors may point
/// at most this far past the edges of the (macroblock-aligned) frame.
pub(crate) const MV_MARGIN: i32 = 16 * 4;

impl Decoder {
    /// A decoder awaiting its first key frame.
    pub fn new() -> Self {
        Decoder {
            probs: Probs::default(),
            seg: Segmentation::default(),
            lf_deltas: LfDeltas::default(),
            width: 0,
            height: 0,
            mbw: 0,
            mbh: 0,
            bufs: Vec::new(),
            last: 0,
            golden: 0,
            altref: 0,
            sign_bias: [false; 4],
            seg_map: Vec::new(),
            mbs: Vec::new(),
            have_key_frame: false,
            scaling: (0, 0),
            last_header: Header::default(),
            pool: Pool::new(1),
        }
    }

    /// A decoder that decodes each frame on up to `threads` threads, the
    /// caller's included; 0 means one per CPU, 1 decodes on the caller's
    /// thread alone. The output is the same whatever the count.
    ///
    /// Rows of macroblocks are reconstructed and loop filtered in a
    /// wavefront; a frame's rows decode in parallel as far as its token
    /// partitions allow (row `r` reads partition `r` mod the count), and the
    /// loop filter runs alongside reconstruction even with one partition.
    pub fn with_threads(threads: usize) -> Self {
        let mut d = Self::new();
        d.set_threads(threads);
        d
    }

    /// Sets the thread count (see [`Self::with_threads`]).
    pub fn set_threads(&mut self, threads: usize) {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        };
        if threads != self.pool.threads() {
            self.pool = Pool::new(threads);
        }
    }

    /// The thread count frames are decoded with.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// The decoder's threads (the encoder decides macroblocks on them too).
    pub(crate) fn pool(&self) -> &Pool {
        &self.pool
    }

    /// The probabilities in force for the next frame (the encoder codes
    /// against them).
    pub(crate) fn probs(&self) -> &Probs {
        &self.probs
    }

    /// The last-frame reference (the encoder predicts from it).
    pub(crate) fn last_reference(&self) -> &FrameBuf {
        &self.bufs[self.last]
    }

    /// The coded size of the stream (0x0 before the first key frame).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The upscaling the last key frame asks for, horizontal and vertical:
    /// 0 none, 1 by 5/4, 2 by 5/3, 3 by 2 (RFC 6386 section 9.1). Decoding
    /// is unaffected — pictures come out at the coded size — and any
    /// resampler may apply it for display.
    pub fn scaling(&self) -> (u8, u8) {
        self.scaling
    }

    /// The last frame decoded (shown or not), cropped to the coded size,
    /// or `None` before the first key frame.
    pub fn last_frame(&self) -> Option<Frame> {
        self.have_key_frame
            .then(|| self.bufs[self.last].to_frame(self.width, self.height))
    }

    /// Decodes one compressed frame. Returns the picture if the frame is
    /// meant to be shown (`show_frame`), `None` if it only updates the
    /// reference buffers (an altref, say).
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        let (hdr, cur) = self.decode_frame(data)?;
        Ok(hdr
            .show_frame
            .then(|| self.bufs[cur].to_frame(self.width, self.height)))
    }

    /// Decodes a frame and returns its header and the buffer index holding
    /// the reconstruction (valid until the next call).
    pub(crate) fn decode_frame(&mut self, data: &[u8]) -> Result<(Header, usize)> {
        let mut hdr = Header::default();
        if data.len() < 3 {
            return Err(bitstream(format!(
                "frame of {} bytes is shorter than the frame tag",
                data.len()
            )));
        }
        let tag = data[0] as u32 | (data[1] as u32) << 8 | (data[2] as u32) << 16;
        hdr.key_frame = tag & 1 == 0;
        hdr.version = ((tag >> 1) & 7) as u8;
        hdr.show_frame = (tag >> 4) & 1 == 1;
        hdr.first_part_size = (tag >> 5) as usize;
        if hdr.version > 3 {
            return Err(unsupported(format!(
                "bitstream version {} (RFC 6386 defines 0-3)",
                hdr.version
            )));
        }
        let mut pos = 3;
        if hdr.key_frame {
            if data.len() < 10 {
                return Err(bitstream("key frame shorter than its 10-byte header"));
            }
            if data[3..6] != [0x9d, 0x01, 0x2a] {
                return Err(bitstream("key frame start code is not 9d 01 2a"));
            }
            let w = data[6] as u32 | (data[7] as u32) << 8;
            let h = data[8] as u32 | (data[9] as u32) << 8;
            hdr.width = w & 0x3fff;
            hdr.height = h & 0x3fff;
            hdr.horiz_scale = (w >> 14) as u8;
            hdr.vert_scale = (h >> 14) as u8;
            if hdr.width == 0 || hdr.height == 0 {
                return Err(bitstream(format!(
                    "key frame size {}x{}",
                    hdr.width, hdr.height
                )));
            }
            pos = 10;
        } else if !self.have_key_frame {
            return Err(bitstream("inter frame before the first key frame"));
        }
        if data.len() < pos + hdr.first_part_size {
            return Err(bitstream(format!(
                "first partition of {} bytes runs past the {}-byte frame",
                hdr.first_part_size,
                data.len()
            )));
        }
        let first = &data[pos..pos + hdr.first_part_size];
        let rest = &data[pos + hdr.first_part_size..];

        if hdr.key_frame {
            if hdr.width != self.width || hdr.height != self.height || self.bufs.is_empty() {
                self.resize(hdr.width, hdr.height);
            }
            self.scaling = (hdr.horiz_scale, hdr.vert_scale);
            // A key frame restores the decoder's initial state (section 4).
            self.probs = Probs::default();
            self.seg = Segmentation::default();
            self.lf_deltas = LfDeltas::default();
            self.sign_bias = [false; 4];
        } else {
            hdr.width = self.width;
            hdr.height = self.height;
        }

        let mut bd = BoolDecoder::new(first);
        let saved_probs = self.read_header(&mut bd, &mut hdr)?;
        let cur = match self.decode_partitions(&hdr, &mut bd, rest) {
            Ok(cur) => cur,
            Err(e) => {
                if let Some(p) = saved_probs {
                    self.probs = p;
                }
                return Err(e);
            }
        };

        // References (sections 9.7, 9.8). Copies read the buffers as they
        // were before this frame.
        if hdr.key_frame {
            self.last = cur;
            self.golden = cur;
            self.altref = cur;
            self.have_key_frame = true;
        } else {
            let (old_last, old_golden, old_altref) = (self.last, self.golden, self.altref);
            if hdr.refresh_golden {
                self.golden = cur;
            } else {
                match hdr.copy_to_golden {
                    1 => self.golden = old_last,
                    2 => self.golden = old_altref,
                    _ => {}
                }
            }
            if hdr.refresh_altref {
                self.altref = cur;
            } else {
                match hdr.copy_to_altref {
                    1 => self.altref = old_last,
                    2 => self.altref = old_golden,
                    _ => {}
                }
            }
            if hdr.refresh_last {
                self.last = cur;
            }
        }
        if let Some(p) = saved_probs {
            self.probs = p;
        }
        self.last_header = hdr.clone();
        Ok((hdr, cur))
    }

    /// Sets up the token partitions (section 9.5) and decodes the
    /// macroblocks into a free buffer; returns its index.
    fn decode_partitions<'d>(
        &mut self,
        hdr: &Header,
        bd: &mut BoolDecoder<'d>,
        rest: &'d [u8],
    ) -> Result<usize> {
        let np = hdr.partitions;
        let sizes_len = 3 * (np - 1);
        if rest.len() < sizes_len {
            return Err(bitstream(
                "token partition sizes run past the end of the frame",
            ));
        }
        let mut parts = Vec::with_capacity(np);
        let mut off = sizes_len;
        for i in 0..np {
            let size = if i + 1 < np {
                let s = &rest[3 * i..3 * i + 3];
                s[0] as usize | (s[1] as usize) << 8 | (s[2] as usize) << 16
            } else {
                rest.len() - off
            };
            if off + size > rest.len() {
                return Err(bitstream(format!(
                    "token partition {i} runs past the end of the frame"
                )));
            }
            parts.push(BoolDecoder::new(&rest[off..off + size]));
            off += size;
        }

        // A buffer no reference holds receives the new frame.
        let cur = (0..self.bufs.len())
            .find(|&i| i != self.last && i != self.golden && i != self.altref)
            .expect("four buffers, three references");
        let mut frame = std::mem::take(&mut self.bufs[cur]);
        let result = self.decode_macroblocks(hdr, bd, parts, &mut frame);
        self.bufs[cur] = frame;
        result.map(|()| cur)
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.mbw = (width as usize).div_ceil(16);
        self.mbh = (height as usize).div_ceil(16);
        self.bufs = vec![FrameBuf::new(self.mbw, self.mbh); 4];
        self.last = 0;
        self.golden = 0;
        self.altref = 0;
        self.seg_map = vec![0; self.mbw * self.mbh];
        self.mbs = vec![MbInfo::default(); (self.mbw + 1) * (self.mbh + 1)];
    }

    /// The rest of the frame header, from the first partition (section 9).
    /// Returns the probabilities to restore after the frame when its
    /// updates are for this frame only (`refresh_entropy_probs` 0).
    fn read_header(&mut self, bd: &mut BoolDecoder, hdr: &mut Header) -> Result<Option<Probs>> {
        if hdr.key_frame {
            hdr.color_space = bd.flag();
            hdr.clamping_type = bd.flag();
            if hdr.color_space {
                return Err(unsupported("colour space 1 (RFC 6386 reserves it)"));
            }
        }
        // Segmentation (sections 9.3, 19.2).
        self.seg.enabled = bd.flag();
        self.seg.update_map = false;
        if self.seg.enabled {
            self.seg.update_map = bd.flag();
            let update_data = bd.flag();
            if update_data {
                // Section 19.2: 1 means absolute values, 0 deltas. (Section
                // 9.3 states the opposite; the test vectors follow 19.2.)
                self.seg.absolute = bd.flag();
                for q in self.seg.quant.iter_mut() {
                    *q = bd.optional_signed(7) as i8;
                }
                for l in self.seg.lf.iter_mut() {
                    *l = bd.optional_signed(6) as i8;
                }
            }
            if self.seg.update_map {
                for p in self.seg.tree_probs.iter_mut() {
                    *p = if bd.flag() { bd.literal(8) as u8 } else { 255 };
                }
            }
        }
        // Loop filter (section 9.4).
        hdr.simple_filter = bd.flag();
        hdr.filter_level = bd.literal(6) as u8;
        hdr.sharpness = bd.literal(3) as u8;
        self.lf_deltas.enabled = bd.flag();
        if self.lf_deltas.enabled && bd.flag() {
            for d in self
                .lf_deltas
                .refs
                .iter_mut()
                .chain(self.lf_deltas.modes.iter_mut())
            {
                if bd.flag() {
                    *d = bd.signed(6) as i8;
                }
            }
        }
        hdr.partitions = 1 << bd.literal(2);
        // Quantiser indices (section 9.6).
        hdr.quant = QuantIndices {
            y_ac: bd.literal(7) as i32,
            y_dc_delta: bd.optional_signed(4),
            y2_dc_delta: bd.optional_signed(4),
            y2_ac_delta: bd.optional_signed(4),
            uv_dc_delta: bd.optional_signed(4),
            uv_ac_delta: bd.optional_signed(4),
        };
        // Reference updates (sections 9.7, 9.8).
        if hdr.key_frame {
            hdr.refresh_golden = true;
            hdr.refresh_altref = true;
            hdr.refresh_last = true;
            hdr.refresh_entropy = bd.flag();
        } else {
            hdr.refresh_golden = bd.flag();
            hdr.refresh_altref = bd.flag();
            if !hdr.refresh_golden {
                hdr.copy_to_golden = bd.literal(2) as u8;
            }
            if !hdr.refresh_altref {
                hdr.copy_to_altref = bd.literal(2) as u8;
            }
            hdr.sign_bias_golden = bd.flag();
            hdr.sign_bias_altref = bd.flag();
            self.sign_bias[GOLDEN as usize] = hdr.sign_bias_golden;
            self.sign_bias[ALTREF as usize] = hdr.sign_bias_altref;
            hdr.refresh_entropy = bd.flag();
            hdr.refresh_last = bd.flag();
        }
        // Saved before this frame's updates: if refresh_entropy_probs is 0
        // the updates last for this frame only. The caller restores.
        let before = self.probs.clone();
        // Token probability updates (section 13.4).
        for i in 0..4 {
            for j in 0..8 {
                for k in 0..3 {
                    for t in 0..11 {
                        if bd.read(COEFF_UPDATE_PROBS[i][j][k][t]) {
                            self.probs.coeff[i][j][k][t] = bd.literal(8) as u8;
                        }
                    }
                }
            }
        }
        hdr.mb_no_skip_coeff = bd.flag();
        if hdr.mb_no_skip_coeff {
            hdr.prob_skip = bd.literal(8) as u8;
        }
        if !hdr.key_frame {
            hdr.prob_intra = bd.literal(8) as u8;
            hdr.prob_last = bd.literal(8) as u8;
            hdr.prob_golden = bd.literal(8) as u8;
            if bd.flag() {
                for p in self.probs.ymode.iter_mut() {
                    *p = bd.literal(8) as u8;
                }
            }
            if bd.flag() {
                for p in self.probs.uvmode.iter_mut() {
                    *p = bd.literal(8) as u8;
                }
            }
            // Motion vector probability updates (section 17.2).
            for i in 0..2 {
                for j in 0..MVP_COUNT {
                    if bd.read(MV_UPDATE_PROBS[i][j]) {
                        let x = bd.literal(7) as u8;
                        self.probs.mv[i][j] = if x != 0 { x << 1 } else { 1 };
                    }
                }
            }
        }
        Ok((!hdr.refresh_entropy).then_some(before))
    }

    /// Decodes every macroblock of the frame into `frame`: first the modes
    /// and vectors of all of them (the first partition), then the residue,
    /// reconstruction and loop filter, a row at a time, on up to
    /// [`Self::threads`] threads (see `Schedule`).
    fn decode_macroblocks<'d>(
        &mut self,
        hdr: &Header,
        bd: &mut BoolDecoder<'d>,
        parts: Vec<BoolDecoder<'d>>,
        frame: &mut FrameBuf,
    ) -> Result<()> {
        let (mbw, mbh) = (self.mbw, self.mbh);
        if frame.planes[0].width != mbw * 16 || frame.planes[0].height != mbh * 16 {
            *frame = FrameBuf::new(mbw, mbh);
        }
        // Dequantisation and loop filter level by segment (sections 9.3, 9.6).
        let mut dq = [Dequant::default(); 4];
        let mut seg_level = [hdr.filter_level as i32; 4];
        for s in 0..4 {
            let mut q = hdr.quant.y_ac;
            if self.seg.enabled {
                q = if self.seg.absolute {
                    self.seg.quant[s] as i32
                } else {
                    q + self.seg.quant[s] as i32
                };
                let l = seg_level[s];
                seg_level[s] = if self.seg.absolute {
                    self.seg.lf[s] as i32
                } else {
                    l + self.seg.lf[s] as i32
                }
                .clamp(0, 63);
            }
            dq[s] = Dequant::new(&hdr.quant, q.clamp(0, 127));
        }

        let Decoder {
            probs,
            seg,
            seg_map,
            mbs,
            bufs,
            sign_bias,
            lf_deltas,
            last,
            golden,
            altref,
            pool,
            ..
        } = self;
        let ctx = FrameCtx {
            mbw,
            mbh,
            dq: &dq,
            coeff_probs: &probs.coeff,
            refs: [&bufs[*last], &bufs[*golden], &bufs[*altref]],
            filters: if hdr.version == 0 {
                &SIXTAP_FILTERS
            } else {
                &BILINEAR_FILTERS
            },
            full_pixel: hdr.version == 3,
            params: filter_params(hdr.sharpness, hdr.key_frame),
            simple: hdr.simple_filter,
            filter: hdr.filter_level != 0,
        };
        let modes = ModeJob {
            reader: ModeReader {
                seg,
                seg_map,
                mbs,
                probs,
                sign_bias: *sign_bias,
                mbw,
                mbh,
            },
            bd,
            hdr,
            seg_level,
            lf_deltas,
        };
        let shared = SharedFrame::new(frame);
        let schedule = Schedule::new(&ctx, modes, parts);
        schedule.run(&ctx, &shared, pool);
        drop(schedule);
        if bd.overran() {
            return Err(bitstream("first partition ends before the last macroblock"));
        }
        Ok(())
    }
}

/// What reading a macroblock's modes needs of the decoder's state.
struct ModeReader<'a> {
    seg: &'a Segmentation,
    seg_map: &'a mut [u8],
    /// Macroblock info with a one-macroblock border above and to the left
    /// (stride `mbw + 1`).
    mbs: &'a mut [MbInfo],
    probs: &'a Probs,
    sign_bias: [bool; 4],
    mbw: usize,
    mbh: usize,
}

/// The modes and loop filter levels of one row of macroblocks.
struct RowModes {
    infos: Vec<MbInfo>,
    levels: Vec<u8>,
}

/// The job that reads every macroblock's modes from the first partition.
struct ModeJob<'a, 'd> {
    reader: ModeReader<'a>,
    bd: &'a mut BoolDecoder<'d>,
    hdr: &'a Header,
    seg_level: [i32; 4],
    lf_deltas: &'a LfDeltas,
}

impl ModeJob<'_, '_> {
    /// Reads row `mby`.
    fn row(&mut self, mby: usize) -> RowModes {
        let (mbw, hdr) = (self.reader.mbw, self.hdr);
        let stride = mbw + 1;
        let mut infos = Vec::with_capacity(mbw);
        let mut levels = Vec::with_capacity(mbw);
        for mbx in 0..mbw {
            let info = self.reader.read_mb_header(self.bd, hdr, mbx, mby);
            self.reader.mbs[(mby + 1) * stride + mbx + 1] = info;
            // Section 15: no filtering at all when the frame's level is 0,
            // whatever the deltas would add; otherwise a macroblock is
            // skipped when its final level (segment, then deltas) is 0. (A
            // segment level of 0 raised by the deltas is filtered: vector
            // 013 has such macroblocks.)
            let base = self.seg_level[info.segment as usize];
            let level = if hdr.filter_level == 0 {
                0
            } else {
                self.lf_deltas.apply(base, info.ref_frame, info.ymode)
            };
            infos.push(info);
            levels.push(level as u8);
        }
        RowModes { infos, levels }
    }
}

impl ModeReader<'_> {
    /// The macroblock header: segment, skip flag, modes and vectors
    /// (sections 10, 11, 16, 17; layout in section 19.3).
    fn read_mb_header(
        &mut self,
        bd: &mut BoolDecoder,
        hdr: &Header,
        mbx: usize,
        mby: usize,
    ) -> MbInfo {
        let mut info = MbInfo::default();
        let mi = mby * self.mbw + mbx;
        if self.seg.update_map {
            self.seg_map[mi] = bd.tree(&MB_SEGMENT_TREE, &self.seg.tree_probs, 0);
        }
        info.segment = if self.seg.enabled {
            self.seg_map[mi]
        } else {
            0
        };
        info.skip = hdr.mb_no_skip_coeff && bd.read(hdr.prob_skip);

        let stride = self.mbw + 1;
        let idx = (mby + 1) * stride + mbx + 1;
        if hdr.key_frame {
            info.ymode = bd.tree(&KF_YMODE_TREE, &KF_YMODE_PROBS, 0);
            if info.ymode == B_PRED {
                let above = self.mbs[idx - stride].bmodes;
                let left = self.mbs[idx - 1].bmodes;
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
                    info.bmodes[b] =
                        bd.tree(&BMODE_TREE, &KF_BMODE_PROBS[a as usize][l as usize], 0);
                }
            } else {
                info.bmodes = [implied_bmode(info.ymode); 16];
            }
            info.uvmode = bd.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, 0);
            return info;
        }

        if !bd.read(hdr.prob_intra) {
            // An intra macroblock in an inter frame (section 16.1).
            info.ymode = bd.tree(&YMODE_TREE, &self.probs.ymode, 0);
            if info.ymode == B_PRED {
                for b in 0..16 {
                    info.bmodes[b] = bd.tree(&BMODE_TREE, &BMODE_PROBS, 0);
                }
            } else {
                info.bmodes = [implied_bmode(info.ymode); 16];
            }
            info.uvmode = bd.tree(&UV_MODE_TREE, &self.probs.uvmode, 0);
            return info;
        }

        // Inter prediction (sections 16.2-16.4).
        info.ref_frame = if !bd.read(hdr.prob_last) {
            LAST
        } else if !bd.read(hdr.prob_golden) {
            GOLDEN
        } else {
            ALTREF
        };
        let near = find_near_mvs(self.mbs, idx, stride, info.ref_frame, &self.sign_bias);
        let probs: [u8; 4] = std::array::from_fn(|i| MODE_CONTEXTS[near.cnt[i] as usize][i]);
        info.ymode = bd.tree(&MV_REF_TREE, &probs, 0);
        let (mbw, mbh) = (self.mbw, self.mbh);
        let clamp = |mv: Mv| clamp_mv(mv, mbx, mby, mbw, mbh);
        let best = clamp(near.best);
        match info.ymode {
            NEARESTMV => info.mv = clamp(near.nearest),
            NEARMV => info.mv = clamp(near.near),
            ZEROMV => info.mv = Mv::ZERO,
            NEWMV => {
                let d = read_mv(bd, &self.probs.mv);
                info.mv = clamp(best.add(d));
            }
            _ => {
                // SPLITMV.
                let partition = bd.tree(&MV_PARTITION_TREE, &MV_PARTITION_PROBS, 0) as usize;
                let layout = &MV_PARTITIONS[partition];
                let above = self.mbs[idx - stride].mvs;
                let left = self.mbs[idx - 1].mvs;
                for part in 0..MV_PARTITION_COUNT[partition] {
                    let k = layout
                        .iter()
                        .position(|&p| p as usize == part)
                        .expect("every part has a subblock");
                    let lmv = if k & 3 != 0 {
                        info.mvs[k - 1]
                    } else {
                        left[k + 3]
                    };
                    let amv = if k >= 4 {
                        info.mvs[k - 4]
                    } else {
                        above[k + 12]
                    };
                    let mv = match bd.tree(
                        &SUB_MV_REF_TREE,
                        &SUB_MV_REF_PROBS[split_context(lmv, amv)],
                        0,
                    ) {
                        LEFT4X4 => lmv,
                        ABOVE4X4 => amv,
                        ZERO4X4 => Mv::ZERO,
                        _ => best.add(read_mv(bd, &self.probs.mv)),
                    };
                    for (b, &p) in layout.iter().enumerate() {
                        if p as usize == part {
                            info.mvs[b] = mv;
                        }
                    }
                }
                info.mv = info.mvs[15];
                return info;
            }
        }
        info.mvs = [info.mv; 16];
        info
    }
}

impl Mv {
    pub(crate) fn add(self, d: Mv) -> Mv {
        Mv {
            row: self.row.wrapping_add(d.row),
            col: self.col.wrapping_add(d.col),
        }
    }
}

/// Section 16.4's `vp8_mvCont`: the subblock mode context from the vectors
/// to the left and above.
pub(crate) fn split_context(left: Mv, above: Mv) -> usize {
    if left == above {
        if above.is_zero() { 4 } else { 3 }
    } else if above.is_zero() {
        2
    } else if left.is_zero() {
        1
    } else {
        0
    }
}

/// The subblock mode a whole-macroblock luma mode stands for in key-frame
/// subblock contexts (section 11.3, item 4).
pub(crate) fn implied_bmode(ymode: u8) -> u8 {
    match ymode {
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        _ => B_DC_PRED,
    }
}

/// The outcome of section 16.3's survey of neighbouring vectors.
pub(crate) struct NearMvs {
    pub best: Mv,
    pub nearest: Mv,
    pub near: Mv,
    pub cnt: [u8; 4],
}

/// Section 16.3: the reference vectors and mode census from the
/// macroblocks above, to the left and above-left of `mbs[idx]`.
pub(crate) fn find_near_mvs(
    mbs: &[MbInfo],
    idx: usize,
    stride: usize,
    ref_frame: u8,
    sign_bias: &[bool; 4],
) -> NearMvs {
    let neighbours = [
        (&mbs[idx - stride], 2u8),
        (&mbs[idx - 1], 2),
        (&mbs[idx - stride - 1], 1),
    ];
    let mut mvs = [Mv::ZERO; 4];
    let mut cnt = [0u8; 4];
    let mut n = 0;
    for &(nb, weight) in &neighbours {
        if nb.ref_frame == INTRA {
            continue;
        }
        if nb.mv.is_zero() {
            cnt[0] += weight;
            continue;
        }
        let mut mv = nb.mv;
        if sign_bias[nb.ref_frame as usize] != sign_bias[ref_frame as usize] {
            mv = Mv {
                row: mv.row.wrapping_neg(),
                col: mv.col.wrapping_neg(),
            };
        }
        // A vector equal to the last one entered adds to its weight;
        // otherwise it is a new entry. (Entry 0 is the zero vector, which a
        // non-zero vector never equals.)
        if mv != mvs[n] {
            n += 1;
            mvs[n] = mv;
        }
        cnt[n] += weight;
    }
    // With three distinct vectors, the third merges into "nearest" if
    // equal to it.
    if cnt[3] > 0 && mvs[3] == mvs[1] {
        cnt[1] += 1;
    }
    cnt[3] = neighbours
        .iter()
        .map(|&(nb, w)| if nb.ymode == SPLITMV { w } else { 0 })
        .sum();
    if cnt[2] > cnt[1] {
        cnt.swap(1, 2);
        mvs.swap(1, 2);
    }
    if cnt[1] >= cnt[0] {
        mvs[0] = mvs[1];
    }
    NearMvs {
        best: mvs[0],
        nearest: mvs[1],
        near: mvs[2],
        cnt,
    }
}

/// Section 16.3's `vp8_clamp_mv`: keeps a vector within [`MV_MARGIN`] of
/// the frame for the macroblock at (`mbx`, `mby`).
pub(crate) fn clamp_mv(mv: Mv, mbx: usize, mby: usize, mbw: usize, mbh: usize) -> Mv {
    let to_left = -((mbx * 64) as i32) - MV_MARGIN;
    let to_right = ((mbw - 1 - mbx) * 64) as i32 + MV_MARGIN;
    let to_top = -((mby * 64) as i32) - MV_MARGIN;
    let to_bottom = ((mbh - 1 - mby) * 64) as i32 + MV_MARGIN;
    Mv {
        row: (mv.row as i32).clamp(to_top, to_bottom) as i16,
        col: (mv.col as i32).clamp(to_left, to_right) as i16,
    }
}

/// One motion vector component (section 17.1).
pub(crate) fn read_mv_component(bd: &mut BoolDecoder, p: &[u8; MVP_COUNT]) -> i16 {
    let a = if bd.read(p[MVP_IS_SHORT]) {
        let mut a = 0i32;
        for i in 0..3 {
            a += (bd.read(p[MVP_BITS + i]) as i32) << i;
        }
        for i in (4..10).rev() {
            a += (bd.read(p[MVP_BITS + i]) as i32) << i;
        }
        // Bit 3 is implicit when no higher bit is set (a long value is >= 8).
        if a & 0xfff0 == 0 || bd.read(p[MVP_BITS + 3]) {
            a += 8;
        }
        a
    } else {
        bd.tree(&SMALL_MV_TREE, &p[MVP_SHORT..MVP_SHORT + 7], 0) as i32
    };
    if a != 0 && bd.read(p[MVP_SIGN]) {
        -a as i16
    } else {
        a as i16
    }
}

/// A motion vector difference: row, then column (section 17.2).
fn read_mv(bd: &mut BoolDecoder, p: &[[u8; MVP_COUNT]; 2]) -> Mv {
    let row = read_mv_component(bd, &p[0]);
    let col = read_mv_component(bd, &p[1]);
    Mv { row, col }
}

/// The residue of one macroblock (sections 13, 19.3), dequantised into
/// `c`. Returns whether any block coded a token other than an immediate
/// end of block.
fn read_residual(
    bd: &mut BoolDecoder,
    probs: &[[[[u8; 11]; 3]; 8]; 4],
    has_y2: bool,
    dq: &Dequant,
    above: &mut [u8; 9],
    left: &mut [u8; 9],
    c: &mut Coeffs,
) -> bool {
    let mut coded = false;
    let (ytype, yfirst) = if has_y2 { (0, 1) } else { (3, 0) };
    if has_y2 {
        let ctx = (above[8] + left[8]) as usize;
        let (nz, any) = read_block(bd, &probs[1], 0, ctx, &mut c.blocks[24], dq.y2);
        above[8] = nz as u8;
        left[8] = nz as u8;
        c.nonzero[24] = any;
        coded |= any;
    }
    for b in 0..16 {
        let (x, y) = (b & 3, b >> 2);
        let ctx = (above[x] + left[y]) as usize;
        let (nz, any) = read_block(bd, &probs[ytype], yfirst, ctx, &mut c.blocks[b], dq.y);
        above[x] = nz as u8;
        left[y] = nz as u8;
        c.nonzero[b] = any;
        coded |= any;
    }
    for (base, ctxoff) in [(16, 4), (20, 6)] {
        for b in 0..4 {
            let (x, y) = (b & 1, b >> 1);
            let ctx = (above[ctxoff + x] + left[ctxoff + y]) as usize;
            let (nz, any) = read_block(bd, &probs[2], 0, ctx, &mut c.blocks[base + b], dq.uv);
            above[ctxoff + x] = nz as u8;
            left[ctxoff + y] = nz as u8;
            c.nonzero[base + b] = any;
            coded |= any;
        }
    }
    coded
}

/// One block's tokens (sections 13.2-13.3), dequantised with `dq` (DC, AC)
/// into `out` in raster order. Returns (any non-zero coefficient, any token
/// coded before the end of block).
///
/// The token tree of section 13.2 is walked as nested branches rather than
/// through its array form (`COEFF_TREE`): node `2k` reads probability `k`;
/// `tests::token_walk_matches_the_tree` checks the two agree.
#[inline]
fn read_block(
    bd: &mut BoolDecoder,
    probs: &[[[u8; 11]; 3]; 8],
    first: usize,
    ctx: usize,
    out: &mut [i16; 16],
    dq: [i32; 2],
) -> (bool, bool) {
    let mut i = first;
    let mut ctx = ctx;
    let mut nonzero = false;
    let mut coded = false;
    // After a zero the end-of-block branch is skipped (it cannot follow).
    let mut after_zero = false;
    while i < 16 {
        let p = &probs[COEFF_BANDS[i]][ctx];
        if !after_zero && !bd.read(p[0]) {
            break; // DCT_EOB
        }
        coded = true;
        if !bd.read(p[1]) {
            // DCT_0
            ctx = 0;
            after_zero = true;
            i += 1;
            continue;
        }
        let v = if !bd.read(p[2]) {
            1
        } else if !bd.read(p[3]) {
            if !bd.read(p[4]) {
                2
            } else if !bd.read(p[5]) {
                3
            } else {
                4
            }
        } else {
            let cat = if !bd.read(p[6]) {
                bd.read(p[7]) as usize
            } else if !bd.read(p[8]) {
                2 + bd.read(p[9]) as usize
            } else {
                4 + bd.read(p[10]) as usize
            };
            let mut extra = 0;
            for &p in PCAT[cat] {
                extra = (extra << 1) | bd.read(p) as i32;
            }
            CAT_BASE[cat] + extra
        };
        ctx = if v == 1 { 1 } else { 2 };
        let v = if bd.flag() { -v } else { v };
        // Section 14.1: products are stored as 16-bit signed integers.
        out[ZIGZAG[i]] = (v * dq[(i > 0) as usize]) as i16;
        nonzero = true;
        after_zero = false;
        i += 1;
    }
    (nonzero, coded)
}

/// The chroma vectors of a macroblock: for each chroma subblock, the
/// average of the four luma vectors covering it, in eighths of a chroma
/// sample (section 18.1), whole samples only for version 3.
pub(crate) fn chroma_mvs(info: &MbInfo, full_pixel: bool) -> [(i32, i32); 4] {
    std::array::from_fn(|k| {
        let (bx, by) = (2 * (k & 1), 2 * (k >> 1));
        let blocks = [
            by * 4 + bx,
            by * 4 + bx + 1,
            by * 4 + bx + 4,
            by * 4 + bx + 5,
        ];
        let avg = |f: fn(Mv) -> i32| {
            // Luma vectors doubled to eighth samples, summed, divided by 8.
            let s: i32 = blocks.iter().map(|&b| 2 * f(info.mvs[b])).sum();
            let v = if s >= 0 {
                (s + 4) >> 3
            } else {
                -((-s + 4) >> 3)
            };
            if full_pixel { v & !7 } else { v }
        };
        (avg(|m| m.col as i32), avg(|m| m.row as i32))
    })
}

/// Inter prediction and residue of one macroblock into `w` (sections 14,
/// 18).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_inter(
    w: &mut MbWork,
    reference: &FrameBuf,
    info: &MbInfo,
    mbx: usize,
    mby: usize,
    blocks: &[[i16; 16]; 25],
    nonzero: &[bool; 25],
    filters: &'static [[i32; 6]; 8],
    full_pixel: bool,
) {
    // Luma vectors are doubled to eighth samples (section 18.1).
    let (x0, y0) = (mbx * 16, mby * 16);
    let src = reference.planes[0].as_ref();
    if info.ymode == SPLITMV {
        for b in 0..16 {
            let (x, y) = (x0 + 4 * (b & 3), y0 + 4 * (b >> 2));
            let mv = info.mvs[b];
            predict_inter(
                src,
                &mut w.luma,
                MbWork::sub_off(b),
                LS,
                x as i32,
                y as i32,
                4,
                4,
                2 * mv.col as i32,
                2 * mv.row as i32,
                filters,
            );
        }
    } else {
        let mv = info.mv;
        predict_inter(
            src,
            &mut w.luma,
            LUMA0,
            LS,
            x0 as i32,
            y0 as i32,
            16,
            16,
            2 * mv.col as i32,
            2 * mv.row as i32,
            filters,
        );
    }

    let cmv = chroma_mvs(info, full_pixel);
    let (x0, y0) = (mbx * 8, mby * 8);
    for pi in 1..3 {
        let buf = if pi == 1 { &mut w.u } else { &mut w.v };
        let src = reference.planes[pi].as_ref();
        if cmv.iter().all(|&m| m == cmv[0]) {
            let (mx, my) = cmv[0];
            predict_inter(
                src, buf, CHROMA0, CS, x0 as i32, y0 as i32, 8, 8, mx, my, filters,
            );
        } else {
            for (k, &(mx, my)) in cmv.iter().enumerate() {
                let (x, y) = (x0 + 4 * (k & 1), y0 + 4 * (k >> 1));
                let off = CHROMA0 + 4 * (k >> 1) * CS + 4 * (k & 1);
                predict_inter(src, buf, off, CS, x as i32, y as i32, 4, 4, mx, my, filters);
            }
        }
    }
    w.add_residues(blocks, nonzero, true);
}

/// What every thread decoding a frame shares (read-only).
struct FrameCtx<'a> {
    mbw: usize,
    mbh: usize,
    dq: &'a [Dequant; 4],
    coeff_probs: &'a [[[[u8; 11]; 3]; 8]; 4],
    /// Last, golden, altref.
    refs: [&'a FrameBuf; 3],
    filters: &'static [[i32; 6]; 8],
    full_pixel: bool,
    params: [Params; 64],
    simple: bool,
    /// Whether the loop filter runs at all.
    filter: bool,
}

/// Packs a macroblock's bottom token contexts (section 13.3: four Y
/// columns, two U, two V, Y2) into 9 bits.
fn pack_nz(nz: &[u8; 9]) -> u16 {
    nz.iter()
        .enumerate()
        .fold(0, |a, (i, &v)| a | ((v as u16 & 1) << i))
}

fn unpack_nz(v: u16) -> [u8; 9] {
    std::array::from_fn(|i| ((v >> i) & 1) as u8)
}

/// One unit of work: reading every macroblock's modes, the residue and
/// reconstruction of a row of macroblocks, or the loop filtering of one.
#[derive(Clone, Copy, Debug)]
enum Job {
    Modes,
    Recon(usize),
    Filter(usize),
}

/// How a frame's macroblock rows are spread over threads.
///
/// Reading the modes of every macroblock (M), and each row's
/// reconstruction (R) and loop filtering (F), are jobs; threads take jobs in
/// a fixed order — M, R0, R1, F0, R2, F1, ... — and within a job wait,
/// macroblock by macroblock, for the progress of earlier jobs:
///
/// - M reads the first partition from start to end, publishing each row's
///   modes as it finishes them; R(y) starts once row y's are out.
/// - R(y) at macroblock x reads the unfiltered pixels of macroblocks x - 1
///   to x + 1 of row y - 1 (intra edges, the above-right four), so it waits
///   until R(y - 1) has finished x + 1. Its tokens come from partition y mod
///   P, which row y - P used before it: R(y) starts once R(y - P) is done.
/// - F(y) at macroblock x changes the pixels of macroblock x, three columns
///   of x - 1 and three rows of x in row y - 1. It runs after R(y) and
///   R(y + 1) have read those unfiltered (both have finished x + 1), after
///   F(y - 1) has finished x + 1 (whose left edge reaches into x's
///   columns), and after F(y) at x - 1 (the same thread, just before).
///
/// Every job waits only on jobs before it in the order, and every job taken
/// is being run, so the frame always completes; with one thread the order
/// itself satisfies every wait. The waits are also what keeps the shared
/// picture race-free (see `recon`): two jobs running at once never touch
/// the same pixel unless both only read it.
struct Schedule<'a, 'd> {
    jobs: Vec<Job>,
    modes: Mutex<ModeJob<'a, 'd>>,
    /// Each row's modes, once read.
    rows: Vec<OnceLock<RowModes>>,
    parts: Vec<Mutex<BoolDecoder<'d>>>,
    /// Macroblocks reconstructed, by row.
    recon: Vec<Padded>,
    /// Macroblocks loop filtered, by row.
    filtered: Vec<Padded>,
    /// Each macroblock's bottom token contexts.
    nz: Vec<AtomicU16>,
    /// Whether each macroblock's inner edges are filtered.
    inner: Vec<AtomicBool>,
    next: AtomicUsize,
    /// Set when a thread panics, so that the others stop waiting.
    poisoned: AtomicBool,
}

impl<'a, 'd> Schedule<'a, 'd> {
    fn new(ctx: &FrameCtx, modes: ModeJob<'a, 'd>, parts: Vec<BoolDecoder<'d>>) -> Self {
        let (mbw, mbh) = (ctx.mbw, ctx.mbh);
        let mut jobs = Vec::with_capacity(2 * mbh + 1);
        jobs.push(Job::Modes);
        for y in 0..mbh {
            jobs.push(Job::Recon(y));
            if ctx.filter && y > 0 {
                jobs.push(Job::Filter(y - 1));
            }
        }
        if ctx.filter {
            jobs.push(Job::Filter(mbh - 1));
        }
        Schedule {
            jobs,
            modes: Mutex::new(modes),
            rows: (0..mbh).map(|_| OnceLock::new()).collect(),
            parts: parts.into_iter().map(Mutex::new).collect(),
            recon: (0..mbh).map(|_| Padded(AtomicUsize::new(0))).collect(),
            filtered: (0..mbh).map(|_| Padded(AtomicUsize::new(0))).collect(),
            nz: (0..mbw * mbh).map(|_| AtomicU16::new(0)).collect(),
            inner: (0..mbw * mbh).map(|_| AtomicBool::new(false)).collect(),
            next: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Runs every job on the pool's threads.
    fn run(&self, ctx: &FrameCtx, frame: &SharedFrame, pool: &Pool) {
        // Reconstruction runs on at most as many rows at once as there are
        // token partitions, the filter on about as many again: more threads
        // would only wait.
        let limit = (2 * self.parts.len()).min(ctx.mbh + 1).max(2);
        if pool.threads() == 1 {
            self.work(ctx, frame);
        } else {
            let active = AtomicUsize::new(0);
            pool.run(&|| {
                if active.fetch_add(1, Ordering::Relaxed) < limit {
                    self.work(ctx, frame);
                }
            });
        }
    }

    fn work(&self, ctx: &FrameCtx, frame: &SharedFrame) {
        let _guard = PanicGuard(&self.poisoned);
        let mut w = MbWork::new();
        loop {
            let j = self.next.fetch_add(1, Ordering::Relaxed);
            match self.jobs.get(j) {
                Some(&Job::Modes) => self.read_modes(),
                Some(&Job::Recon(y)) => self.recon_row(ctx, frame, &mut w, y),
                Some(&Job::Filter(y)) => self.filter_row(ctx, frame, y),
                None => return,
            }
        }
    }

    /// Waits until `counter` reaches `target`.
    fn wait(&self, counter: &AtomicUsize, target: usize) {
        wait(counter, target, &self.poisoned);
    }

    /// M: every row's modes, in order.
    fn read_modes(&self) {
        let mut job = self.modes.lock().unwrap_or_else(|e| e.into_inner());
        for (y, slot) in self.rows.iter().enumerate() {
            let row = job.row(y);
            let _ = slot.set(row);
        }
    }

    /// Row `y`'s modes, waiting for M to read them.
    fn row(&self, y: usize) -> &RowModes {
        let mut spins = 0u32;
        loop {
            if let Some(r) = self.rows[y].get() {
                return r;
            }
            if self.poisoned.load(Ordering::Relaxed) {
                panic!("another thread decoding this frame panicked");
            }
            if spins < 200 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// R(y): the residue, prediction and reconstruction of row `y`.
    fn recon_row(&self, ctx: &FrameCtx, frame: &SharedFrame, w: &mut MbWork, y: usize) {
        let (mbw, np) = (ctx.mbw, self.parts.len());
        if y >= np {
            self.wait(&self.recon[y - np], mbw);
        }
        let mut part = self.parts[y % np].lock().unwrap_or_else(|e| e.into_inner());
        let mut coeffs = Coeffs {
            blocks: [[0; 16]; 25],
            nonzero: [false; 25],
        };
        let modes = self.row(y);
        let mut left_nz = [0u8; 9];
        for x in 0..mbw {
            if y > 0 {
                self.wait(&self.recon[y - 1], (x + 2).min(mbw));
            }
            let i = y * mbw + x;
            let info = &modes.infos[x];
            let mut above_nz = if y > 0 {
                unpack_nz(self.nz[i - mbw].load(Ordering::Relaxed))
            } else {
                [0; 9]
            };
            coeffs.blocks = [[0; 16]; 25];
            coeffs.nonzero = [false; 25];
            let has_y2 = info.ymode != B_PRED && info.ymode != SPLITMV;
            let coded = if info.skip {
                // No coefficients: the blocks' contexts become empty; a
                // macroblock without Y2 leaves the Y2 context alone.
                let (keep_left, keep_above) = (left_nz[8], above_nz[8]);
                left_nz = [0; 9];
                above_nz = [0; 9];
                if !has_y2 {
                    left_nz[8] = keep_left;
                    above_nz[8] = keep_above;
                }
                false
            } else {
                read_residual(
                    &mut part,
                    ctx.coeff_probs,
                    has_y2,
                    &ctx.dq[info.segment as usize],
                    &mut above_nz,
                    &mut left_nz,
                    &mut coeffs,
                )
            };
            self.nz[i].store(pack_nz(&above_nz), Ordering::Relaxed);
            if has_y2 && coeffs.nonzero[24] {
                let dc = inverse_wht(&coeffs.blocks[24]);
                for (b, &d) in dc.iter().enumerate() {
                    coeffs.blocks[b][0] = d;
                }
            }
            // SAFETY: the edges read are the unfiltered bottom row of
            // macroblocks x - 1 to x + 1 of row y - 1 (finished: waited for
            // above) and the right column of x - 1 in this row (finished by
            // this thread); F jobs leave them alone until this macroblock
            // and the next are done. The macroblock written is touched by
            // nobody else until this row's progress counts it.
            unsafe { w.load_edges(frame, x, y, mbw) };
            if info.ref_frame == INTRA {
                reconstruct_intra(w, info, x, y, &coeffs.blocks, &coeffs.nonzero);
            } else {
                reconstruct_inter(
                    w,
                    ctx.refs[info.ref_frame as usize - 1],
                    info,
                    x,
                    y,
                    &coeffs.blocks,
                    &coeffs.nonzero,
                    ctx.filters,
                    ctx.full_pixel,
                );
            }
            // SAFETY: as above.
            unsafe { w.store(frame, x, y) };
            self.inner[i].store(!has_y2 || coded, Ordering::Relaxed);
            self.recon[y].store(x + 1, Ordering::Release);
        }
    }

    /// F(y): the loop filter over row `y`.
    fn filter_row(&self, ctx: &FrameCtx, frame: &SharedFrame, y: usize) {
        let (mbw, mbh) = (ctx.mbw, ctx.mbh);
        for x in 0..mbw {
            let ready = (x + 2).min(mbw);
            self.wait(&self.recon[y], ready);
            if y + 1 < mbh {
                self.wait(&self.recon[y + 1], ready);
            }
            if y > 0 {
                self.wait(&self.filtered[y - 1], ready);
            }
            let i = y * mbw + x;
            // Set before R(y) started (waited for above).
            let level = self.rows[y].get().map_or(0, |r| r.levels[x]);
            if level != 0 {
                let inner = self.inner[i].load(Ordering::Relaxed);
                // SAFETY: the waits above (see Schedule) leave this thread
                // the only one touching the macroblock, the columns left of
                // it and the rows above it that the filter reaches.
                unsafe { filter_mb(frame, x, y, &ctx.params[level as usize], inner, ctx.simple) };
            }
            self.filtered[y].store(x + 1, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inter(ref_frame: u8, row: i16, col: i16) -> MbInfo {
        let mv = Mv { row, col };
        MbInfo {
            ymode: NEWMV,
            ref_frame,
            mv,
            mvs: [mv; 16],
            ..MbInfo::default()
        }
    }

    /// A 2x2 grid of macroblock info with a border, the current macroblock
    /// at the bottom right (index 4 + 4 = 8, stride 3), its neighbours set
    /// to `above`, `left`, `above_left`.
    fn census(
        above: MbInfo,
        left: MbInfo,
        above_left: MbInfo,
        ref_frame: u8,
        bias: [bool; 4],
    ) -> NearMvs {
        let mut mbs = vec![MbInfo::default(); 9];
        mbs[5] = above;
        mbs[7] = left;
        mbs[4] = above_left;
        find_near_mvs(&mbs, 8, 3, ref_frame, &bias)
    }

    #[test]
    fn census_weights_and_order() {
        // Section 16.3: above and left weigh 2, above-left 1; equal vectors
        // merge; zero vectors count towards cnt[0].
        let n = census(
            inter(LAST, 4, 8),
            inter(LAST, 4, 8),
            inter(LAST, 0, 0),
            LAST,
            [false; 4],
        );
        assert_eq!(n.cnt, [1, 4, 0, 0]);
        assert_eq!(
            (n.nearest, n.best),
            (Mv { row: 4, col: 8 }, Mv { row: 4, col: 8 })
        );
        // Distinct vectors: the heavier becomes "nearest" (swapped in).
        let n = census(
            inter(LAST, 2, 2),
            inter(LAST, 6, 6),
            inter(LAST, 6, 6),
            LAST,
            [false; 4],
        );
        assert_eq!(n.cnt, [0, 3, 2, 0]);
        assert_eq!(
            (n.nearest, n.near),
            (Mv { row: 6, col: 6 }, Mv { row: 2, col: 2 })
        );
        let n = census(
            inter(LAST, 2, 2),
            inter(LAST, 6, 6),
            inter(LAST, 2, 2),
            LAST,
            [false; 4],
        );
        // Three entries (the third equal to the first, but not to the one
        // before it): the third merges into "nearest".
        assert_eq!(n.cnt, [0, 3, 2, 0]);
        assert_eq!(
            (n.nearest, n.near),
            (Mv { row: 2, col: 2 }, Mv { row: 6, col: 6 })
        );
        // Intra neighbours add nothing; SPLITMV neighbours fill cnt[3].
        let mut split = inter(LAST, 1, 1);
        split.ymode = SPLITMV;
        let n = census(MbInfo::default(), split, split, LAST, [false; 4]);
        assert_eq!(n.cnt, [0, 3, 0, 3]);
        // "best" stays zero while the zero vector outweighs "nearest".
        let n = census(
            inter(LAST, 0, 0),
            inter(LAST, 0, 0),
            inter(LAST, 5, 5),
            LAST,
            [false; 4],
        );
        assert_eq!((n.cnt[0], n.best), (4, Mv::ZERO));
    }

    #[test]
    fn census_applies_sign_bias() {
        // A golden-frame neighbour's vector is negated for a last-frame
        // macroblock when the two references' sign biases differ.
        let bias = [false, false, true, false];
        let n = census(
            inter(GOLDEN, 4, -8),
            MbInfo::default(),
            MbInfo::default(),
            LAST,
            bias,
        );
        assert_eq!(n.nearest, Mv { row: -4, col: 8 });
        let n = census(
            inter(GOLDEN, 4, -8),
            MbInfo::default(),
            MbInfo::default(),
            GOLDEN,
            bias,
        );
        assert_eq!(n.nearest, Mv { row: 4, col: -8 });
        // After negation it can merge with a last-frame neighbour.
        let n = census(
            inter(GOLDEN, 4, -8),
            inter(LAST, -4, 8),
            MbInfo::default(),
            LAST,
            bias,
        );
        assert_eq!(n.cnt[1], 4);
    }

    /// The nested-branch token reader against the generic tree walk, on
    /// random bits at random probabilities.
    #[test]
    fn token_walk_matches_the_tree() {
        let mut seed = 0x5eed_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..2000 {
            let data: Vec<u8> = (0..64).map(|_| rnd() as u8).collect();
            let mut probs = [[[0u8; 11]; 3]; 8];
            for v in probs.iter_mut().flatten().flatten() {
                *v = (rnd() % 255 + 1) as u8;
            }
            let first = (rnd() % 2) as usize;
            let ctx = (rnd() % 3) as usize;
            let mut a = BoolDecoder::new(&data);
            let mut b = BoolDecoder::new(&data);
            let mut out_a = [0i16; 16];
            let got = read_block(&mut a, &probs, first, ctx, &mut out_a, [3, 5]);
            // The reference: section 13's walk through the array tree.
            let mut out_b = [0i16; 16];
            let (mut i, mut c, mut start) = (first, ctx, 0);
            let (mut nz, mut coded) = (false, false);
            while i < 16 {
                let token = b.tree(&COEFF_TREE, &probs[COEFF_BANDS[i]][c], start);
                if token == DCT_EOB {
                    break;
                }
                coded = true;
                if token == 0 {
                    c = 0;
                    start = 2;
                    i += 1;
                    continue;
                }
                let v = if token < DCT_CAT1 {
                    token as i32
                } else {
                    let cat = (token - DCT_CAT1) as usize;
                    let mut extra = 0;
                    for &p in PCAT[cat] {
                        extra = (extra << 1) | b.read(p) as i32;
                    }
                    CAT_BASE[cat] + extra
                };
                c = if v == 1 { 1 } else { 2 };
                let v = if b.flag() { -v } else { v };
                out_b[ZIGZAG[i]] = (v * [3, 5][(i > 0) as usize]) as i16;
                nz = true;
                start = 0;
                i += 1;
            }
            assert_eq!(got, (nz, coded));
            assert_eq!(out_a, out_b);
            assert_eq!(a.read(128), b.read(128), "decoders out of step");
        }
    }

    #[test]
    fn split_contexts() {
        let z = Mv::ZERO;
        let a = Mv { row: 1, col: 0 };
        let b = Mv { row: 0, col: 1 };
        assert_eq!(split_context(z, z), 4);
        assert_eq!(split_context(a, a), 3);
        assert_eq!(split_context(a, z), 2);
        assert_eq!(split_context(z, a), 1);
        assert_eq!(split_context(a, b), 0);
    }

    #[test]
    fn clamp_margins() {
        // 16 samples (64 quarter samples) past the macroblock-aligned edges.
        let far = Mv {
            row: -2000,
            col: 2000,
        };
        assert_eq!(
            clamp_mv(far, 0, 0, 4, 3),
            Mv {
                row: -64,
                col: 3 * 64 + 64
            }
        );
        assert_eq!(
            clamp_mv(far, 3, 2, 4, 3),
            Mv {
                row: -2 * 64 - 64,
                col: 64
            }
        );
    }

    /// Prints the headers of an IVF file's frames: `VP8_DUMP=path cargo
    /// test dump_headers -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_headers() {
        let path = std::env::var("VP8_DUMP").expect("VP8_DUMP names an IVF file");
        let data = std::fs::read(path).unwrap();
        let mut r = crate::ivf::IvfReader::new(&data[..]).unwrap();
        let mut d = Decoder::new();
        let mut i = 0;
        while let Some(f) = r.next_frame().unwrap() {
            i += 1;
            let res = d.decode_frame(&f.data).map(|_| ());
            let h = &d.last_header;
            println!(
                "{i}: key={} ver={} show={} {}x{} scale={},{} simple={} level={} sharp={} parts={} q={:?} refresh_ent={} gold={}/{} alt={}/{} bias={},{} last={} noskip={} seg={:?} lfd={:?} {:?}",
                h.key_frame,
                h.version,
                h.show_frame,
                h.width,
                h.height,
                h.horiz_scale,
                h.vert_scale,
                h.simple_filter,
                h.filter_level,
                h.sharpness,
                h.partitions,
                h.quant,
                h.refresh_entropy,
                h.refresh_golden,
                h.copy_to_golden,
                h.refresh_altref,
                h.copy_to_altref,
                h.sign_bias_golden,
                h.sign_bias_altref,
                h.refresh_last,
                h.mb_no_skip_coeff,
                d.seg,
                d.lf_deltas,
                res.err()
            );
        }
    }
}
