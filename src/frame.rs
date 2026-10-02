//! The picture type the decoder hands back and the encoder takes.

use crate::error::{Result, config};

/// One plane of a [`Frame`]: where it sits in the frame's data. Samples are
/// 8-bit and tightly packed (stride == width).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Byte offset of the plane's first sample in [`Frame::data`].
    pub offset: usize,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

impl Plane {
    /// The plane's size in bytes.
    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Whether the plane has no samples (never, for a frame this crate
    /// makes).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An 8-bit 4:2:0 picture: one buffer holding the planes one after the
/// other (Y, then U, then V), each tightly packed. The chroma planes are
/// `(width + 1) / 2` by `(height + 1) / 2` — the layout of a packed I420
/// frame, which is what the VP8 test vectors' MD5s hash, so
/// [`Self::packed`] is that frame without a copy.
///
/// VP8 is 4:2:0 only (RFC 6386 section 2), so there is no chroma-format
/// field; the shape otherwise mirrors `h26x::Picture`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Luma width (the display width; the coded width rounded up to whole
    /// macroblocks is internal to the decoder).
    pub width: u32,
    /// Luma height.
    pub height: u32,
    /// The samples of every plane, packed.
    pub data: Vec<u8>,
    /// Y, then U, then V.
    pub planes: [Plane; 3],
}

impl Frame {
    /// A frame of the given size, mid-grey (128 in every plane). Fails for a
    /// zero dimension or one above 16383, the largest VP8 can code.
    pub fn new(width: u32, height: u32) -> Result<Frame> {
        check_size(width, height)?;
        let planes = layout(width, height);
        let len = planes[2].offset + planes[2].len();
        Ok(Frame {
            width,
            height,
            data: vec![128; len],
            planes,
        })
    }

    /// A frame from three planes, each tightly packed: `y` is `width *
    /// height` bytes, `u` and `v` `((width + 1) / 2) * ((height + 1) / 2)`.
    pub fn from_planes(width: u32, height: u32, y: &[u8], u: &[u8], v: &[u8]) -> Result<Frame> {
        check_size(width, height)?;
        let planes = layout(width, height);
        for (p, s, name) in [
            (&planes[0], y, "Y"),
            (&planes[1], u, "U"),
            (&planes[2], v, "V"),
        ] {
            if s.len() != p.len() {
                return Err(config(format!(
                    "{name} plane is {} bytes, a {width}x{height} frame needs {}",
                    s.len(),
                    p.len()
                )));
            }
        }
        let mut data = Vec::with_capacity(planes[2].offset + planes[2].len());
        data.extend_from_slice(y);
        data.extend_from_slice(u);
        data.extend_from_slice(v);
        Ok(Frame {
            width,
            height,
            data,
            planes,
        })
    }

    /// A frame from a packed I420 buffer (Y, U, V back to back, as
    /// [`Self::packed`] returns and `.yuv` files hold).
    pub fn from_packed(width: u32, height: u32, data: Vec<u8>) -> Result<Frame> {
        check_size(width, height)?;
        let planes = layout(width, height);
        let len = planes[2].offset + planes[2].len();
        if data.len() != len {
            return Err(config(format!(
                "packed frame is {} bytes, a {width}x{height} I420 frame is {len}",
                data.len()
            )));
        }
        Ok(Frame {
            width,
            height,
            data,
            planes,
        })
    }

    /// The samples of plane `i` (0 Y, 1 U, 2 V).
    pub fn plane(&self, i: usize) -> &[u8] {
        let p = &self.planes[i];
        &self.data[p.offset..p.offset + p.len()]
    }

    /// The samples of plane `i`, mutable.
    pub fn plane_mut(&mut self, i: usize) -> &mut [u8] {
        let p = self.planes[i];
        &mut self.data[p.offset..p.offset + p.len()]
    }

    /// The planes concatenated: Y then U then V — a packed I420 frame.
    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    /// The packed planes, taking the buffer.
    pub fn into_packed(self) -> Vec<u8> {
        self.data
    }
}

fn check_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(config(format!(
            "frame size {width}x{height}: VP8 codes 1 to 16383 samples each way"
        )));
    }
    Ok(())
}

fn layout(width: u32, height: u32) -> [Plane; 3] {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let ylen = width as usize * height as usize;
    let clen = cw as usize * ch as usize;
    [
        Plane {
            offset: 0,
            width,
            height,
        },
        Plane {
            offset: ylen,
            width: cw,
            height: ch,
        },
        Plane {
            offset: ylen + clen,
            width: cw,
            height: ch,
        },
    ]
}
