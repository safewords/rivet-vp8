//! A VP8 decoder and encoder.
//!
//! Rust, no C, no system libraries, no build script. Written from RFC 6386
//! (*VP8 Data Format and Decoding Guide*), its prose and tables — not from
//! the reference decoder whose source the RFC attaches, and not from any
//! other implementation.
//!
//! - [`Decoder`] decodes every feature of the format: key and inter frames,
//!   all intra and inter prediction modes, split motion vectors, six-tap and
//!   bilinear interpolation and whole-pixel chroma (versions 0-3), golden
//!   and altref references with sign bias, segmentation, loop filter deltas,
//!   the normal and simple loop filters, multiple token partitions, frame
//!   probability persistence, and any frame size. It is bit-exact on the
//!   VP8 comprehensive test vectors.
//! - [`Encoder`] writes key frames (16x16 and 4x4 intra modes chosen by
//!   rate-distortion cost) and inter frames (a motion search down to
//!   quarter samples against the last frame), at a fixed quantiser, with
//!   token probabilities re-estimated per frame and 1 to 8 token
//!   partitions.
//! - [`ivf`] reads and writes the IVF container.
//!
//! Pictures in and out are [`Frame`]s: 8-bit 4:2:0 planar, the planes
//! packed one after the other (I420), the only format VP8 codes.
//!
//! ```no_run
//! # fn main() -> vp8::Result<()> {
//! let picture = vp8::Frame::new(320, 240)?;
//! let mut enc = vp8::Encoder::new(vp8::Config { width: 320, height: 240, ..Default::default() })?;
//! let packet = enc.encode(&picture)?;
//! let mut dec = vp8::Decoder::new();
//! let decoded = dec.decode(&packet)?.expect("shown");
//! assert_eq!((decoded.width, decoded.height), (320, 240));
//! # Ok(()) }
//! ```

#![warn(missing_docs)]
// The RFC's tables are indexed by several positions at once (plane type,
// band, context, node); index loops say that more plainly than zipped
// iterators.
#![allow(clippy::needless_range_loop)]

mod boolcoder;
mod decoder;
mod dsp;
mod encoder;
mod error;
mod frame;
pub mod ivf;
mod loopfilter;
mod pool;
mod predict;
mod recon;
mod tables;
mod tables_rfc;
mod transform;

pub use decoder::Decoder;
pub use dsp::simd_level;
pub use encoder::{Config, Encoder};
pub use error::{Error, Result};
pub use frame::{Frame, Plane};
