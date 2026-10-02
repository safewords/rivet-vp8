//! The crate's error type.

/// Errors the decoder, the encoder and the IVF reader can report.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The data breaks the VP8 (or IVF) syntax: a frame shorter than its
    /// header says, a missing start code, a partition that runs past the
    /// end of the frame, an inter frame before any key frame.
    #[error("invalid VP8 data: {0}")]
    Bitstream(String),
    /// Valid data this crate does not implement, named: a bitstream version
    /// above 3 (RFC 6386 reserves them), the reserved colour space.
    #[error("unsupported VP8 feature: {0}")]
    Unsupported(String),
    /// A configuration or input the caller supplied that cannot be coded: a
    /// frame size of zero or above 16383, a frame whose planes do not match
    /// its dimensions, a quantiser out of range.
    #[error("invalid VP8 configuration: {0}")]
    Config(String),
    /// An I/O error from the reader or writer an [`ivf`](crate::ivf) type
    /// wraps.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

// Cold and out of line: errors are built on the failure paths of hot parsing
// loops, and the String construction does not belong inlined there.
#[cold]
#[inline(never)]
pub(crate) fn bitstream(msg: impl Into<String>) -> Error {
    Error::Bitstream(msg.into())
}

#[cold]
#[inline(never)]
pub(crate) fn unsupported(msg: impl Into<String>) -> Error {
    Error::Unsupported(msg.into())
}

#[cold]
#[inline(never)]
pub(crate) fn config(msg: impl Into<String>) -> Error {
    Error::Config(msg.into())
}
