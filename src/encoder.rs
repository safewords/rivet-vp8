//! The encoder (stub while the decoder is brought up).
use crate::error::Result;
use crate::frame::Frame;

/// Encoder settings.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// Encoder.
pub struct Encoder {}

impl Encoder {
    /// New.
    pub fn new(_c: Config) -> Result<Self> {
        Ok(Encoder {})
    }
    /// Encode.
    pub fn encode(&mut self, _f: &Frame) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}
