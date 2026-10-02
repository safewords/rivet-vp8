//! The IVF container: a 32-byte file header, then each frame as a 12-byte
//! header (size, 64-bit timestamp) and the compressed frame. All fields are
//! little-endian. It is the container of the VP8 test vectors and the
//! simplest way to store a VP8 stream.

use std::io::{Read, Write};

use crate::error::{Result, bitstream};

/// The file header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IvfHeader {
    /// The codec's FourCC, `VP80` for VP8.
    pub fourcc: [u8; 4],
    /// Frame width.
    pub width: u16,
    /// Frame height.
    pub height: u16,
    /// Frame rate numerator: frames per second are `rate / scale`.
    pub rate: u32,
    /// Frame rate denominator.
    pub scale: u32,
    /// Frame count as written (writers often leave it 0 or stale).
    pub frame_count: u32,
}

/// One frame of an IVF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IvfFrame {
    /// Presentation timestamp in time-base units.
    pub timestamp: u64,
    /// The compressed frame.
    pub data: Vec<u8>,
}

/// Reads frames from an IVF stream.
pub struct IvfReader<R: Read> {
    inner: R,
    header: IvfHeader,
}

impl<R: Read> IvfReader<R> {
    /// Reads the file header.
    pub fn new(mut inner: R) -> Result<Self> {
        let mut h = [0u8; 32];
        inner
            .read_exact(&mut h)
            .map_err(|_| bitstream("IVF file shorter than its 32-byte header"))?;
        if &h[0..4] != b"DKIF" {
            return Err(bitstream("not an IVF file (no DKIF signature)"));
        }
        let header_len = u16::from_le_bytes([h[6], h[7]]) as usize;
        if header_len < 32 {
            return Err(bitstream(format!("IVF header length {header_len}")));
        }
        // Skip any header extension.
        let mut skip = vec![0u8; header_len - 32];
        inner
            .read_exact(&mut skip)
            .map_err(|_| bitstream("IVF header cut short"))?;
        let header = IvfHeader {
            fourcc: [h[8], h[9], h[10], h[11]],
            width: u16::from_le_bytes([h[12], h[13]]),
            height: u16::from_le_bytes([h[14], h[15]]),
            rate: u32::from_le_bytes([h[16], h[17], h[18], h[19]]),
            scale: u32::from_le_bytes([h[20], h[21], h[22], h[23]]),
            frame_count: u32::from_le_bytes([h[24], h[25], h[26], h[27]]),
        };
        Ok(IvfReader { inner, header })
    }

    /// The file header.
    pub fn header(&self) -> &IvfHeader {
        &self.header
    }

    /// The next frame, or `None` at the end of the stream.
    pub fn next_frame(&mut self) -> Result<Option<IvfFrame>> {
        let mut h = [0u8; 12];
        let mut got = 0;
        while got < 12 {
            let n = self.inner.read(&mut h[got..])?;
            if n == 0 {
                if got == 0 {
                    return Ok(None);
                }
                return Err(bitstream("IVF frame header cut short"));
            }
            got += n;
        }
        let size = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize;
        let timestamp = u64::from_le_bytes([h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11]]);
        // Read in steps so a corrupt size cannot demand a huge allocation.
        let mut data = Vec::new();
        let read = (&mut self.inner).take(size as u64).read_to_end(&mut data)?;
        if read != size {
            return Err(bitstream(format!(
                "IVF frame of {size} bytes cut short at {read}"
            )));
        }
        Ok(Some(IvfFrame { timestamp, data }))
    }
}

/// Writes an IVF stream.
pub struct IvfWriter<W: Write> {
    inner: W,
}

impl<W: Write> IvfWriter<W> {
    /// Writes the file header for a VP8 stream of `width` x `height` at
    /// `rate / scale` frames per second, declaring `frame_count` frames.
    pub fn new(
        mut inner: W,
        width: u16,
        height: u16,
        rate: u32,
        scale: u32,
        frame_count: u32,
    ) -> Result<Self> {
        let mut h = [0u8; 32];
        h[0..4].copy_from_slice(b"DKIF");
        h[6..8].copy_from_slice(&32u16.to_le_bytes());
        h[8..12].copy_from_slice(b"VP80");
        h[12..14].copy_from_slice(&width.to_le_bytes());
        h[14..16].copy_from_slice(&height.to_le_bytes());
        h[16..20].copy_from_slice(&rate.to_le_bytes());
        h[20..24].copy_from_slice(&scale.to_le_bytes());
        h[24..28].copy_from_slice(&frame_count.to_le_bytes());
        inner.write_all(&h)?;
        Ok(IvfWriter { inner })
    }

    /// Writes one compressed frame.
    pub fn write_frame(&mut self, timestamp: u64, data: &[u8]) -> Result<()> {
        let size = u32::try_from(data.len()).map_err(|_| bitstream("frame larger than 4 GiB"))?;
        self.inner.write_all(&size.to_le_bytes())?;
        self.inner.write_all(&timestamp.to_le_bytes())?;
        self.inner.write_all(data)?;
        Ok(())
    }

    /// The underlying writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = IvfWriter::new(Vec::new(), 176, 144, 30, 1, 2).unwrap();
        w.write_frame(0, &[1, 2, 3]).unwrap();
        w.write_frame(1, &[]).unwrap();
        let bytes = w.into_inner();
        let mut r = IvfReader::new(&bytes[..]).unwrap();
        assert_eq!(r.header().fourcc, *b"VP80");
        assert_eq!((r.header().width, r.header().height), (176, 144));
        assert_eq!(
            r.next_frame().unwrap().unwrap(),
            IvfFrame {
                timestamp: 0,
                data: vec![1, 2, 3]
            }
        );
        assert_eq!(
            r.next_frame().unwrap().unwrap(),
            IvfFrame {
                timestamp: 1,
                data: vec![]
            }
        );
        assert!(r.next_frame().unwrap().is_none());
    }

    #[test]
    fn truncation_is_an_error() {
        let mut w = IvfWriter::new(Vec::new(), 16, 16, 30, 1, 1).unwrap();
        w.write_frame(0, &[9; 100]).unwrap();
        let bytes = w.into_inner();
        let mut r = IvfReader::new(&bytes[..bytes.len() - 1]).unwrap();
        assert!(r.next_frame().is_err());
        assert!(IvfReader::new(&bytes[..20]).is_err());
    }
}
