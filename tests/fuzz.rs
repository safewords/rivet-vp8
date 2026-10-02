//! Malformed input: arbitrary bytes, and valid streams with bits flipped,
//! bytes cut and garbage spliced, must give errors or pictures — never a
//! panic. Run in debug builds too, where arithmetic overflow panics.

use proptest::prelude::*;
use vp8::{Config, Decoder, Encoder, Frame};

/// A short valid stream: a key frame then inter frames with motion.
fn stream() -> &'static [Vec<u8>] {
    static S: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let (w, h) = (48u32, 32u32);
        let mut enc = Encoder::new(Config { width: w, height: h, quantizer: 30, keyframe_interval: 0, ..Default::default() }).unwrap();
        (0..4)
            .map(|t| {
                let mut f = Frame::new(w, h).unwrap();
                for (i, p) in f.plane_mut(0).iter_mut().enumerate() {
                    let (x, y) = (i as u32 % w, i as u32 / w);
                    *p = ((x * 7 + y * 3 + t * 5) % 256) as u8 ^ if (x + t) / 8 % 2 == 0 { 0x40 } else { 0 };
                }
                enc.encode(&f).unwrap()
            })
            .collect()
    })
}

/// Frames of a comprehensive test vector (split vectors, golden and altref
/// references, segmentation: syntax the encoder does not write).
fn vector() -> &'static [Vec<u8>] {
    static S: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vp80-00-comprehensive-007.ivf");
        let data = std::fs::read(path).unwrap();
        let mut r = vp8::ivf::IvfReader::new(&data[..]).unwrap();
        let mut frames = Vec::new();
        while let Some(f) = r.next_frame().unwrap() {
            frames.push(f.data);
            if frames.len() == 6 {
                break;
            }
        }
        frames
    })
}

fn decode_all(frames: &[Vec<u8>]) {
    let mut dec = Decoder::new();
    for f in frames {
        let _ = dec.decode(f);
    }
}

/// Applies a mutation to one frame, leaving a key frame's size bytes
/// (6..10) alone so a mutation cannot ask for a gigabyte picture.
fn mutate(frames: &[Vec<u8>], which: usize, kind: u8, pos: usize, val: u8) -> Vec<Vec<u8>> {
    let mut out = frames.to_vec();
    let f = &mut out[which % frames.len()];
    if f.is_empty() {
        return out;
    }
    let mut p = pos % f.len();
    if f[0] & 1 == 0 && (6..10).contains(&p) {
        p = 10.min(f.len() - 1);
    }
    match kind % 4 {
        0 => f[p] ^= 1 << (val % 8),
        1 => f.truncate(p),
        2 => f[p] = val,
        _ => {
            let tail = f.split_off(p);
            f.extend(std::iter::repeat_n(val, (val % 32) as usize));
            f.extend(tail);
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    #[test]
    fn arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        let mut dec = Decoder::new();
        let _ = dec.decode(&data);
        // The same bytes after a valid key frame, as an inter frame.
        let mut dec = Decoder::new();
        dec.decode(&stream()[0]).unwrap();
        let _ = dec.decode(&data);
    }

    #[test]
    fn mutated_encoder_stream(which in 0usize..4, kind in any::<u8>(), pos in any::<usize>(), val in any::<u8>()) {
        decode_all(&mutate(stream(), which, kind, pos, val));
    }

    #[test]
    fn mutated_test_vector(which in 0usize..6, kind in any::<u8>(), pos in any::<usize>(), val in any::<u8>()) {
        decode_all(&mutate(vector(), which, kind, pos, val));
    }

    #[test]
    fn many_flips(flips in proptest::collection::vec((0usize..6, any::<usize>(), 0u8..8), 1..40)) {
        let mut frames = vector().to_vec();
        for (w, pos, bit) in flips {
            let f = &mut frames[w];
            let mut p = pos % f.len();
            if f[0] & 1 == 0 && (6..10).contains(&p) {
                p = 10;
            }
            f[p] ^= 1 << bit;
        }
        decode_all(&frames);
    }
}

#[test]
fn truncated_headers_are_errors() {
    let key = &stream()[0];
    let mut dec = Decoder::new();
    for n in 0..10 {
        assert!(dec.decode(&key[..n]).is_err(), "{n} bytes");
    }
    // An inter frame first.
    assert!(Decoder::new().decode(&stream()[1]).is_err());
    // A bad start code.
    let mut bad = key.clone();
    bad[3] = 0;
    assert!(Decoder::new().decode(&bad).is_err());
}
