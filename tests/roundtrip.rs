//! Encoder -> decoder round trips: the decoder must reproduce the encoder's
//! own reconstruction exactly, and that reconstruction must be close to the
//! source (PSNR).

use vp8::ivf::{IvfReader, IvfWriter};
use vp8::{Config, Decoder, Encoder, Frame};

/// A synthetic sequence with texture, edges and motion: a panning sine
/// pattern, a square moving diagonally and smooth chroma.
fn picture(w: u32, h: u32, t: u32) -> Frame {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; (w * h) as usize];
    for r in 0..h {
        for c in 0..w {
            let (x, yy) = ((c + 2 * t) as f64, (r + t) as f64);
            let mut v =
                128.0 + 50.0 * (x / 7.0).sin() * (yy / 11.0).cos() + 20.0 * ((x + yy) / 3.0).sin();
            let (sx, sy) = (10 + 3 * t, 6 + 2 * t);
            if c >= sx && c < sx + 24 && r >= sy && r < sy + 20 {
                v = 230.0 - (c - sx) as f64 * 3.0;
            }
            y[(r * w + c) as usize] = v.clamp(0.0, 255.0) as u8;
        }
    }
    let u: Vec<u8> = (0..cw * ch)
        .map(|i| (96 + (i % cw) * 64 / cw + t) as u8)
        .collect();
    let v: Vec<u8> = (0..cw * ch)
        .map(|i| (160 - (i / cw) * 48 / ch) as u8)
        .collect();
    Frame::from_planes(w, h, &y, &u, &v).unwrap()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Encodes `n` frames, decodes them, and returns (luma PSNR of each frame,
/// total bytes).
fn round_trip(cfg: Config, n: u32) -> (Vec<f64>, usize) {
    let (w, h) = (cfg.width, cfg.height);
    let mut enc = Encoder::new(cfg).unwrap();
    let mut dec = Decoder::new();
    let mut psnrs = Vec::new();
    let mut bytes = 0;
    for t in 0..n {
        let src = picture(w, h, t);
        let packet = enc.encode(&src).unwrap();
        bytes += packet.len();
        let out = dec
            .decode(&packet)
            .unwrap()
            .expect("every encoded frame is shown");
        // The decoder agrees with the encoder's reconstruction exactly.
        assert_eq!(
            out,
            enc.reconstruction().unwrap(),
            "frame {t}: decoder and encoder disagree"
        );
        assert_eq!((out.width, out.height), (w, h));
        psnrs.push(psnr(src.plane(0), out.plane(0)));
    }
    (psnrs, bytes)
}

#[test]
fn key_frames_at_several_quantisers() {
    let mut last = f64::MAX;
    for q in [4u8, 24, 48, 80, 127] {
        let cfg = Config {
            width: 96,
            height: 64,
            quantizer: q,
            keyframe_interval: 1,
            ..Default::default()
        };
        let (p, bytes) = round_trip(cfg, 2);
        let mean = p.iter().sum::<f64>() / p.len() as f64;
        println!("key frames q={q}: {mean:.2} dB, {bytes} bytes");
        assert!(
            mean <= last + 0.5,
            "PSNR should fall as the quantiser rises"
        );
        last = mean;
        let floor = match q {
            4 => 47.0,
            24 => 39.5,
            48 => 36.0,
            80 => 32.0,
            _ => 27.5,
        };
        assert!(mean > floor, "q={q}: {mean:.2} dB");
    }
}

#[test]
fn inter_frames_track_motion() {
    let cfg = Config {
        width: 128,
        height: 96,
        quantizer: 30,
        keyframe_interval: 0,
        ..Default::default()
    };
    let (p, bytes) = round_trip(cfg.clone(), 8);
    let mean = p.iter().sum::<f64>() / p.len() as f64;
    println!("inter q=30: {mean:.2} dB, {bytes} bytes; per frame {p:.1?}");
    assert!(p.iter().all(|&x| x > 38.5), "{p:?}");
    // Inter frames cost much less than coding every frame as a key frame.
    let (_, intra_bytes) = round_trip(
        Config {
            keyframe_interval: 1,
            ..cfg
        },
        8,
    );
    println!("same frames all-intra: {intra_bytes} bytes");
    assert!(
        bytes * 10 < intra_bytes * 7,
        "inter {bytes} vs intra {intra_bytes}"
    );
}

#[test]
fn odd_sizes() {
    for (w, h) in [(1, 1), (17, 9), (33, 47), (100, 3)] {
        let cfg = Config {
            width: w,
            height: h,
            quantizer: 20,
            keyframe_interval: 3,
            ..Default::default()
        };
        let (p, _) = round_trip(cfg, 4);
        assert!(p.iter().all(|&x| x > 30.0), "{w}x{h}: {p:?}");
    }
}

#[test]
fn ivf_file_round_trip() {
    let (w, h) = (64, 48);
    let mut enc = Encoder::new(Config {
        width: w,
        height: h,
        keyframe_interval: 4,
        ..Default::default()
    })
    .unwrap();
    let mut ivf = IvfWriter::new(Vec::new(), w as u16, h as u16, 30, 1, 6).unwrap();
    let mut recon = Vec::new();
    for t in 0..6 {
        let packet = enc.encode(&picture(w, h, t)).unwrap();
        ivf.write_frame(t as u64, &packet).unwrap();
        recon.push(enc.reconstruction().unwrap());
    }
    let file = ivf.into_inner();
    let mut r = IvfReader::new(&file[..]).unwrap();
    assert_eq!(&r.header().fourcc, b"VP80");
    let mut dec = Decoder::new();
    let mut i = 0;
    while let Some(f) = r.next_frame().unwrap() {
        assert_eq!(dec.decode(&f.data).unwrap().unwrap(), recon[i]);
        i += 1;
    }
    assert_eq!(i, 6);
}

#[test]
fn forced_key_frame_restarts_decoding() {
    let (w, h) = (48, 32);
    let mut enc = Encoder::new(Config {
        width: w,
        height: h,
        keyframe_interval: 0,
        ..Default::default()
    })
    .unwrap();
    enc.encode(&picture(w, h, 0)).unwrap();
    enc.encode(&picture(w, h, 1)).unwrap();
    enc.force_key_frame();
    let key = enc.encode(&picture(w, h, 2)).unwrap();
    assert_eq!(key[0] & 1, 0, "a key frame");
    // A fresh decoder starts from it.
    let mut dec = Decoder::new();
    assert_eq!(
        dec.decode(&key).unwrap().unwrap(),
        enc.reconstruction().unwrap()
    );
}

#[test]
fn configuration_errors() {
    assert!(Encoder::new(Config::default()).is_err());
    assert!(
        Encoder::new(Config {
            width: 16,
            height: 16,
            quantizer: 128,
            ..Default::default()
        })
        .is_err()
    );
    let mut enc = Encoder::new(Config {
        width: 16,
        height: 16,
        ..Default::default()
    })
    .unwrap();
    assert!(enc.encode(&Frame::new(32, 16).unwrap()).is_err());
}

/// Re-encodes the decoded pictures of a test vector (camera content, 320x240)
/// and checks quality and the size against all-intra coding.
#[test]
fn camera_content() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/vp80-00-comprehensive-015.ivf");
    let data = std::fs::read(path).unwrap();
    let mut r = IvfReader::new(&data[..]).unwrap();
    let mut dec = Decoder::new();
    let mut pictures = Vec::new();
    while let Some(f) = r.next_frame().unwrap() {
        if let Some(p) = dec.decode(&f.data).unwrap() {
            pictures.push(p);
        }
        if pictures.len() == 30 {
            break;
        }
    }
    let (w, h) = (pictures[0].width, pictures[0].height);
    for (q, interval, floor) in [(20u8, 0u32, 42.0), (60, 0, 36.0), (60, 1, 35.0)] {
        let mut enc = Encoder::new(Config {
            width: w,
            height: h,
            quantizer: q,
            keyframe_interval: interval,
            ..Default::default()
        })
        .unwrap();
        let mut d = Decoder::new();
        let mut bytes = 0;
        let mut sum = 0.0;
        for p in &pictures {
            let packet = enc.encode(p).unwrap();
            bytes += packet.len();
            let out = d.decode(&packet).unwrap().unwrap();
            sum += psnr(p.plane(0), out.plane(0));
        }
        let mean = sum / pictures.len() as f64;
        println!(
            "camera 320x240, q={q}, {}: {mean:.2} dB luma, {} kbit/s at 30 fps",
            if interval == 1 {
                "all key frames"
            } else {
                "one key frame"
            },
            bytes * 8 * 30 / pictures.len() / 1000
        );
        assert!(mean > floor, "q={q}: {mean:.2} dB");
    }
}
