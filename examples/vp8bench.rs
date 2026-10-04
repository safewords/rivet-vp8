//! End-to-end speed: encodes a clip and decodes the result, timing both.
//!
//! ```text
//! cargo run --release --example vp8bench -- SOURCE.y4m [options]
//!   --size WxH      scale the source to this size (default: the source's)
//!   --frames N      frames to use (default: all; the clip repeats if short)
//!   --q Q           quantiser index (default 40)
//!   --threads T     decoder and encoder threads (default 1)
//!   --partitions P  token partitions (default 8 with threads > 1, else 1)
//!   --reps R        decode repetitions; the fastest is reported (default 5)
//!   --enc-reps R    encode repetitions; the fastest is reported (default 1)
//!   --out FILE      write the encoded stream (IVF)
//!   --ivf FILE      decode this stream instead of encoding
//! ```
//!
//! The source is 8-bit 4:2:0 YUV4MPEG2. Scaling is bilinear, done here so
//! that one camera clip serves every size. The encoder's reconstruction of
//! every frame is compared with what the decoder makes of the stream, so a
//! run is also a check. `VP8_FORCE_SCALAR=1` times the scalar kernels.

use std::time::Instant;
use vp8::ivf::{IvfReader, IvfWriter};
use vp8::{Config, Decoder, Encoder, Frame};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut src_path = None;
    let mut size = None;
    let mut frames = usize::MAX;
    let mut q = 40u8;
    let mut threads = 1usize;
    let mut partitions = None;
    let mut reps = 5;
    let mut enc_reps = 1;
    let mut out = None;
    let mut ivf_in = None;
    let mut i = 0;
    while i < args.len() {
        let val = || args.get(i + 1).cloned().expect("option needs a value");
        match args[i].as_str() {
            "--size" => {
                let v = val();
                let (w, h) = v.split_once('x').expect("WxH");
                size = Some((w.parse::<u32>().unwrap(), h.parse::<u32>().unwrap()));
                i += 1;
            }
            "--frames" => {
                frames = val().parse().unwrap();
                i += 1;
            }
            "--q" => {
                q = val().parse().unwrap();
                i += 1;
            }
            "--threads" => {
                threads = val().parse().unwrap();
                i += 1;
            }
            "--partitions" => {
                partitions = Some(val().parse::<u8>().unwrap());
                i += 1;
            }
            "--reps" => {
                reps = val().parse().unwrap();
                i += 1;
            }
            "--enc-reps" => {
                enc_reps = val().parse().unwrap();
                i += 1;
            }
            "--out" => {
                out = Some(val());
                i += 1;
            }
            "--ivf" => {
                ivf_in = Some(val());
                i += 1;
            }
            a => src_path = Some(a.to_string()),
        }
        i += 1;
    }
    let partitions = partitions.unwrap_or(if threads > 1 { 8 } else { 1 });

    let (packets, w, h) = if let Some(path) = ivf_in {
        let data = std::fs::read(&path).expect("read IVF");
        let mut r = IvfReader::new(&data[..]).expect("IVF header");
        let mut packets = Vec::new();
        while let Some(f) = r.next_frame().expect("IVF frame") {
            packets.push(f.data);
        }
        (packets, 0, 0)
    } else {
        let path = src_path.expect("a .y4m source");
        let mut pics = read_y4m(&path, size);
        if frames != usize::MAX {
            let n = pics.len();
            pics = (0..frames).map(|i| pics[i % n].clone()).collect();
        }
        let (w, h) = (pics[0].width, pics[0].height);
        let mut best = f64::MAX;
        let mut packets = Vec::new();
        let mut recons = Vec::new();
        for _ in 0..enc_reps {
            let mut enc = Encoder::new(make_config(w, h, q, partitions, threads)).unwrap();
            packets.clear();
            recons.clear();
            let t = Instant::now();
            for p in &pics {
                packets.push(enc.encode(p).unwrap());
                recons.push(enc.reconstruction().unwrap());
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bytes: usize = packets.iter().map(Vec::len).sum();
        let mut dec = make_decoder(threads);
        let mut psnr = 0.0;
        for ((p, rec), src) in packets.iter().zip(&recons).zip(&pics) {
            let d = dec.decode(p).unwrap().expect("shown");
            assert!(
                d == *rec,
                "decoder output differs from the encoder's reconstruction"
            );
            psnr += psnr_y(src.plane(0), d.plane(0));
        }
        println!(
            "encode {w}x{h} q{q} threads {threads} partitions {partitions}: {} frames, {:.3} s, {:.2} fps, {} bytes ({:.0} kbit/s at 30 fps), Y PSNR {:.3} dB",
            pics.len(),
            best,
            pics.len() as f64 / best,
            bytes,
            bytes as f64 * 8.0 * 30.0 / pics.len() as f64 / 1000.0,
            psnr / pics.len() as f64
        );
        if let Some(o) = &out {
            let f = std::fs::File::create(o).unwrap();
            let mut wr =
                IvfWriter::new(f, w as u16, h as u16, 30, 1, packets.len() as u32).unwrap();
            for (i, p) in packets.iter().enumerate() {
                wr.write_frame(i as u64, p).unwrap();
            }
        }
        (packets, w, h)
    };

    let mut best = f64::MAX;
    let mut digest = 0u64;
    for _ in 0..reps {
        let mut dec = make_decoder(threads);
        let t = Instant::now();
        let mut d = 0u64;
        for p in &packets {
            if let Some(f) = dec.decode(p).unwrap() {
                d = d
                    .wrapping_mul(31)
                    .wrapping_add(f.data.iter().map(|&v| v as u64).sum());
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
        digest = d;
    }
    println!(
        "decode {w}x{h} threads {threads}: {} frames, {:.4} s, {:.1} fps (checksum {digest:x})",
        packets.len(),
        best,
        packets.len() as f64 / best
    );
}

#[allow(unused_variables)]
fn make_config(w: u32, h: u32, q: u8, partitions: u8, threads: usize) -> Config {
    Config {
        width: w,
        height: h,
        quantizer: q,
        token_partitions: partitions,
        threads,
        ..Default::default()
    }
}

#[allow(unused_variables)]
fn make_decoder(threads: usize) -> Decoder {
    Decoder::with_threads(threads)
}

fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
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

/// Reads an 8-bit 4:2:0 Y4M file, scaling each frame to `size`.
fn read_y4m(path: &str, size: Option<(u32, u32)>) -> Vec<Frame> {
    let data = std::fs::read(path).expect("read the source");
    let nl = data.iter().position(|&b| b == b'\n').expect("Y4M header");
    let header = std::str::from_utf8(&data[..nl]).unwrap();
    let (mut w, mut h) = (0usize, 0usize);
    for tok in header.split_whitespace() {
        if let Some(v) = tok.strip_prefix('W') {
            w = v.parse().unwrap();
        } else if let Some(v) = tok.strip_prefix('H') {
            h = v.parse().unwrap();
        } else if let Some(c) = tok.strip_prefix('C') {
            assert!(c.starts_with("420"), "4:2:0 sources only, not C{c}");
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let fsize = w * h + 2 * cw * ch;
    let (tw, th) = size.map_or((w, h), |(a, b)| (a as usize, b as usize));
    let (tcw, tch) = (tw.div_ceil(2), th.div_ceil(2));
    let mut pos = nl + 1;
    let mut out = Vec::new();
    while pos < data.len() {
        let fnl = pos + data[pos..].iter().position(|&b| b == b'\n').unwrap();
        pos = fnl + 1;
        if pos + fsize > data.len() {
            break;
        }
        let f = &data[pos..pos + fsize];
        pos += fsize;
        let y = scale(&f[..w * h], w, h, tw, th);
        let u = scale(&f[w * h..w * h + cw * ch], cw, ch, tcw, tch);
        let v = scale(&f[w * h + cw * ch..], cw, ch, tcw, tch);
        out.push(Frame::from_planes(tw as u32, th as u32, &y, &u, &v).unwrap());
    }
    out
}

/// Bilinear resampling, sample centres aligned.
fn scale(src: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    if (w, h) == (tw, th) {
        return src.to_vec();
    }
    let mut out = vec![0u8; tw * th];
    for y in 0..th {
        let fy = ((y as f64 + 0.5) * h as f64 / th as f64 - 0.5).clamp(0.0, (h - 1) as f64);
        let (y0, wy) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..tw {
            let fx = ((x as f64 + 0.5) * w as f64 / tw as f64 - 0.5).clamp(0.0, (w - 1) as f64);
            let (x0, wx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            let p = |yy: usize, xx: usize| src[yy * w + xx] as f64;
            let v = (p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx) * (1.0 - wy)
                + (p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx) * wy;
            out[y * tw + x] = v.round() as u8;
        }
    }
    out
}
