//! Every SIMD kernel against the scalar reference, bit for bit, on random
//! and extreme inputs. With `VP8_REQUIRE_SIMD=1` a missing SIMD rung (AVX2
//! on x86-64, NEON on aarch64) fails the tests instead of skipping them.

use super::*;
use crate::tables::{BILINEAR_FILTERS, SIXTAP_FILTERS};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

/// The SIMD kernel sets this machine runs, besides the scalar one.
fn simd_sets() -> Vec<&'static Dsp> {
    #[cfg(target_arch = "x86_64")]
    let v: Vec<&'static Dsp> = x86::sse41().into_iter().chain(x86::avx2()).collect();
    #[cfg(target_arch = "aarch64")]
    let v: Vec<&'static Dsp> = vec![neon::neon()];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let v: Vec<&'static Dsp> = Vec::new();
    let required = std::env::var_os("VP8_REQUIRE_SIMD").is_some_and(|x| x != "0" && !x.is_empty());
    if required {
        let want = if cfg!(target_arch = "x86_64") {
            "avx2"
        } else if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "none"
        };
        assert!(
            v.iter().any(|d| d.name == want),
            "VP8_REQUIRE_SIMD: the {want} kernels are not available on this machine"
        );
    }
    if v.is_empty() {
        println!("no SIMD kernels on this machine: nothing to compare");
    }
    v
}

#[test]
fn dispatch_names_a_set() {
    let name = dsp().name;
    assert!(["scalar", "sse4.1", "avx2", "neon"].contains(&name));
    // The chosen set is the best one unless forced to scalar.
    if std::env::var_os("VP8_FORCE_SCALAR").is_none_or(|v| v == "0" || v.is_empty()) {
        assert_eq!(name, best().name);
    } else {
        assert_eq!(name, "scalar");
    }
}

/// A picture-like buffer: one of several patterns that between them make
/// the filters saturate, the masks vary, and the edges both pass and fail.
fn fill(rng: &mut Rng, buf: &mut [u8], stride: usize) {
    match rng.below(6) {
        0 => buf.iter_mut().for_each(|v| *v = rng.byte()),
        1 => {
            // Smooth with a little noise.
            let base = rng.byte() as i32;
            let (dx, dy) = (rng.below(7) as i32 - 3, rng.below(7) as i32 - 3);
            for (i, v) in buf.iter_mut().enumerate() {
                let (x, y) = ((i % stride) as i32, (i / stride) as i32);
                let n = rng.below(5) as i32 - 2;
                *v = (base + dx * x + dy * y + n).clamp(0, 255) as u8;
            }
        }
        2 => {
            // Extremes only.
            buf.iter_mut()
                .for_each(|v| *v = if rng.below(2) == 0 { 0 } else { 255 });
        }
        3 => {
            // A step at a random place, flat either side.
            let (a, b) = (rng.byte(), rng.byte());
            let at = rng.below(stride as u64) as usize;
            for (i, v) in buf.iter_mut().enumerate() {
                *v = if i % stride < at { a } else { b };
            }
        }
        4 => {
            // Small differences around a level.
            let base = rng.byte() as i32;
            buf.iter_mut()
                .for_each(|v| *v = (base + rng.below(17) as i32 - 8).clamp(0, 255) as u8);
        }
        _ => {
            // Horizontal step.
            let (a, b) = (rng.byte(), rng.byte());
            let at = rng.below((buf.len() / stride) as u64) as usize;
            for (i, v) in buf.iter_mut().enumerate() {
                *v = if i / stride < at { a } else { b };
            }
        }
    }
}

#[test]
fn subpel_matches_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for d in &sets {
        for iter in 0..4000 {
            let (w, h) = [(16, 16), (8, 8), (4, 4), (16, 8), (8, 4)][iter % 5];
            let sstride = w + 5 + SUBPEL_READ + rng.below(8) as usize;
            let mut src = vec![0u8; sstride * (h + 5) + SUBPEL_READ];
            fill(&mut rng, &mut src, sstride);
            let filters = if rng.below(4) == 0 {
                &BILINEAR_FILTERS
            } else {
                &SIXTAP_FILTERS
            };
            let (fx, fy) = (rng.below(8) as usize, rng.below(8) as usize);
            let dstride = w + rng.below(5) as usize;
            let mut want = vec![0xaau8; dstride * h];
            let mut got = want.clone();
            // SAFETY: `src` has h + 5 rows of `sstride` >= SUBPEL_READ
            // bytes, plus SUBPEL_READ after; `dst` has h rows of `dstride`
            // >= w.
            unsafe {
                scalar::subpel(
                    src.as_ptr(),
                    sstride,
                    want.as_mut_ptr(),
                    dstride,
                    w,
                    h,
                    fx,
                    fy,
                    filters,
                );
                (d.subpel)(
                    src.as_ptr(),
                    sstride,
                    got.as_mut_ptr(),
                    dstride,
                    w,
                    h,
                    fx,
                    fy,
                    filters,
                );
            }
            assert_eq!(got, want, "{} subpel {w}x{h} f=({fx},{fy})", d.name);
        }
    }
}

#[test]
fn idct_add_matches_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for d in &sets {
        for iter in 0..20000 {
            let mut c = [0i16; 16];
            let range = [64u64, 512, 4096, 65536][iter % 4];
            for v in c.iter_mut() {
                if rng.below(3) > 0 {
                    *v = (rng.below(2 * range) as i64 - range as i64).clamp(-32768, 32767) as i16;
                }
            }
            if iter % 97 == 0 {
                c = [if iter % 2 == 0 { i16::MAX } else { i16::MIN }; 16];
            }
            let stride = 4 + rng.below(12) as usize;
            let mut want = vec![0u8; stride * 4];
            fill(&mut rng, &mut want, stride);
            let mut got = want.clone();
            // SAFETY: 4 rows of `stride` >= 4 bytes.
            unsafe {
                scalar::idct_add(&c, want.as_mut_ptr(), stride);
                (d.idct_add)(&c, got.as_mut_ptr(), stride);
            }
            assert_eq!(got, want, "{} idct_add {c:?}", d.name);
        }
    }
}

#[test]
fn loop_filters_match_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(0x0dd_ba11_cafe_f00d);
    for d in &sets {
        let kernels: [(&str, EdgeFn, EdgeFn, bool); 6] = [
            ("mb_h", SCALAR.lf_mb_h, d.lf_mb_h, true),
            ("mb_v", SCALAR.lf_mb_v, d.lf_mb_v, false),
            ("sub_h", SCALAR.lf_sub_h, d.lf_sub_h, true),
            ("sub_v", SCALAR.lf_sub_v, d.lf_sub_v, false),
            ("simple_h", SCALAR.lf_simple_h, d.lf_simple_h, true),
            ("simple_v", SCALAR.lf_simple_v, d.lf_simple_v, false),
        ];
        let mut changed = [0usize; 6];
        for iter in 0..30000 {
            let (name, reference, simd, horizontal) = kernels[iter % 6];
            let stride = 32 + rng.below(16) as usize;
            let mut want = vec![0u8; stride * 40];
            fill(&mut rng, &mut want, stride);
            let lim = EdgeLimits {
                edge: rng.below(200) as u8,
                interior: 1 + rng.below(63) as u8,
                hev: rng.below(if iter % 3 == 0 { 64 } else { 4 }) as u8,
            };
            // The edge at line 8 (horizontal) or column 8 (vertical); the
            // second half of the segments either continues the first (a
            // luma edge) or lies elsewhere in the buffer (chroma's V).
            let a = 8 * stride + 8;
            let contiguous = rng.below(2) == 0;
            let b = match (horizontal, contiguous) {
                (true, true) => a + 8,
                (false, true) => a + 8 * stride,
                (true, false) => 24 * stride + 20,
                (false, false) => 24 * stride + 20,
            };
            let before = want.clone();
            let mut got = want.clone();
            // SAFETY: every segment's p3..q3 lie inside the buffer (lines
            // 4-11 or 20-27 / columns 4-27 for horizontal edges, lines 8-31
            // and columns 4-23 for vertical ones).
            unsafe {
                let pw = want.as_mut_ptr();
                reference(pw.add(a), pw.add(b), stride, lim);
                let pg = got.as_mut_ptr();
                simd(pg.add(a), pg.add(b), stride, lim);
            }
            assert_eq!(got, want, "{} {name} {lim:?}", d.name);
            changed[iter % 6] += (want != before) as usize;
        }
        // The inputs exercise the filters: most calls change something.
        assert!(changed.iter().all(|&c| c > 1000), "{changed:?}");
    }
}

#[test]
fn sad_and_sse_match_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(42);
    for d in &sets {
        for iter in 0..6000 {
            let (w, h) = [(16, 16), (8, 8), (4, 4), (16, 8), (8, 16), (4, 8)][iter % 6];
            let (sa, sb) = (w + rng.below(20) as usize, w + rng.below(20) as usize);
            let mut a = vec![0u8; sa * h];
            let mut b = vec![0u8; sb * h];
            fill(&mut rng, &mut a, sa);
            fill(&mut rng, &mut b, sb);
            // SAFETY: h rows of at least w bytes in both.
            unsafe {
                assert_eq!(
                    (d.sad)(a.as_ptr(), sa, b.as_ptr(), sb, w, h),
                    scalar::sad(a.as_ptr(), sa, b.as_ptr(), sb, w, h),
                    "{} sad {w}x{h}",
                    d.name
                );
                assert_eq!(
                    (d.sse)(a.as_ptr(), sa, b.as_ptr(), sb, w, h),
                    scalar::sse(a.as_ptr(), sa, b.as_ptr(), sb, w, h),
                    "{} sse {w}x{h}",
                    d.name
                );
            }
        }
    }
}

#[test]
fn fdct_matches_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(7);
    for d in &sets {
        for iter in 0..20000 {
            let r = [255u64, 32, 4][iter % 3];
            let mut x: [i16; 16] =
                std::array::from_fn(|_| (rng.below(2 * r + 1) as i64 - r as i64) as i16);
            if iter % 50 == 0 {
                x = [if iter % 100 == 0 { 255 } else { -255 }; 16];
            }
            assert_eq!((d.fdct)(&x), scalar::fdct(&x), "{} fdct {x:?}", d.name);
        }
    }
}

/// The fixed-point forward DCT stays within a unit of twice the
/// orthonormal transform, and the decoder's inverse undoes it.
#[test]
fn fdct_is_accurate() {
    let mut rng = Rng(99);
    for _ in 0..5000 {
        let x: [i16; 16] = std::array::from_fn(|_| rng.below(511) as i16 - 255);
        let got = scalar::fdct(&x);
        for u in 0..4 {
            for v in 0..4 {
                let mut s = 0.0f64;
                for y in 0..4 {
                    for xx in 0..4 {
                        let b = |k: usize, n: usize| {
                            let c = if k == 0 { 0.5 } else { 0.5f64.sqrt() };
                            c * (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 8.0).cos()
                        };
                        s += x[y * 4 + xx] as f64 * b(u, y) * b(v, xx);
                    }
                }
                let want = 2.0 * s;
                assert!(
                    (got[u * 4 + v] as f64 - want).abs() <= 1.0,
                    "{x:?} ({u},{v}): {} vs {want}",
                    got[u * 4 + v]
                );
            }
        }
        let back = scalar::inverse_dct(&got);
        for i in 0..16 {
            assert!((back[i] - x[i]).abs() <= 1, "{x:?} -> {back:?}");
        }
    }
}

#[test]
fn quant_matches_scalar() {
    let sets = simd_sets();
    let mut rng = Rng(31337);
    for d in &sets {
        for iter in 0..20000 {
            let step = [4 + rng.below(320) as i32, 4 + rng.below(320) as i32];
            let round = [
                rng.below(step[0] as u64) as i32,
                rng.below(step[1] as u64) as i32,
            ];
            let q = QuantParams::new(step, round);
            let r = [32768u64, 4096, 300][iter % 3];
            let mut c: [i16; 16] = std::array::from_fn(|_| {
                (rng.below(2 * r) as i64 - r as i64).clamp(-32768, 32767) as i16
            });
            if iter % 41 == 0 {
                c = [if iter % 2 == 0 { i16::MIN } else { i16::MAX }; 16];
            }
            let skip = rng.below(2) == 0;
            assert_eq!(
                (d.quant)(&c, &q, skip),
                scalar::quant(&c, &q, skip),
                "{} quant {c:?} {q:?}",
                d.name
            );
        }
    }
}

/// The reciprocal is the division it replaces.
#[test]
fn quant_reciprocal_is_exact() {
    for step in 2..1024i32 {
        let q = QuantParams::new([step; 2], [0; 2]);
        for n in (0..65536u64).step_by(3).chain([65535]) {
            let got = (n * q.recip[0] as u64) >> 32;
            assert_eq!(got, n / step as u64, "{n} / {step}");
        }
    }
}

/// Times every kernel in every set this machine runs:
/// `cargo test --release --lib kernel_timings -- --ignored --nocapture`.
/// Nanoseconds per call, the fastest of several runs.
#[test]
#[ignore]
fn kernel_timings() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut sets: Vec<&'static Dsp> = vec![&SCALAR];
    sets.extend(simd_sets());
    let mut rng = Rng(5);
    let stride = 64;
    let mut src = vec![0u8; stride * 64];
    fill(&mut rng, &mut src, stride);
    let mut pic = vec![0u8; stride * 64];
    for (i, v) in pic.iter_mut().enumerate() {
        *v = (100 + (i % 7) as i32 * 2 + (i / stride % 5) as i32) as u8;
    }
    let coeffs: [i16; 16] = std::array::from_fn(|i| (i as i16 * 37 % 200) - 100);
    let resid: [i16; 16] = std::array::from_fn(|i| (i as i16 * 29 % 120) - 60);
    let qp = QuantParams::new([20, 25], [10, 9]);
    let lim = EdgeLimits {
        edge: 120,
        interior: 20,
        hev: 2,
    };
    let time = |f: &mut dyn FnMut()| {
        let n = 20000;
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
        }
        best
    };
    let names: Vec<&str> = sets.iter().map(|d| d.name).collect();
    println!(
        "{:<28}{}",
        "kernel (ns/call)",
        names.iter().map(|n| format!("{n:>10}")).collect::<String>()
    );
    let row = |label: &str, f: &mut dyn FnMut(&'static Dsp) -> f64| {
        let v: Vec<f64> = sets.iter().map(|d| f(d)).collect();
        println!(
            "{label:<28}{}",
            v.iter().map(|x| format!("{x:>10.1}")).collect::<String>()
        );
    };
    for (w, h, fx, fy, label) in [
        (16, 16, 3, 5, "subpel 16x16 6-tap 2D"),
        (16, 16, 4, 0, "subpel 16x16 6-tap H"),
        (8, 8, 2, 6, "subpel 8x8 6-tap 2D"),
        (4, 4, 1, 7, "subpel 4x4 6-tap 2D"),
    ] {
        row(label, &mut |d| {
            let mut out = [0u8; 256];
            time(&mut || unsafe {
                // SAFETY: the window (h + 5 rows of 32 bytes) is inside `src`.
                (d.subpel)(
                    black_box(src.as_ptr()),
                    stride,
                    out.as_mut_ptr(),
                    16,
                    w,
                    h,
                    fx,
                    fy,
                    &SIXTAP_FILTERS,
                )
            })
        });
    }
    row("subpel 16x16 bilinear 2D", &mut |d| {
        let mut out = [0u8; 256];
        time(&mut || unsafe {
            // SAFETY: as above.
            (d.subpel)(
                black_box(src.as_ptr()),
                stride,
                out.as_mut_ptr(),
                16,
                16,
                16,
                3,
                5,
                &BILINEAR_FILTERS,
            )
        })
    });
    row("idct4x4 + add", &mut |d| {
        let mut px = pic.clone();
        time(&mut || unsafe {
            // SAFETY: 4 rows of 4 inside `px`.
            (d.idct_add)(
                black_box(&coeffs),
                px.as_mut_ptr().add(stride * 8 + 8),
                stride,
            )
        })
    });
    type Pick = fn(&Dsp) -> EdgeFn;
    let edges: [(&str, Pick, bool); 6] = [
        ("loop filter MB edge H", |d| d.lf_mb_h, true),
        ("loop filter MB edge V", |d| d.lf_mb_v, false),
        ("loop filter inner edge H", |d| d.lf_sub_h, true),
        ("loop filter inner edge V", |d| d.lf_sub_v, false),
        ("loop filter simple H", |d| d.lf_simple_h, true),
        ("loop filter simple V", |d| d.lf_simple_v, false),
    ];
    for (label, k, horizontal) in edges {
        row(label, &mut |d| {
            let mut px = pic.clone();
            let f = k(d);
            let a = 16 * stride + 16;
            let b = if horizontal { a + 8 } else { a + 8 * stride };
            time(&mut || unsafe {
                // SAFETY: p3..q3 of all 16 segments lie inside `px`.
                let p = px.as_mut_ptr();
                f(black_box(p.add(a)), p.add(b), stride, lim)
            })
        });
    }
    row("SAD 16x16", &mut |d| {
        time(&mut || unsafe {
            // SAFETY: 16 rows of 16 in both.
            black_box((d.sad)(
                black_box(src.as_ptr()),
                stride,
                pic.as_ptr().add(3),
                stride,
                16,
                16,
            ));
        })
    });
    row("SSE 16x16", &mut |d| {
        time(&mut || unsafe {
            // SAFETY: 16 rows of 16 in both.
            black_box((d.sse)(
                black_box(src.as_ptr()),
                stride,
                pic.as_ptr().add(3),
                stride,
                16,
                16,
            ));
        })
    });
    row("SSE 4x4", &mut |d| {
        time(&mut || unsafe {
            // SAFETY: 4 rows of 4 in both.
            black_box((d.sse)(
                black_box(src.as_ptr()),
                stride,
                pic.as_ptr().add(3),
                stride,
                4,
                4,
            ));
        })
    });
    row("forward DCT 4x4", &mut |d| {
        time(&mut || {
            black_box((d.fdct)(black_box(&resid)));
        })
    });
    row("quantise 4x4", &mut |d| {
        time(&mut || {
            black_box((d.quant)(black_box(&coeffs), &qp, false));
        })
    });
}
