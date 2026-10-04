# rivet-vp8

[![CI](https://github.com/safewords/rivet-vp8/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-vp8/actions/workflows/ci.yml)

A **VP8** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from RFC 6386 — its
prose and tables, not the reference decoder source the RFC attaches — and
not translated from any other implementation. The decoder is **bit-exact**
on all 62 public VP8 test vectors, every frame (the figures are
[below](#how-it-is-checked)). SIMD kernels (SSE4.1, AVX2, NEON, chosen at
run time) and threads make it [fast](#speed).

Written for the **[rivet](https://github.com/safewords/rivet)**
transcoder, where it is the VP8 codec on both sides: the decoder for VP8 in
WebM / Matroska / IVF sources, and the encoder behind VP8 output. Usable on
its own by anything that has VP8 frames and wants planar pictures back, or
pictures and wants VP8.

Published as `rivet-vp8`; **imported as `vp8`** (`use vp8::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
vp8 = { package = "rivet-vp8", git = "https://github.com/safewords/rivet-vp8", branch = "develop" }
```

## What it decodes

All of VP8. `Error::Unsupported` is kept for the two things RFC 6386
reserves: bitstream versions above 3 and colour space 1.

| | |
|---|---|
| **Frames** | key frames and inter frames; hidden frames (`show_frame` 0: they update references and return no picture); any size from 1x1 to 16383x16383, odd sizes included; size changes at key frames |
| **Intra** | the four 16x16 luma and chroma modes, B_PRED with all ten subblock modes and their key-frame contexts, the 127 / 129 frame edges and the above-right rule |
| **Inter** | last, golden and altref references with sign bias; NEAREST, NEAR, ZERO, NEW and SPLITMV (all four partitionings, LEFT / ABOVE / ZERO / NEW subblock vectors); quarter-sample luma and eighth-sample chroma vectors; the six-tap filters (version 0), bilinear (versions 1-2), whole-sample chroma (version 3); vectors pointing anywhere outside the frame |
| **Residue** | token partitions (1, 2, 4 or 8), coefficient probability updates and their persistence or one-frame use (`refresh_entropy_probs`), the Y2 block and the inverse WHT, the inverse DCT, the six dequantisation factors |
| **Segmentation** | segment maps (updated or persisting), quantiser and loop filter levels per segment, absolute or delta |
| **Loop filter** | normal and simple filters, sharpness, per-reference and per-mode level deltas, inner-edge skipping |
| **References** | golden / altref refresh, the copy rules (last or the other reference into golden / altref), `refresh_last` |

Output is a `Frame`: 8-bit 4:2:0, the Y, U and V planes packed one after
the other (I420), chroma `(width + 1) / 2` by `(height + 1) / 2` — the
layout `h26x::Picture` uses, minus the fields VP8 does not need. The
key frame's upscaling codes (RFC 6386 section 9.1) are reported by
`Decoder::scaling()` and not applied: pictures come out at the coded size,
and upscaling is a display matter the RFC leaves to the player.

Malformed input gives an `Error`, never a panic (property-tested; see
below). An inter frame before the first key frame, a cut header, a missing
start code or a partition running past the frame are errors; data after the
end of a token partition reads as zeros, as the coder's arithmetic implies.

## What it encodes

Key frames and inter frames, at a fixed quantiser — the rate control is the
quantiser index, 0 to 127, chosen in `Config`:

- **Key frames**: every macroblock tries the four 16x16 luma modes and
  B_PRED (each subblock trying all ten subblock modes with its real context
  probabilities), coding each candidate in full and keeping the lowest
  rate-distortion cost; chroma takes the mode with the least prediction
  error.
- **Inter frames**: a motion search against the last frame — a coarse grid
  over the search range, then a descending diamond down to quarter samples —
  around the zero vector and the neighbours' vectors; the macroblock takes
  ZEROMV, NEARESTMV, NEARMV, NEWMV or a 16x16 intra mode, whichever predicts
  best for its rate.
- **Entropy**: token probabilities are re-estimated from each frame's own
  statistics and sent as updates where they save bits; skip flags and the
  intra / inter probability are measured per frame.
- **Loop filter**: the normal filter, its level derived from the quantiser
  (or set in `Config`).

Token partitions: 1, 2, 4 or 8, macroblock rows taking them in turn.

Not used (all optional for an encoder): SPLITMV, golden and altref
references, B_PRED in inter frames, segmentation, loop filter deltas, mode
and motion vector probability updates, bit-rate targeting. The encoder decodes every frame it writes with this
crate's decoder and predicts from that decoder's references, so encoder and
decoder cannot drift.

## Speed

The pixel work runs in SIMD kernels — six-tap and bilinear interpolation,
the inverse DCT and its add, the loop filters (16 segments of an edge at
a time), and the encoder's SAD, SSE, forward DCT and quantiser — in SSE4.1
and AVX2 on x86-64 and NEON on aarch64, picked once at run time from what
the CPU has. Each reproduces its scalar reference bit for bit (tested on
every CI host); `VP8_FORCE_SCALAR=1` in the environment keeps the scalar
ones. `vp8::simd_level()` names the set in use.

Both sides use threads, as many as asked for, with the same output for any
count:

- `Decoder::with_threads(n)` (or `set_threads`; 0 = one per CPU): rows of
  macroblocks are reconstructed and loop filtered in a wavefront. Rows
  reconstruct in parallel as far as the stream's token partitions allow
  (row r reads partition r mod the count); with one partition the loop
  filter still runs alongside reconstruction.
- `Config::threads` for the encoder: macroblocks are decided in a
  wavefront, rows in parallel, and the token partitions are written in
  parallel. More `token_partitions` also make the stream faster to decode
  on several threads.

Measured on a Ryzen 9 9950X (16 cores, Windows 11), 30 frames of a camera
clip scaled to each size, quantiser 40, frames per second (the fastest of
three runs; "before" is this crate before the SIMD and thread work, decoding
the same stream):

| | before | 1 thread, scalar | 1 thread, SIMD | 16 threads |
|---|---|---|---|---|
| decode 1280x720 | 144 | 132 | 468 | 1226 (8 partitions), 617 (1) |
| decode 1920x1080 | 62 | 48 | 213 | 537 (8 partitions), 259 (1) |
| encode 1280x720 | 7.6 | 12.6 | 50 | 181 (8 partitions) |
| encode 1920x1080 | 2.9 | 4.7 | 24 | 96 (8 partitions) |

The encoder's gains are only partly SIMD: its forward DCT is now fixed
point, whole-sample motion candidates are compared straight against the
reference, mode and vector costs are tabulated per frame, and quantisation
multiplies by exact reciprocals. `examples/vp8bench.rs` produces these
figures (`tools/bench.sh SOURCE.y4m` runs the set), and `cargo test
--release --lib kernel_timings -- --ignored --nocapture` times each kernel
in each instruction set.

## How it is checked

- **The comprehensive test vectors** (`tests/vectors.rs`):
  `vp80-00-comprehensive-001` to `-018` from the WebM project's public test
  data, committed in [`tests/data`](tests/data/README.md) with the MD5 of
  every frame they show. **18 of 18 streams, 872 of 872 frames bit-exact.**
  They cover versions 0-3, both loop filters, sharpness, filter deltas,
  segmentation, split vectors, golden and altref references and the copies
  between them, two token partitions, one-frame probabilities,
  175x143 and 1432x888 pictures and a hidden frame. Two things they do not
  exercise are checked otherwise: sign bias (set in none of them) by unit
  tests of the vector census against RFC 6386 section 16.3, and four and
  eight token partitions by the encoder's round trips.
- **The other 44 public test vectors** (`vp80-01` to `vp80-06`: intra,
  inter, segmentation, partitions, sharpness, small sizes), downloaded by
  `tools/fetch-vectors.sh` into `tests/vectors` and checked by the same
  test when present. **44 of 44 streams, 1060 of 1060 frames bit-exact.**
  CI downloads them and runs all 62 with the SIMD kernels and with
  `VP8_FORCE_SCALAR=1`, on x86-64 and arm64.
- **Threads**: every vector is decoded on one thread and on three; the
  encoder's round trips compare one and four encoder threads (the streams
  must be identical) and one and three decoder threads.
- **SIMD kernels** (`src/dsp/tests.rs`): every SSE4.1, AVX2 and NEON kernel
  against its scalar reference on random and extreme input (saturating
  filter sums, overflowing coefficients, every filter index, every
  quantiser step); `VP8_REQUIRE_SIMD=1` turns a missing AVX2 or NEON rung
  into a failure rather than a skip.
- **Encoder round trips** (`tests/roundtrip.rs`): the decoder must
  reproduce the encoder's own reconstruction byte for byte, frame after
  frame (key and inter, odd sizes from 1x1 up, 1 to 8 token partitions,
  forced key frames, through an IVF file), and the reconstruction must be
  near the source. Measured
  2026-10-02 (luma PSNR; kbit/s at 30 frames per second):

  | source | settings | PSNR | rate |
  |---|---|---|---|
  | camera, 320x240, 30 frames (vector 015 decoded) | quantiser 20, one key frame | 42.9 dB | 685 kbit/s |
  | camera, 320x240, 30 frames | quantiser 60, one key frame | 36.8 dB | 264 kbit/s |
  | camera, 320x240, 30 frames | quantiser 60, all key frames | 35.8 dB | 878 kbit/s |
  | synthetic 96x64, key frames | quantiser 4 / 24 / 48 / 80 / 127 | 48.9 / 40.7 / 37.1 / 32.9 / 28.7 dB | — |
  | synthetic 128x96, panning and a moving square, 8 frames | quantiser 30, one key frame | 40.4 dB, 3.6 KB (8.3 KB all-intra) | — |

- **Spec-derived unit tests**: the boolean coder against a bit-at-a-time
  transcription of RFC 6386 section 7.2 and in round trips (with carries
  forced through runs of 0xff), tree coding, the inverse DCT against a
  floating-point inverse DCT, the WHT and DCT round trips and the WHT's
  single-DC shortcut, every intra mode against its defining diagonals, the
  subpixel filters on ramps and far outside the frame, the loop filter's
  thresholds and rounding, the transcribed tables' spot values, every tree
  naming each leaf once, the zig-zag order derived from the zig-zag walk,
  motion vector components through the decoder's reader.
- **Malformed input** (`tests/fuzz.rs`, proptest): arbitrary bytes, and
  valid streams (this encoder's, and a test vector using split vectors and
  golden / altref) with bits flipped, bytes cut, replaced and spliced, decode
  to errors or pictures and never panic — in a debug build too, where
  arithmetic overflow would — and to the same pictures and errors on one
  thread and on three.

All of it runs with `cargo test` (the 44 downloaded vectors are skipped
until fetched); nothing else external is needed.

## Provenance and licensing

Written from RFC 6386 (sections 1-19); **no VP8 implementation's source was
read** — not libvpx, not FFmpeg's, not the reference decoder in the RFC's
section 20 — and no implementation was run, even as a black box: the tests
check against the test vectors' MD5s and this crate's own encoder. The
RFC's tables were transcribed by `tools/tables_from_rfc.py`.
[docs/PROVENANCE.md](docs/PROVENANCE.md) lists every source and every place
the RFC is silent or contradicts itself (the zig-zag order, the Y2 and
chroma dequantisation, the loop filter deltas' order, the motion vector
clamp's margins, the meaning of `segment_feature_mode`), with the choice
made and whether the test vectors confirm it.

**Patents.** Implementations of VP8 may be covered by patents. Nothing here
is a licence to any patent, and the authors make no claim about whether
anyone needs one.

## Using it

```rust
// Decoding an IVF file.
let data = std::fs::read("in.ivf")?;
let mut ivf = vp8::ivf::IvfReader::new(&data[..])?;
let mut dec = vp8::Decoder::with_threads(0); // 0: a thread per CPU
while let Some(frame) = ivf.next_frame()? {
    if let Some(picture) = dec.decode(&frame.data)? {
        // picture.plane(0), plane(1), plane(2): Y, U, V; picture.packed(): I420
    }
}

// Encoding.
let mut enc = vp8::Encoder::new(vp8::Config {
    width: 320, height: 240, quantizer: 40, threads: 4, token_partitions: 4,
    ..Default::default()
})?;
let mut out = vp8::ivf::IvfWriter::new(std::fs::File::create("out.ivf")?, 320, 240, 30, 1, 0)?;
for (t, picture) in pictures.iter().enumerate() {
    out.write_frame(t as u64, &enc.encode(picture)?)?;
}
```

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI
open-source) license, royalty-free, with a commercial-attribution
requirement. See [LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
