# Provenance

Where every part of this crate came from. The short version: the code is
this repository's own, written from the prose and tables of RFC 6386; the
four large tables were transcribed from the RFC's text by a script; and the
only outside data the tests use are the WebM project's public test vectors
(bitstreams and the MD5s of their decoded frames).

## Clean-room rules

- **The source is RFC 6386** (*VP8 Data Format and Decoding Guide*,
  November 2011), fetched as plain text from
  <https://www.rfc-editor.org/rfc/rfc6386.txt>. Sections 1-19 were read: the
  prose, the tables, and the short C fragments the prose uses to state
  algorithms exactly (the boolean coder of section 7.3, the transforms of
  14.3-14.4, the loop filter of 15.2-15.4, the vector census of 16.3, the
  interpolation of 18.3). Those fragments were re-expressed in Rust, not
  copied.
- **Section 20, the reference decoder's source listings attached to the
  RFC, was not read**, and `tools/tables_from_rfc.py` stops reading the RFC
  text where section 20 begins.
- **No VP8 implementation's source was opened, fetched or searched for**:
  not libvpx, not FFmpeg's libavcodec, not any other decoder or encoder. No
  implementation was run either — not even as a black box: the tests
  compare against the test vectors' MD5s and against this crate's own
  encoder and decoder only.
- Where the RFC is silent or contradicts itself, the choice made is listed
  below with the evidence for it. "Confirmed" means the test vectors decode
  bit-exactly with the choice and stop doing so with the alternative (each
  alternative was tried); "not exercised" means the vectors decode
  bit-exactly either way, so the choice rests on the RFC's text alone.

## The RFC, by section

- 7-8: the boolean entropy coder (decoder and encoder) and tree coding.
- 9, 19.1-19.2: the uncompressed chunk and the frame header, field by field.
- 10: segment ids; 9.3, 9.6: segment quantiser and filter levels.
- 11, 16: macroblock prediction records for key and inter frames, the
  subblock mode contexts, the vector census (`find_near_mvs`), split
  vectors and their contexts.
- 12: intra prediction, the 127 / 129 edges, the above-right rule.
- 13: coefficient tokens, bands, contexts, probability updates.
- 14: dequantisation, the inverse WHT and DCT, reconstruction.
- 15: the loop filters and their thresholds.
- 17: motion vector components and probability updates.
- 18: vector bounds, chroma vectors, the six-tap and bilinear filters.

## The tables

Transcribed from the RFC's prose sections:

| table | RFC section | how |
|---|---|---|
| `kf_bmode_prob` (10 x 10 x 9) | 11.5 | `tools/tables_from_rfc.py` |
| `coeff_update_probs` (4 x 8 x 3 x 11) | 13.4 | `tools/tables_from_rfc.py` |
| `default_coeff_probs` (4 x 8 x 3 x 11) | 13.5 | `tools/tables_from_rfc.py` |
| `dc_qlookup`, `ac_qlookup` (128 each) | 14.1 | `tools/tables_from_rfc.py` |
| the trees, the fixed mode probabilities, `Pcat1`-`Pcat6`, `coeff_bands`, `vp8_mode_contexts`, `mvpartition_probs`, `sub_mv_ref_prob`, the MV update and default probabilities, the filter taps | 8.2, 10, 11.2-11.4, 13.2-13.3, 16.1-16.4, 17.2, 18.3 | by hand, into `src/tables.rs` |

The script finds each table by its declaration in the RFC text and copies
its numbers in order; `cargo test` checks spot values, and the tree tables
are checked to name every leaf exactly once.

## Where the RFC is silent or contradicts itself

| point | what the RFC says | choice | evidence |
|---|---|---|---|
| Zig-zag order | 13: coefficients are coded "in zig-zag order", no table | the zig-zag walk of a 4x4 block, derived in a unit test | confirmed |
| Plane types | 13.3's list numbers Y-after-Y2 0, Y2 1, chroma 2, Y-with-DC 3; its pseudo-code assigns Y2 0 and Y-after-Y2 1 | the list | confirmed |
| Zero tokens | 13.3's pseudo-code sets `prevCoeffWasZero = true` after every token | only after a DCT_0 (the prose: "if the preceding coefficient is a DCT_0") | confirmed |
| Y2 and chroma dequantisation | 14.1: "scaling or clamping ... in dixie.c" (section 20) | Y2 DC x2; Y2 AC x155/100, at least 8; chroma DC at most 132 | confirmed (each adjustment) |
| `segment_feature_mode` | 9.3: 0 absolute, 1 delta; 19.2: 0 delta, 1 absolute | 19.2 | confirmed |
| Segment map not updated | 9.3, 10: ids are coded only when the map is updated | ids persist from earlier frames | confirmed |
| `refresh_entropy_probs` 0 | 19.2: "updated token probabilities are used only for this frame" | every probability the frame updates (token, mode, MV) reverts after it | confirmed (token-only reverting fails) |
| `filter_type` | 9.4, 15: "normal or simple", not which bit | 1 = simple | confirmed |
| Loop filter deltas | 9.4: four reference and four mode deltas, not their order or use | reference: intra, last, golden, altref; mode: B_PRED (intra only), ZEROMV, other whole-MB inter modes, SPLITMV; added after the segment level, clamped to 0..63 | confirmed, except the order of the golden and altref deltas (not exercised) |
| Filter level 0 | 15: skip filtering when the level "at either the frame header level or macroblock override level is 0" | frame level 0: no filtering at all; otherwise a macroblock is skipped when its final level (segment, then deltas) is 0 — a segment level of 0 that the deltas raise is filtered | confirmed (vector 013 has both cases) |
| `vp8_clamp_mv` margins | 16.3: `LEFT_TOP_MARGIN`, `RIGHT_BOTTOM_MARGIN`, `mb_to_*_edge` not defined | vectors may point at most 16 luma samples past the macroblock-aligned frame | confirmed (0, 19 and 32 fail) |
| `best_mv` | 16.3 clamps it | clamped before NEWMV / NEW4x4 add to it | confirmed |
| A SPLITMV macroblock's vector, for later neighbours' census | not stated | its last subblock's (15) | confirmed |
| Macroblocks outside the frame, in the census | 16.3: "a border ... filled with 0,0 motion vectors" | they count as intra (no vector, no weight) | confirmed (counting them as zero vectors fails) |
| Above-right pixels of the last macroblock of a row | 12.3: the pixel at (-1, 15) repeated | as stated | confirmed (127 fails) |
| The pixel above-left of a left-column block below the top row | 12: 129 left of the frame, 127 above it | 129 | confirmed |
| Edge extension for inter prediction | 5: copies of the "visible" edge pixels; 9.1: the excess pixels are kept for prediction | the macroblock-aligned decoded area is extended, to any distance | confirmed (175x143 vectors fail from the visible edge) |
| Key-frame reset | 4: a key frame resets the decoder | probabilities, segmentation, filter deltas and sign biases reset on a key frame | probabilities confirmed; the rest not exercised |
| NEWMV's second clamp | 18.1: the final vector "is clamped again" | the stored vector is clamped (prediction is the same either way, since the clamp only moves a vector further into the extended border) | not exercised |
| Golden / altref copies in one frame | 9.7: "last frame" / "golden" / "altref" are copied, the order is not stated | copies read the buffers as they were before the frame | not exercised |
| Coefficient context | 13.3: neighbours with "at least one non-zero coefficient" | as stated (a block of explicit zeros counts as empty) | not exercised |
| Sign bias | 9.7, 16.3: a neighbour's vector is negated when its reference's sign bias differs from the macroblock's | as stated | not exercised (no vector sets a sign bias); unit-tested against 16.3's census |
| Four and eight token partitions | 9.5 | as stated | not exercised by the vectors (they use one and two); covered by the encoder's round trips |
| Version 3 luma | 9.1: version 3 has no reconstruction filter; 18.1 truncates chroma vectors to whole pixels | chroma whole-pixel (confirmed); luma uses the bilinear filter | luma not exercised |

## Test data

`tests/data` holds the eighteen comprehensive test vectors and their MD5
lists, downloaded unchanged from the WebM project's public test-data bucket;
[its README](../tests/data/README.md) gives the URLs, the date and a hash of
each file.
