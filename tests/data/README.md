# Test vectors

The VP8 comprehensive test vectors, `vp80-00-comprehensive-001` to `-018`:
eighteen short IVF streams built to exercise the format's features (every
intra and inter mode, split motion vectors, golden and altref references
with sign bias, segmentation, loop filter deltas and both filter types,
bitstream versions 0 to 3, multiple token partitions, probability
persistence, odd frame sizes, a hidden frame), each with an `.ivf.md5` file
listing the MD5 of every frame it shows, in order. Each MD5 is of the frame
as packed I420 — the Y plane, then U, then V, each `(width + 1) / 2` by
`(height + 1) / 2` — which is what `Frame::packed` returns.

They are bitstream data and expected outputs, not code: nothing from any
decoder was used to make or check this crate beyond the MD5s listed in these
files. `tests/vectors.rs` decodes every stream and compares every frame.

## Source

Downloaded on 2026-10-02 from the WebM project's public test-data bucket:

```sh
for i in $(seq -w 1 18); do
  for ext in ivf ivf.md5; do
    curl -O https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/vp80-00-comprehensive-0$i.$ext
  done
done
```

The files are committed unchanged (830 KB in all).

| stream | size | frames | frames shown | SHA-256 (first 16 hex digits) |
|---|---|---|---|---|
| `vp80-00-comprehensive-001` | 176x144 | 29 | 29 | `e01278ada71e61d4` |
| `vp80-00-comprehensive-002` | 176x144 | 49 | 49 | `348b9d4dc50d8889` |
| `vp80-00-comprehensive-003` | 176x144 | 49 | 49 | `d5123023dfa81c94` |
| `vp80-00-comprehensive-004` | 176x144 | 29 | 29 | `0550a581cf2c85e0` |
| `vp80-00-comprehensive-005` | 176x144 | 49 | 49 | `698f865dfdd150cb` |
| `vp80-00-comprehensive-006` | 175x143 | 48 | 48 | `46a74dbaa7ad895c` |
| `vp80-00-comprehensive-007` | 176x144 | 29 | 29 | `7c8191908d724f44` |
| `vp80-00-comprehensive-008` | 1432x888 | 2 | 2 | `553fb67d473c8b21` |
| `vp80-00-comprehensive-009` | 176x144 | 49 | 49 | `52d7e373bbecefdf` |
| `vp80-00-comprehensive-010` | 320x240 | 57 | 57 | `df220c09c1553c39` |
| `vp80-00-comprehensive-011` | 176x144 | 29 | 29 | `8826d37f4fb2332f` |
| `vp80-00-comprehensive-012` | 176x144 | 29 | 29 | `504d6d942bb665a4` |
| `vp80-00-comprehensive-013` | 176x144 | 29 | 29 | `d721d4b2c91f48a2` |
| `vp80-00-comprehensive-014` | 175x143 | 49 | 49 | `f2848f9f098c12f1` |
| `vp80-00-comprehensive-015` | 320x240 | 260 | 260 | `bed192ee30432b30` |
| `vp80-00-comprehensive-016` | 176x144 | 29 | 29 | `f1864da217b83dda` |
| `vp80-00-comprehensive-017` | 176x144 | 29 | 29 | `1377f7be0851664f` |
| `vp80-00-comprehensive-018` | 176x144 | 29 | 28 | `58fc48a7c9611335` |

Stream 018's first frame is not shown (`show_frame` 0), so its MD5 list
starts at frame 2.
