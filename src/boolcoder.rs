//! The boolean entropy coder (RFC 6386 section 7) and tree coding
//! (section 8).
//!
//! Both sides keep the same 8-bit `range` (128..=255 between bools) and
//! compute the same `split = 1 + (((range - 1) * prob) >> 8)`. The decoder
//! keeps `value` — the coded number less the interval's left end — in the
//! top bits of a 64-bit window, so it refills a byte at a time and compares
//! against `split` shifted to the top. The encoder keeps the interval's
//! left end (`bottom`) and propagates carries into bytes already written.

/// A tree in RFC 6386's array form (section 8.1): entries come in pairs (the
/// 0 and 1 branches of a node); a positive entry is the index of a deeper
/// pair, zero or a negative entry `-v` is the leaf `v`. The node at index
/// `2k` takes probability `k`.
pub(crate) type Tree = [i8];

/// Reads bools from one partition.
pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// The comparison window is the top 8 bits; below it, `count` more
    /// valid bits of input.
    value: u64,
    count: i32,
    range: u32,
    /// Bytes of zeros supplied past the end of `data`.
    overrun: usize,
}

impl<'a> BoolDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut d = BoolDecoder { data, pos: 0, value: 0, count: -8, range: 255, overrun: 0 };
        d.fill();
        d
    }

    /// Tops the window up to at least 48 bits past the comparison byte.
    /// Past the end of the data the coded number continues with zeros —
    /// what a partition's final bytes imply, since the encoder writes the
    /// interval's left end — and `overrun` counts them.
    #[inline]
    fn fill(&mut self) {
        while self.count <= 40 {
            let byte = if self.pos < self.data.len() {
                let b = self.data[self.pos];
                self.pos += 1;
                b
            } else {
                self.overrun += 1;
                0
            };
            self.value |= (byte as u64) << (48 - self.count);
            self.count += 8;
        }
    }

    /// One bool whose probability of being 0 is `prob / 256`.
    #[inline]
    pub(crate) fn read(&mut self, prob: u8) -> bool {
        if self.count < 8 {
            self.fill();
        }
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let big = (split as u64) << 56;
        let bit = if self.value >= big {
            self.range -= split;
            self.value -= big;
            true
        } else {
            self.range = split;
            false
        };
        // Renormalise: double range (and value) until range >= 128.
        let shift = self.range.leading_zeros() - 24;
        self.range <<= shift;
        self.value <<= shift;
        self.count -= shift as i32;
        bit
    }

    /// A one-bit flag, `F` / `L(1)`.
    #[inline]
    pub(crate) fn flag(&mut self) -> bool {
        self.read(128)
    }

    /// An unsigned `n`-bit literal, high bit first, `L(n)`.
    pub(crate) fn literal(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.flag() as u32;
        }
        v
    }

    /// A magnitude of `n` bits followed by a sign bit (1 = negative), the
    /// form of every signed header field (sections 9.3, 9.4, 9.6).
    pub(crate) fn signed(&mut self, n: u32) -> i32 {
        let m = self.literal(n) as i32;
        if self.flag() { -m } else { m }
    }

    /// `F? L(n) with sign : 0`.
    pub(crate) fn optional_signed(&mut self, n: u32) -> i32 {
        if self.flag() { self.signed(n) } else { 0 }
    }

    /// A tree-coded value (section 8.1), descending from node `start`.
    #[inline]
    pub(crate) fn tree(&mut self, tree: &Tree, probs: &[u8], start: usize) -> u8 {
        let mut i = start;
        loop {
            let next = tree[i + self.read(probs[i >> 1]) as usize];
            if next <= 0 {
                return (-next) as u8;
            }
            i = next as usize;
        }
    }

    /// Whether the decoder has needed more than two bytes past the end of
    /// its data. Section 7.3's decoder holds two bytes ahead of the bool
    /// being decoded, so a partition's last bool can legitimately ask for
    /// them; anything beyond means the partition was cut short.
    pub(crate) fn overran(&self) -> bool {
        // Bits shifted out of the comparison byte so far; section 7.3's
        // decoder has then loaded 2 + shifts / 8 bytes.
        let shifts = (self.pos + self.overrun) as i64 * 8 - (8 + self.count as i64);
        2 + shifts / 8 > self.data.len() as i64 + 2
    }
}

/// Writes bools into a partition (RFC 6386 sections 7.2-7.3).
pub(crate) struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    /// Left end of the interval: 24 bits of pending output plus the bits not
    /// yet shifted out.
    bottom: u32,
    /// Shifts left before the top byte of `bottom` is complete.
    bit_count: i32,
}

impl BoolEncoder {
    pub(crate) fn new() -> Self {
        BoolEncoder { out: Vec::new(), range: 255, bottom: 0, bit_count: 24 }
    }

    /// Adds one to the bytes already written (a carry out of `bottom`).
    fn carry(&mut self) {
        for b in self.out.iter_mut().rev() {
            if *b == 255 {
                *b = 0;
            } else {
                *b += 1;
                return;
            }
        }
        // The coded number is always below one, so a carry never runs off
        // the front of the partition.
        debug_assert!(false, "bool encoder carry past the start of the partition");
    }

    /// Codes `bit`, whose probability of being 0 is `prob / 256`.
    #[inline]
    pub(crate) fn write(&mut self, prob: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        if bit {
            self.bottom = self.bottom.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            if self.bottom & (1 << 31) != 0 {
                self.carry();
            }
            self.bottom <<= 1;
            self.bit_count -= 1;
            if self.bit_count == 0 {
                self.out.push((self.bottom >> 24) as u8);
                self.bottom &= (1 << 24) - 1;
                self.bit_count = 8;
            }
        }
    }

    pub(crate) fn flag(&mut self, bit: bool) {
        self.write(128, bit);
    }

    pub(crate) fn literal(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.flag((v >> i) & 1 != 0);
        }
    }

    /// Magnitude then sign, the inverse of [`BoolDecoder::signed`].
    pub(crate) fn signed(&mut self, n: u32, v: i32) {
        self.literal(n, v.unsigned_abs());
        self.flag(v < 0);
    }

    /// Codes leaf `value` of `tree`, starting at node `start`.
    pub(crate) fn tree(&mut self, tree: &Tree, probs: &[u8], start: usize, value: u8) {
        // Find the path to the leaf, then write it from the root down.
        let mut path = [(0usize, false); 16];
        let n = tree_path(tree, start, value, &mut path).expect("value is a leaf of the tree");
        for &(node, bit) in &path[..n] {
            self.write(probs[node >> 1], bit);
        }
    }

    /// Writes out what remains of `bottom`, padding the partition so the
    /// decoder's two-byte window never reads past it, and returns the bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        // Flushing 32 zero bits at probability 1/2 pushes every pending bit
        // of `bottom` (and any carry) out through the normal path.
        for _ in 0..32 {
            self.flag(false);
        }
        self.out
    }

    /// Bytes written so far (a lower bound on the final size).
    pub(crate) fn len(&self) -> usize {
        self.out.len()
    }
}

/// The (node, bit) steps from node `start` to leaf `value`.
fn tree_path(tree: &Tree, start: usize, value: u8, path: &mut [(usize, bool); 16]) -> Option<usize> {
    fn walk(tree: &Tree, node: usize, value: u8, depth: usize, path: &mut [(usize, bool); 16]) -> Option<usize> {
        for bit in 0..2 {
            path[depth] = (node, bit == 1);
            let next = tree[node + bit];
            if next <= 0 {
                if (-next) as u8 == value {
                    return Some(depth + 1);
                }
            } else if let Some(n) = walk(tree, next as usize, value, depth + 1, path) {
                return Some(n);
            }
        }
        None
    }
    walk(tree, start, value, 0, path)
}

/// Cost in 1/256 bits of coding `bit` at `prob`, for the encoder's choices.
pub(crate) fn cost(prob: u8, bit: bool) -> u32 {
    let p = if bit { 256 - prob as u32 } else { prob as u32 };
    COST_TABLE[p as usize] as u32
}

/// `-log2(p / 256) * 256` for p in 0..=256 (entry 0 unused).
static COST_TABLE: std::sync::LazyLock<[u16; 257]> = std::sync::LazyLock::new(|| {
    let mut t = [0u16; 257];
    for (p, e) in t.iter_mut().enumerate().skip(1) {
        *e = (-(p as f64 / 256.0).log2() * 256.0).round() as u16;
    }
    t[0] = t[1];
    t
});

/// Cost of coding `value` in `tree` from `start`, in 1/256 bits.
pub(crate) fn tree_cost(tree: &Tree, probs: &[u8], start: usize, value: u8) -> u32 {
    let mut path = [(0usize, false); 16];
    let n = tree_path(tree, start, value, &mut path).expect("value is a leaf of the tree");
    path[..n].iter().map(|&(node, bit)| cost(probs[node >> 1], bit)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight transcription of the bit-at-a-time decoder of section 7.2
    /// — value and range 8 bits wide, one input bit shifted in per doubling —
    /// to check the windowed decoder above against.
    struct BitDecoder<'a> {
        data: &'a [u8],
        bitpos: usize,
        value: u32,
        range: u32,
    }

    impl<'a> BitDecoder<'a> {
        fn bit(&mut self) -> u32 {
            let byte = self.data.get(self.bitpos / 8).copied().unwrap_or(0);
            let b = (byte >> (7 - self.bitpos % 8)) & 1;
            self.bitpos += 1;
            b as u32
        }
        fn new(data: &'a [u8]) -> Self {
            let mut d = BitDecoder { data, bitpos: 0, value: 0, range: 255 };
            for _ in 0..16 {
                d.value = (d.value << 1) | d.bit();
            }
            d
        }
        fn read(&mut self, prob: u8) -> bool {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            let big = split << 8;
            let r = if self.value >= big {
                self.range -= split;
                self.value -= big;
                true
            } else {
                self.range = split;
                false
            };
            while self.range < 128 {
                self.range <<= 1;
                self.value = (self.value << 1) | self.bit();
            }
            r
        }
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn round_trip_random_bools() {
        let mut seed = 7;
        for trial in 0..50 {
            let n = 1 + (lcg(&mut seed) % 5000) as usize;
            let items: Vec<(u8, bool)> = (0..n)
                .map(|_| {
                    let p = (lcg(&mut seed) % 255 + 1) as u8;
                    // Bias the bits towards the probability, as real data is.
                    let bit = (lcg(&mut seed) % 256) >= p as u32;
                    (p, if trial % 3 == 0 { !bit } else { bit })
                })
                .collect();
            let mut e = BoolEncoder::new();
            for &(p, b) in &items {
                e.write(p, b);
            }
            let data = e.finish();
            let mut d = BoolDecoder::new(&data);
            let mut r = BitDecoder::new(&data);
            for (i, &(p, b)) in items.iter().enumerate() {
                assert_eq!(d.read(p), b, "trial {trial} bool {i}");
                assert_eq!(r.read(p), b, "reference decoder, trial {trial} bool {i}");
            }
            assert!(!d.overran());
        }
    }

    #[test]
    fn literals_and_trees() {
        let tree: [i8; 8] = [-0, 2, 4, 6, -1, -2, -3, -4];
        let probs = [100, 50, 200, 30];
        let mut e = BoolEncoder::new();
        e.literal(7, 93);
        e.signed(4, -11);
        e.signed(6, 0);
        for v in 0..5 {
            e.tree(&tree, &probs, 0, v);
        }
        let data = e.finish();
        let mut d = BoolDecoder::new(&data);
        assert_eq!(d.literal(7), 93);
        assert_eq!(d.signed(4), -11);
        assert_eq!(d.signed(6), 0);
        for v in 0..5 {
            assert_eq!(d.tree(&tree, &probs, 0), v);
        }
    }

    #[test]
    fn carries_propagate() {
        // Long runs of probable-one bools at high probability force carries
        // through bytes of 0xff.
        let mut e = BoolEncoder::new();
        let mut bits = Vec::new();
        let mut seed = 99;
        for i in 0..20000 {
            let b = i % 97 != 0 && lcg(&mut seed) % 50 != 0;
            bits.push(b);
            e.write(1, b);
        }
        let data = e.finish();
        let mut d = BoolDecoder::new(&data);
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(d.read(1), b, "bool {i}");
        }
    }

    #[test]
    fn costs_are_monotone() {
        assert_eq!(cost(128, false), 256);
        assert!(cost(250, false) < cost(250, true));
        assert_eq!(tree_cost(&[-0, -1], &[128], 0, 1), 256);
    }
}
