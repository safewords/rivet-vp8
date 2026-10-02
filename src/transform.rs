//! The inverse transforms (RFC 6386 sections 14.3-14.5) and the encoder's
//! forward transforms.
//!
//! Section 14 specifies the inverse WHT and DCT exactly, with 16-bit signed
//! storage for the input, the intermediate and the output; the `as i16`
//! casts below are those stores (they only matter for streams whose
//! dequantised coefficients overflow, which a conforming decoder must still
//! reproduce). The forward transforms only have to approximate the
//! inverses: they decide what is coded, not how it decodes.

/// sqrt(2) * cos(pi / 8) - 1, and sqrt(2) * sin(pi / 8), in 16-bit fixed
/// point (section 14.4).
const COS_PI8_SQRT2_MINUS1: i32 = 20091;
const SIN_PI8_SQRT2: i32 = 35468;

/// The inverse Walsh-Hadamard transform of the Y2 block (section 14.3):
/// `input` is the dequantised block in raster order, the result is the DC
/// of each Y subblock in raster order.
pub(crate) fn inverse_wht(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i16; 16];
    for i in 0..4 {
        let (i0, i4, i8, i12) = (
            input[i] as i32,
            input[4 + i] as i32,
            input[8 + i] as i32,
            input[12 + i] as i32,
        );
        let a1 = i0 + i12;
        let b1 = i4 + i8;
        let c1 = i4 - i8;
        let d1 = i0 - i12;
        tmp[i] = (a1 + b1) as i16;
        tmp[4 + i] = (c1 + d1) as i16;
        tmp[8 + i] = (a1 - b1) as i16;
        tmp[12 + i] = (d1 - c1) as i16;
    }
    let mut out = [0i16; 16];
    for r in 0..4 {
        let row = &tmp[r * 4..r * 4 + 4];
        let a1 = row[0] as i32 + row[3] as i32;
        let b1 = row[1] as i32 + row[2] as i32;
        let c1 = row[1] as i32 - row[2] as i32;
        let d1 = row[0] as i32 - row[3] as i32;
        out[r * 4] = ((a1 + b1 + 3) >> 3) as i16;
        out[r * 4 + 1] = ((c1 + d1 + 3) >> 3) as i16;
        out[r * 4 + 2] = ((a1 - b1 + 3) >> 3) as i16;
        out[r * 4 + 3] = ((d1 - c1 + 3) >> 3) as i16;
    }
    out
}

#[inline]
fn mul_sin(x: i32) -> i32 {
    (x * SIN_PI8_SQRT2) >> 16
}

#[inline]
fn mul_cos(x: i32) -> i32 {
    x + ((x * COS_PI8_SQRT2_MINUS1) >> 16)
}

/// The inverse DCT of one 4x4 block (section 14.4): dequantised
/// coefficients in raster order in, residue in raster order out.
pub(crate) fn inverse_dct(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i16; 16];
    // Vertical pass: one column at a time.
    for i in 0..4 {
        let (i0, i4, i8, i12) = (
            input[i] as i32,
            input[4 + i] as i32,
            input[8 + i] as i32,
            input[12 + i] as i32,
        );
        let a1 = i0 + i8;
        let b1 = i0 - i8;
        let c1 = mul_sin(i4) - mul_cos(i12);
        let d1 = mul_cos(i4) + mul_sin(i12);
        tmp[i] = (a1 + d1) as i16;
        tmp[12 + i] = (a1 - d1) as i16;
        tmp[4 + i] = (b1 + c1) as i16;
        tmp[8 + i] = (b1 - c1) as i16;
    }
    let mut out = [0i16; 16];
    // Horizontal pass with the final rounding.
    for r in 0..4 {
        let row = &tmp[r * 4..r * 4 + 4];
        let (i0, i1, i2, i3) = (row[0] as i32, row[1] as i32, row[2] as i32, row[3] as i32);
        let a1 = i0 + i2;
        let b1 = i0 - i2;
        let c1 = mul_sin(i1) - mul_cos(i3);
        let d1 = mul_cos(i1) + mul_sin(i3);
        out[r * 4] = ((a1 + d1 + 4) >> 3) as i16;
        out[r * 4 + 3] = ((a1 - d1 + 4) >> 3) as i16;
        out[r * 4 + 1] = ((b1 + c1 + 4) >> 3) as i16;
        out[r * 4 + 2] = ((b1 - c1 + 4) >> 3) as i16;
    }
    out
}

/// Adds the inverse DCT of `coeffs` to the 4x4 block at `dst[off..]`
/// (section 14.5: the sum is saturated to 8 bits).
#[inline]
pub(crate) fn add_residue(coeffs: &[i16; 16], dst: &mut [u8], off: usize, stride: usize) {
    // A block with only a DC coefficient inverts to a constant (the general
    // transform gives (dc + 4) >> 3 everywhere), the common case.
    if coeffs[1..].iter().all(|&c| c == 0) {
        let dc = (coeffs[0] as i32 + 4) >> 3;
        if dc == 0 {
            return;
        }
        for r in 0..4 {
            for p in &mut dst[off + r * stride..off + r * stride + 4] {
                *p = (*p as i32 + dc).clamp(0, 255) as u8;
            }
        }
        return;
    }
    let res = inverse_dct(coeffs);
    for r in 0..4 {
        for c in 0..4 {
            let p = &mut dst[off + r * stride + c];
            *p = (*p as i32 + res[r * 4 + c] as i32).clamp(0, 255) as u8;
        }
    }
}

/// Forward DCT for the encoder: an integer approximation of the transform
/// whose inverse is [`inverse_dct`] (it returns twice the orthonormal
/// DCT-II coefficients, the scale section 14.4 says the inverse expects).
pub(crate) fn forward_dct(input: &[i16; 16]) -> [i16; 16] {
    let mut out = [0i16; 16];
    let basis = dct_basis();
    for u in 0..4 {
        for v in 0..4 {
            let mut s = 0.0f64;
            for y in 0..4 {
                for x in 0..4 {
                    s += input[y * 4 + x] as f64 * basis[u][y] * basis[v][x];
                }
            }
            out[u * 4 + v] = (s * 2.0).round() as i16;
        }
    }
    out
}

/// Forward WHT for the encoder: the inverse of [`inverse_wht`] up to
/// rounding (input: the 16 Y DCs in raster order).
pub(crate) fn forward_wht(input: &[i16; 16]) -> [i16; 16] {
    // The inverse is H x H / 8 with H the 4-point Hadamard matrix in the
    // order (a+b, c+d, a-b, d-c); its inverse is H^T x H^T / 2.
    let h: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        for c in 0..4 {
            tmp[r * 4 + c] = (0..4).map(|k| h[k][c] * input[r * 4 + k] as i32).sum();
        }
    }
    let mut out = [0i16; 16];
    for r in 0..4 {
        for c in 0..4 {
            let s: i32 = (0..4).map(|k| h[k][r] * tmp[k * 4 + c]).sum();
            out[r * 4 + c] = ((s + if s >= 0 { 1 } else { -1 }) / 2) as i16;
        }
    }
    out
}

/// Rows of the orthonormal 4-point DCT-II.
fn dct_basis() -> [[f64; 4]; 4] {
    let mut b = [[0.0; 4]; 4];
    for (k, row) in b.iter_mut().enumerate() {
        let s = if k == 0 { 0.5 } else { (0.5f64).sqrt() };
        for (n, e) in row.iter_mut().enumerate() {
            *e = s * (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 8.0).cos();
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse DCT against a floating-point orthonormal inverse at the
    /// same scale: the integer transform approximates it to within a unit.
    #[test]
    fn idct_matches_float_reference() {
        let basis = dct_basis();
        let mut seed = 12345u64;
        for _ in 0..2000 {
            let mut coeffs = [0i16; 16];
            for c in coeffs.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *c = ((seed >> 40) % 801) as i16 - 400;
            }
            let got = inverse_dct(&coeffs);
            for y in 0..4 {
                for x in 0..4 {
                    let mut s = 0.0;
                    for u in 0..4 {
                        for v in 0..4 {
                            s += coeffs[u * 4 + v] as f64 * basis[u][y] * basis[v][x];
                        }
                    }
                    let want = s / 2.0;
                    assert!(
                        (got[y * 4 + x] as f64 - want).abs() <= 1.5,
                        "{coeffs:?} at {y},{x}: {} vs {want}",
                        got[y * 4 + x]
                    );
                }
            }
        }
    }

    #[test]
    fn idct_dc_only_is_constant() {
        for dc in [-2048i16, -9, -4, -3, 0, 3, 4, 5, 100, 2047] {
            let mut c = [0i16; 16];
            c[0] = dc;
            let out = inverse_dct(&c);
            assert!(
                out.iter().all(|&v| v as i32 == (dc as i32 + 4) >> 3),
                "{dc}: {out:?}"
            );
        }
    }

    #[test]
    fn wht_round_trips() {
        let mut seed = 5u64;
        for _ in 0..500 {
            let mut dcs = [0i16; 16];
            for c in dcs.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *c = ((seed >> 40) % 2001) as i16 - 1000;
            }
            let back = inverse_wht(&forward_wht(&dcs));
            for i in 0..16 {
                assert!((back[i] - dcs[i]).abs() <= 1, "{dcs:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn wht_dc_only_matches_shortcut() {
        // Section 14.3's single-DC shortcut: every output is (dc + 3) >> 3.
        for dc in [-1000i16, -5, 0, 5, 1000] {
            let mut c = [0i16; 16];
            c[0] = dc;
            assert!(
                inverse_wht(&c)
                    .iter()
                    .all(|&v| v as i32 == (dc as i32 + 3) >> 3)
            );
        }
    }

    #[test]
    fn dct_round_trips() {
        let mut seed = 77u64;
        for _ in 0..500 {
            let mut px = [0i16; 16];
            for c in px.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *c = ((seed >> 40) % 511) as i16 - 255;
            }
            let back = inverse_dct(&forward_dct(&px));
            for i in 0..16 {
                assert!((back[i] - px[i]).abs() <= 1, "{px:?} -> {back:?}");
            }
        }
    }
}
