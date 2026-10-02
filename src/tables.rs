//! Constants of the format: trees, fixed probabilities, filter taps.
//!
//! Transcribed by hand from the prose sections of RFC 6386 (the section is
//! named on each); the four large tables are in `tables_rfc.rs`, which a
//! script transcribes. Mode numbering follows the RFC's enumerations, as
//! section 8.2 recommends.

pub(crate) use crate::tables_rfc::{AC_QLOOKUP, COEFF_UPDATE_PROBS, DC_QLOOKUP, DEFAULT_COEFF_PROBS, KF_BMODE_PROBS};

// Macroblock luma modes (section 8.2), then the inter modes (section 16.2),
// in one numbering so a single field holds either.
pub(crate) const DC_PRED: u8 = 0;
pub(crate) const V_PRED: u8 = 1;
pub(crate) const H_PRED: u8 = 2;
pub(crate) const TM_PRED: u8 = 3;
pub(crate) const B_PRED: u8 = 4;
pub(crate) const NEARESTMV: u8 = 5;
pub(crate) const NEARMV: u8 = 6;
pub(crate) const ZEROMV: u8 = 7;
pub(crate) const NEWMV: u8 = 8;
pub(crate) const SPLITMV: u8 = 9;

// Subblock intra modes (section 11.2).
pub(crate) const B_DC_PRED: u8 = 0;
pub(crate) const B_TM_PRED: u8 = 1;
pub(crate) const B_VE_PRED: u8 = 2;
pub(crate) const B_HE_PRED: u8 = 3;
pub(crate) const B_LD_PRED: u8 = 4;
pub(crate) const B_RD_PRED: u8 = 5;
pub(crate) const B_VR_PRED: u8 = 6;
pub(crate) const B_VL_PRED: u8 = 7;
pub(crate) const B_HD_PRED: u8 = 8;
pub(crate) const B_HU_PRED: u8 = 9;

/// Section 8.2 / 16.1: luma modes in inter frames.
pub(crate) static YMODE_TREE: [i8; 8] = [-(DC_PRED as i8), 2, 4, 6, -(V_PRED as i8), -(H_PRED as i8), -(TM_PRED as i8), -(B_PRED as i8)];
/// Section 11.2: luma modes in key frames.
pub(crate) static KF_YMODE_TREE: [i8; 8] = [-(B_PRED as i8), 2, 4, 6, -(DC_PRED as i8), -(V_PRED as i8), -(H_PRED as i8), -(TM_PRED as i8)];
/// Section 11.4: chroma modes.
pub(crate) static UV_MODE_TREE: [i8; 6] = [-(DC_PRED as i8), 2, -(V_PRED as i8), 4, -(H_PRED as i8), -(TM_PRED as i8)];
/// Section 11.2: subblock modes.
pub(crate) static BMODE_TREE: [i8; 18] = [
    -(B_DC_PRED as i8),
    2,
    -(B_TM_PRED as i8),
    4,
    -(B_VE_PRED as i8),
    6,
    8,
    12,
    -(B_HE_PRED as i8),
    10,
    -(B_RD_PRED as i8),
    -(B_VR_PRED as i8),
    -(B_LD_PRED as i8),
    14,
    -(B_VL_PRED as i8),
    16,
    -(B_HD_PRED as i8),
    -(B_HU_PRED as i8),
];
/// Section 10: segment_id.
pub(crate) static MB_SEGMENT_TREE: [i8; 6] = [2, 4, -0, -1, -2, -3];

/// Section 11.2: fixed key-frame luma mode probabilities.
pub(crate) static KF_YMODE_PROBS: [u8; 4] = [145, 156, 163, 128];
/// Section 11.4: fixed key-frame chroma mode probabilities.
pub(crate) static KF_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];
/// Section 16.1: inter-frame luma and chroma mode defaults.
pub(crate) static DEFAULT_YMODE_PROBS: [u8; 4] = [112, 86, 140, 37];
pub(crate) static DEFAULT_UV_MODE_PROBS: [u8; 3] = [162, 101, 204];
/// Section 16.1: fixed subblock mode probabilities in inter frames.
pub(crate) static BMODE_PROBS: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];

// DCT tokens (section 13.2).
pub(crate) const DCT_EOB: u8 = 11;
pub(crate) const DCT_CAT1: u8 = 5;
/// Section 13.2: the token tree.
pub(crate) static COEFF_TREE: [i8; 22] = [
    -(DCT_EOB as i8),
    2,
    -0,
    4,
    -1,
    6,
    8,
    12,
    -2,
    10,
    -3,
    -4,
    14,
    16,
    -5,
    -6,
    18,
    20,
    -7,
    -8,
    -9,
    -10,
];
/// Section 13.2: extra-bit probabilities of dct_cat1..dct_cat6 and the
/// bases of their ranges.
pub(crate) static PCAT: [&[u8]; 6] = [
    &[159],
    &[165, 145],
    &[173, 148, 140],
    &[176, 155, 140, 135],
    &[180, 157, 141, 134, 130],
    &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
];
pub(crate) static CAT_BASE: [i32; 6] = [5, 7, 11, 19, 35, 67];
/// Section 13.3: coefficient position to band.
pub(crate) static COEFF_BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
/// Section 13: coefficients are coded in zig-zag order. The RFC names the
/// order without tabulating it; this is the zig-zag walk of a 4x4 block
/// (along anti-diagonals, alternating direction, starting rightwards), as
/// raster indices. `zigzag_is_the_zigzag_walk` derives it.
pub(crate) static ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Section 16.2: the macroblock inter mode tree.
pub(crate) static MV_REF_TREE: [i8; 8] = [-(ZEROMV as i8), 2, -(NEARESTMV as i8), 4, -(NEARMV as i8), 6, -(NEWMV as i8), -(SPLITMV as i8)];
/// Section 16.3: mode probabilities by neighbour census.
pub(crate) static MODE_CONTEXTS: [[u8; 4]; 6] =
    [[7, 1, 1, 143], [14, 18, 14, 107], [135, 64, 57, 68], [60, 56, 128, 65], [159, 134, 128, 34], [234, 188, 128, 28]];

// SPLITMV partitionings (section 16.4).
pub(crate) const MV_TOP_BOTTOM: u8 = 0;
pub(crate) const MV_LEFT_RIGHT: u8 = 1;
pub(crate) const MV_QUARTERS: u8 = 2;
pub(crate) const MV_16: u8 = 3;
pub(crate) static MV_PARTITION_TREE: [i8; 6] = [-(MV_16 as i8), 2, -(MV_QUARTERS as i8), 4, -(MV_TOP_BOTTOM as i8), -(MV_LEFT_RIGHT as i8)];
pub(crate) static MV_PARTITION_PROBS: [u8; 3] = [110, 111, 150];
/// Which part each subblock belongs to, by partitioning.
pub(crate) static MV_PARTITIONS: [[u8; 16]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];
pub(crate) static MV_PARTITION_COUNT: [usize; 4] = [2, 2, 4, 16];

// Subblock inter modes (section 16.4), numbered from 0 here.
pub(crate) const LEFT4X4: u8 = 0;
pub(crate) const ABOVE4X4: u8 = 1;
pub(crate) const ZERO4X4: u8 = 2;
pub(crate) const NEW4X4: u8 = 3;
pub(crate) static SUB_MV_REF_TREE: [i8; 6] = [-(LEFT4X4 as i8), 2, -(ABOVE4X4 as i8), 4, -(ZERO4X4 as i8), -(NEW4X4 as i8)];
pub(crate) static SUB_MV_REF_PROBS: [[u8; 3]; 5] = [[147, 136, 18], [106, 145, 1], [179, 121, 1], [223, 1, 34], [208, 1, 1]];

// Motion vector component probabilities (section 17).
pub(crate) const MVP_IS_SHORT: usize = 0;
pub(crate) const MVP_SIGN: usize = 1;
pub(crate) const MVP_SHORT: usize = 2;
pub(crate) const MVP_BITS: usize = 9;
pub(crate) const MVP_COUNT: usize = 19;
pub(crate) static SMALL_MV_TREE: [i8; 14] = [2, 8, 4, 6, -0, -1, -2, -3, 10, 12, -4, -5, -6, -7];
pub(crate) static MV_UPDATE_PROBS: [[u8; MVP_COUNT]; 2] = [
    [237, 246, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 250, 250, 252, 254, 254],
    [231, 243, 245, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 251, 251, 254, 254, 254],
];
pub(crate) static DEFAULT_MV_PROBS: [[u8; MVP_COUNT]; 2] = [
    [162, 128, 225, 146, 172, 147, 214, 39, 156, 128, 129, 132, 75, 145, 178, 206, 239, 254, 254],
    [164, 128, 204, 170, 119, 235, 140, 230, 228, 128, 130, 130, 74, 148, 180, 203, 236, 254, 254],
];

/// Section 18.3: the six-tap ("bicubic") subpixel filters by eighth-pel
/// offset.
pub(crate) static SIXTAP_FILTERS: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, -6, 123, 12, -1, 0],
    [2, -11, 108, 36, -8, 1],
    [0, -9, 93, 50, -6, 0],
    [3, -16, 77, 77, -16, 3],
    [0, -6, 50, 93, -9, 0],
    [1, -8, 36, 108, -11, 2],
    [0, -1, 12, 123, -6, 0],
];
/// Section 18.3: the bilinear filters.
pub(crate) static BILINEAR_FILTERS: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, 0, 112, 16, 0, 0],
    [0, 0, 96, 32, 0, 0],
    [0, 0, 80, 48, 0, 0],
    [0, 0, 64, 64, 0, 0],
    [0, 0, 48, 80, 0, 0],
    [0, 0, 32, 96, 0, 0],
    [0, 0, 16, 112, 0, 0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(tree: &[i8]) -> Vec<u8> {
        let mut v: Vec<u8> = tree.iter().filter(|&&t| t <= 0).map(|&t| (-t) as u8).collect();
        v.sort();
        v
    }

    #[test]
    fn trees_have_every_leaf_once() {
        assert_eq!(leaves(&YMODE_TREE), vec![0, 1, 2, 3, 4]);
        assert_eq!(leaves(&KF_YMODE_TREE), vec![0, 1, 2, 3, 4]);
        assert_eq!(leaves(&UV_MODE_TREE), vec![0, 1, 2, 3]);
        assert_eq!(leaves(&BMODE_TREE), (0..10).collect::<Vec<_>>());
        assert_eq!(leaves(&COEFF_TREE), (0..12).collect::<Vec<_>>());
        assert_eq!(leaves(&MV_REF_TREE), vec![5, 6, 7, 8, 9]);
        assert_eq!(leaves(&SMALL_MV_TREE), (0..8).collect::<Vec<_>>());
        assert_eq!(leaves(&MB_SEGMENT_TREE), vec![0, 1, 2, 3]);
        assert_eq!(leaves(&MV_PARTITION_TREE), vec![0, 1, 2, 3]);
        assert_eq!(leaves(&SUB_MV_REF_TREE), vec![0, 1, 2, 3]);
    }

    #[test]
    fn zigzag_is_the_zigzag_walk() {
        // Walk the anti-diagonals of a 4x4 block, alternating direction.
        let mut order = Vec::new();
        for d in 0..7i32 {
            let cells: Vec<(i32, i32)> = (0..4).filter_map(|r| {
                let c = d - r;
                (0..4).contains(&c).then_some((r, c))
            }).collect();
            // Even diagonals run bottom-left to top-right (rows decreasing).
            if d % 2 == 0 {
                order.extend(cells.iter().rev().map(|&(r, c)| (r * 4 + c) as usize));
            } else {
                order.extend(cells.iter().map(|&(r, c)| (r * 4 + c) as usize));
            }
        }
        assert_eq!(order, ZIGZAG);
    }

    #[test]
    fn filters_sum_to_128() {
        for f in SIXTAP_FILTERS.iter().chain(BILINEAR_FILTERS.iter()) {
            assert_eq!(f.iter().sum::<i32>(), 128);
        }
    }

    #[test]
    fn transcribed_tables_have_rfc_spot_values() {
        assert_eq!(KF_BMODE_PROBS[0][0], [231, 120, 48, 89, 115, 113, 120, 152, 112]);
        assert_eq!(KF_BMODE_PROBS[9][9], [112, 19, 12, 61, 195, 128, 48, 4, 24]);
        assert_eq!(DEFAULT_COEFF_PROBS[0][1][0], [253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128]);
        assert_eq!(DEFAULT_COEFF_PROBS[3][7][2], [238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128]);
        assert_eq!(COEFF_UPDATE_PROBS[0][1][0], [176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!((DC_QLOOKUP[0], DC_QLOOKUP[127]), (4, 157));
        assert_eq!((AC_QLOOKUP[0], AC_QLOOKUP[127]), (4, 284));
        assert!(DC_QLOOKUP.windows(2).all(|w| w[0] <= w[1]));
        assert!(AC_QLOOKUP.windows(2).all(|w| w[0] < w[1]));
    }
}
