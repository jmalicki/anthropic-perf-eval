//! Pure tree-reduce selection semantics for unit testing.
//!
//! These functions mirror the bit extraction and slot selection used in
//! ssa_emit.rs for depth-3 (8-way) and depth-4 (16-way) tree-reduce. They have
//! no dependency on VRegs or the emitter and can be unit tested in isolation.

/// Depth 3: tree indices 7..15, base = 7, 8-way selection.
/// Returns the adjusted index in 0..8 if idx is in 7..15.
#[inline]
pub fn depth3_idx_adj(idx: u32) -> Option<u8> {
    if (7..15).contains(&idx) {
        Some((idx - 7) as u8)
    } else {
        None
    }
}

/// Depth 3: bit extraction from idx_adj (0..8).
/// bit0 = idx_adj & 1, bit1 = (idx_adj >> 1) & 1, bit2 = idx_adj >> 2.
#[inline]
pub fn depth3_bits(idx_adj: u8) -> (u8, u8, u8) {
    let bit0 = idx_adj & 1;
    let bit1 = (idx_adj >> 1) & 1;
    let bit2 = idx_adj >> 2;
    (bit0, bit1, bit2)
}

/// Depth 3: selected slot from bits (must equal idx_adj for correct tree).
#[inline]
pub fn depth3_slot_from_bits(bit0: u8, bit1: u8, bit2: u8) -> u8 {
    bit0 + 2 * bit1 + 4 * bit2
}

/// Depth 4: tree indices 15..31, base = 15, 16-way selection.
/// Returns the adjusted index in 0..16 if idx is in 15..31.
#[inline]
pub fn depth4_idx_adj(idx: u32) -> Option<u8> {
    if (15..31).contains(&idx) {
        Some((idx - 15) as u8)
    } else {
        None
    }
}

/// Depth 4: bit extraction from idx_adj (0..16).
/// bit0..bit3 as in ssa_emit (bit0 = idx_adj & 1, etc.).
#[inline]
pub fn depth4_bits(idx_adj: u8) -> (u8, u8, u8, u8) {
    let bit0 = idx_adj & 1;
    let bit1 = (idx_adj >> 1) & 1;
    let bit2 = (idx_adj >> 2) & 1;
    let bit3 = idx_adj >> 3;
    (bit0, bit1, bit2, bit3)
}

/// Depth 4: selected slot from bits (must equal idx_adj for correct tree).
#[inline]
pub fn depth4_slot_from_bits(bit0: u8, bit1: u8, bit2: u8, bit3: u8) -> u8 {
    bit0 + 2 * bit1 + 4 * bit2 + 8 * bit3
}

/// Simulate depth-3 tree selection: level 0 FMA (bit0 selects 2k vs 2k+1),
/// then two levels of 2-way select by bit1, bit2. Returns values[idx_adj].
#[inline]
pub fn depth3_tree_select_value(idx_adj: u8, values: &[i64; 8]) -> i64 {
    let (bit0, bit1, bit2) = depth3_bits(idx_adj);
    // Level 0: 4 pairs -> p01, p23, p45, p67 (FMA: bit0 ? values[2k+1] : values[2k])
    let p01 = if bit0 != 0 { values[1] } else { values[0] };
    let p23 = if bit0 != 0 { values[3] } else { values[2] };
    let p45 = if bit0 != 0 { values[5] } else { values[4] };
    let p67 = if bit0 != 0 { values[7] } else { values[6] };
    // Level 1: bit1 selects (p23,p01) -> q03, (p67,p45) -> q47
    let q03 = if bit1 != 0 { p23 } else { p01 };
    let q47 = if bit1 != 0 { p67 } else { p45 };
    // Level 2: bit2 selects (q47, q03)
    if bit2 != 0 {
        q47
    } else {
        q03
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth3_idx_adj_range() {
        for idx in 7..15u32 {
            let adj = depth3_idx_adj(idx).unwrap();
            assert_eq!(adj, (idx - 7) as u8, "idx={}", idx);
        }
        assert!(depth3_idx_adj(6).is_none());
        assert!(depth3_idx_adj(15).is_none());
    }

    #[test]
    fn test_depth3_bits_and_slot_roundtrip() {
        for idx in 7..15u32 {
            let idx_adj = depth3_idx_adj(idx).unwrap();
            let (b0, b1, b2) = depth3_bits(idx_adj);
            let slot = depth3_slot_from_bits(b0, b1, b2);
            assert_eq!(slot, idx_adj, "idx={} idx_adj={} bits=({},{},{})", idx, idx_adj, b0, b1, b2);
        }
    }

    #[test]
    fn test_depth3_slot_from_bits_exhaustive() {
        for idx_adj in 0..8u8 {
            let (b0, b1, b2) = depth3_bits(idx_adj);
            assert_eq!(depth3_slot_from_bits(b0, b1, b2), idx_adj);
        }
    }

    #[test]
    fn test_depth4_idx_adj_range() {
        for idx in 15..31u32 {
            let adj = depth4_idx_adj(idx).unwrap();
            assert_eq!(adj, (idx - 15) as u8, "idx={}", idx);
        }
        assert!(depth4_idx_adj(14).is_none());
        assert!(depth4_idx_adj(31).is_none());
    }

    #[test]
    fn test_depth4_bits_and_slot_roundtrip() {
        for idx in 15..31u32 {
            let idx_adj = depth4_idx_adj(idx).unwrap();
            let (b0, b1, b2, b3) = depth4_bits(idx_adj);
            let slot = depth4_slot_from_bits(b0, b1, b2, b3);
            assert_eq!(slot, idx_adj, "idx={} idx_adj={}", idx, idx_adj);
        }
    }

    #[test]
    fn test_depth4_slot_from_bits_exhaustive() {
        for idx_adj in 0..16u8 {
            let (b0, b1, b2, b3) = depth4_bits(idx_adj);
            assert_eq!(depth4_slot_from_bits(b0, b1, b2, b3), idx_adj);
        }
    }

    #[test]
    fn test_depth3_tree_select_value_matches_index() {
        let values: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        for idx_adj in 0..8u8 {
            let selected = depth3_tree_select_value(idx_adj, &values);
            assert_eq!(selected, values[idx_adj as usize], "idx_adj={}", idx_adj);
        }
    }
}
