//! SSA program intermediate representation.
//!
//! `SsaProgram` bundles all artifacts produced by emission (the "front end")
//! and consumed by compilation (scheduling, optimization, allocation, lowering).
//! It replaces the duplicated inline extraction pattern that previously
//! appeared in both `build_and_schedule_with_options` and `build_and_schedule_cpsat`.

use std::collections::HashMap;

use crate::types::{SsaOp, VReg};

use super::emit_ssa::ConstAlternative;

/// The output of emission / input to compilation.
///
/// Bundles all artifacts needed to schedule and lower an SSA program.
/// Created by `SsaTickTockKernelBuilder::build()`.
pub struct SsaProgram {
    /// SSA operations in emission order (unscheduled).
    pub ops: Vec<SsaOp>,
    /// Map from VReg to the op index that defines it.
    pub vreg_def: HashMap<VReg, usize>,
    /// Total number of VRegs allocated during emission.
    pub vreg_count: u32,
    /// Constant alternatives (Load vs AddImm vs AluDerive encodings).
    pub const_alternatives: Vec<ConstAlternative>,
    /// VReg representing constant 0 (address 0), if reserved.
    pub zero_vreg: Option<VReg>,
    /// Maps vector group alias VRegs to their 8 scalar lane VRegs (for scatter loads).
    pub vector_group_lanes: HashMap<VReg, Vec<VReg>>,
}
