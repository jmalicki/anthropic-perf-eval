//! Kernel builder module - generates optimized instruction sequences in Rust.
//!
//! This module ports the Python kernel builders to Rust, eliminating the
//! Python→Rust parsing overhead by creating Op structs directly.
//!
//! ## Module Organization
//!
//! - `barrier_elision` - Remove barrier pseudo-ops before CP-SAT/output
//! - `config` - Kernel configuration (forest height, batch size, etc.)
//! - `context` - Scratch memory allocation context
//! - `context_ssa` - SSA scratch context with VRegs (no pre-allocated addresses)
//! - `emit` - Op creation helpers for clean emission patterns
//! - `emit_ssa` - SSA-based op emission using virtual registers
//! - `partition` - Lazy partition for sorted vload tree access
//! - `tick_tock` - TickTock kernel builder (main implementation)
//! - `tick_tock_ssa` - SSA-based kernel builder (eliminates false dependencies)

pub mod barrier_elision;
pub mod config;
pub mod context;
pub mod context_ssa;
pub mod emit;
pub mod emit_ssa;
pub mod partition;
pub mod ssa_emit;
pub mod ssa_greedy;
pub mod ssa_lower;
pub mod tick_tock;
pub mod tick_tock_ssa;

pub use config::KernelConfig;
pub use context::{ScratchContext, ZERO_REG};
pub use context_ssa::SsaScratchContext;
pub use emit_ssa::SsaEmitContext;
pub use partition::{PartitionBuffer, PartitionHelper};
pub use tick_tock::TickTockKernelBuilder;
pub use tick_tock_ssa::{SsaScheduleResult, SsaTickTockKernelBuilder};
