//! Kernel builder module - generates optimized instruction sequences in Rust.
//!
//! This module ports the Python kernel builders to Rust, eliminating the
//! Python→Rust parsing overhead by creating Op structs directly.
//!
//! ## Module Organization
//!
//! - `config` - Kernel configuration (forest height, batch size, etc.)
//! - `context` - Scratch memory allocation context
//! - `emit` - Op creation helpers for clean emission patterns
//! - `partition` - Lazy partition for sorted vload tree access
//! - `tick_tock` - TickTock kernel builder (main implementation)

pub mod config;
pub mod context;
pub mod emit;
pub mod partition;
pub mod tick_tock;

pub use config::KernelConfig;
pub use context::{ScratchContext, ZERO_REG};
pub use partition::{PartitionBuffer, PartitionHelper};
pub use tick_tock::TickTockKernelBuilder;
