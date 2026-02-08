//! Concrete pass implementations for the scheduling pipeline.
//!
//! Each pass is a unit struct implementing the `Pass` trait that delegates
//! to an existing function in the codebase. No scheduling logic is duplicated.

pub mod compute_deps;
pub mod dce;
pub mod elide_barriers;
pub mod filter_dead;
pub mod greedy;
pub mod assign_addrs;
pub mod lower;

#[cfg(feature = "cpsat")]
pub mod compute_bounds;
#[cfg(feature = "cpsat")]
pub mod cpsat;

pub use compute_deps::ComputeDependenciesPass;
pub use dce::DcePass;
pub use elide_barriers::ElideBarriersPass;
pub use filter_dead::FilterDeadStoresPass;
pub use greedy::GreedySchedulePass;
pub use assign_addrs::AssignAddressesPass;
pub use lower::LowerToOpsPass;

#[cfg(feature = "cpsat")]
pub use compute_bounds::ComputeBoundsPass;
#[cfg(feature = "cpsat")]
pub use cpsat::CpSatSchedulePass;
