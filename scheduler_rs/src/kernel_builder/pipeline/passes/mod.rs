//! Concrete pass implementations for the scheduling pipeline.
//!
//! Each pass is a unit struct implementing the `Pass` trait that delegates
//! to an existing function in the codebase. No scheduling logic is duplicated.

pub mod compute_deps;
pub mod dce;
pub mod elide_barriers;
pub mod filter_dead;
pub mod greedy;
pub mod hint_passes;
pub mod assign_addrs;
pub mod lower;

pub mod compute_bounds;
#[cfg(feature = "cpsat")]
pub mod cpsat;
#[cfg(feature = "cpsat")]
pub mod bulk_move;

pub use compute_deps::ComputeDependenciesPass;
pub use compute_bounds::ComputeBoundsPass;
pub use dce::DcePass;
pub use elide_barriers::ElideBarriersPass;
pub use filter_dead::FilterDeadStoresPass;
pub use greedy::GreedySchedulePass;
pub use hint_passes::{
    ConstChoiceHintPass, ScheduleWithFixedChoicesPass, SelectChoiceHintPass, ValuShiftHintPass,
};
pub use assign_addrs::AssignAddressesPass;
pub use lower::LowerToOpsPass;

#[cfg(feature = "cpsat")]
pub use bulk_move::BulkMoveCompactionPass;
#[cfg(feature = "cpsat")]
pub use cpsat::CpSatSchedulePass;
