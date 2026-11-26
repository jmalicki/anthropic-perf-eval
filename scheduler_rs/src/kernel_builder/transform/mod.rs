//! Rewrite rules over the emitted SSA program: detect sites, instantiate alternatives,
//! cost them.
//!
//! Target pipeline (docs/EMISSION_STRUCTURE_MODEL.md): emit the canonical program; rules
//! match sites in it; every alternative is built and costed by its op counts per engine
//! category, with shared setup as resources paid once; a selection model chooses one
//! alternative per site; the apply pass rewrites, dead code elimination removes what is no
//! longer used, and the existing scheduling pipeline runs unchanged.
//!
//! This module holds the first rule, tree access (`tree_access`), and the cost derivation
//! (`cost`). Emitter region tags (`SsaOp::region`) only propose where to look: every site is
//! verified from the ops' definitions (`Rule::match_site`), and facts that cannot be derived
//! from the ops yet are named on the match with their source (`rule::Fact`).
//!
//! - `graph` - def/use view with dead-code-elimination semantics
//! - `rule` - the `Rule` trait, proposals, facts
//! - `tree_access` - tree access at depth d: matcher, builders, instantiation
//! - `cost` - liveness of every op as a function of the choices, and the cost table
//! - `apply` - rewrite the instantiated canonical program into the program of a selection

pub mod apply;
pub mod cost;
pub mod graph;
pub mod rule;
pub mod tree_access;

#[cfg(test)]
mod tests;

pub use apply::apply_tree_access_plan;
pub use cost::{derive_tree_access_costs, instantiate_tree_access, DerivedCostTable, CATEGORIES};
pub use graph::SsaGraph;
pub use rule::{Fact, FactSource, Proposal, Rejection, Rule};
pub use tree_access::{TreeAccessInstantiation, TreeAccessRule, TreeAccessSite};
