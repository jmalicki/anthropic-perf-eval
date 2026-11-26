//! The rule interface: propose candidate sites, match them structurally, name alternatives.
//!
//! A rule has three parts:
//! * `propose`: where to look. Today proposals come from emitter region tags
//!   (`SsaOp::region`); a future analysis may propose sites in untagged SSA (e.g. a lifted
//!   kernel). A proposal carries no meaning: it is a list of ops and the labels the tag
//!   claimed.
//! * `match_site`: verify the shape from the ops' definitions and bind everything the
//!   replacement builders need (inputs, outputs, the ops of the site). A wrong proposal is
//!   rejected, never rewritten. Facts that cannot be derived from the ops yet are recorded on
//!   the match as named `Fact`s with their source, so an analysis can supply them later.
//! * alternatives: the replacements, each with the ids of its proofs in
//!   `verification/rewrites.py`. Builders take the match's bound operands.

use super::graph::SsaGraph;

/// Where a fact on a match comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FactSource {
    /// Derived from (and checked against) the matched ops.
    Structure,
    /// A whole-program invariant from the problem or the emission options (e.g. the index
    /// form, where the tree lives in memory). Holds for every site of the program.
    Program,
    /// Read from the region tag only. Labels such as round and group are identity, not
    /// semantics; any fact with this source is one an analysis must supply for untagged code.
    Tag,
}

impl FactSource {
    /// Spelling in JSON.
    pub fn as_str(&self) -> &'static str {
        match self {
            FactSource::Structure => "structure",
            FactSource::Program => "program",
            FactSource::Tag => "tag",
        }
    }
}

/// A named fact a match relies on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    /// What is known, e.g. `"depth"`.
    pub name: &'static str,
    /// Its value, printed.
    pub value: String,
    /// Where it comes from.
    pub source: FactSource,
}

/// A place to look for a site: the ops a region tag groups and the labels it claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    /// Region id of the tag that proposed it.
    pub region_id: u32,
    /// Claimed round.
    pub round: Option<u32>,
    /// Claimed group.
    pub group: Option<u32>,
    /// Claimed tree depth.
    pub depth: Option<u32>,
    /// Indices of the tagged ops, in order.
    pub ops: Vec<usize>,
}

/// Why a proposal is not a site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection(pub String);

/// One replacement a rule can build, with the proofs that justify it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlternativeSpec {
    /// Name, as in emission plans (`"gather"`, `"bits"`, ...).
    pub name: &'static str,
    /// Ids of the Z3 checks in `verification/rewrites.py` that prove it.
    pub proofs: &'static [&'static str],
}

/// A rewrite rule: candidate sites, a structural matcher and named alternatives.
pub trait Rule {
    /// What a successful match binds.
    type Site;

    /// Stable rule id (used in JSON and plans).
    fn id(&self) -> &'static str;

    /// The alternatives a site of this rule can be rewritten to.
    fn alternatives(&self) -> &'static [AlternativeSpec];

    /// Candidate sites in `graph` (from tags today). Order is deterministic.
    fn propose(&self, graph: &SsaGraph) -> Vec<Proposal>;

    /// Verify a proposal structurally and bind the site.
    ///
    /// # Arguments
    /// * `graph` - The program (or the part emitted so far).
    /// * `proposal` - Where to look; its labels are claims to check, not facts.
    ///
    /// # Returns
    /// The site, or why the ops do not have the rule's shape.
    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<Self::Site, Rejection>;
}
