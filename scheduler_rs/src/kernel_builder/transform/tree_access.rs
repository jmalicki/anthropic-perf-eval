//! Rule: tree access at depth `d` in round `r` for group `g`.
//!
//! The canonical emission gathers: one vector address op, 8 scalar loads into a vector
//! group, one XOR with the deferred hash constant `C`. The alternatives select the node
//! among the cached level's values with the group's branch bits (`bits`, `bits-flow`,
//! `mobius`; `CachedSelect`). They are built by the emitter's own per-group code
//! (`emit_select_bits_from`, `emit_select_from_table`) on the operands the match binds.
//!
//! What the match derives from the ops (structure), and what it still takes from elsewhere:
//!
//! | Fact | Source |
//! |---|---|
//! | the site's output (node ^ C), index vector, `C`, the 10 ops of the gather | structure |
//! | depth `d` (mirrored index: from the level-end constant `L_d` of the address base, and again from the length of the index chain) | structure (mirrored), tag (zero-based) |
//! | branch bits `b_0 .. b_(d-1)`, the index before the last update, the ones vector | structure: the index chain `n = 2 n' + b` back to the depth-0 bit (mirrored only) |
//! | the index form (`n = L_d - idx`, R10/R11) | program (emission option) |
//! | the tree starts at the pointer in header word 4 | program (problem layout) |
//! | round and group | tag (labels only; nothing depends on them but plan keys) |
//!
//! Builder state the alternatives still read (what blocks untagged input): the cached level
//! tables (`ctx.cached_tables`: node table XORed with `C`, leaf pairs, Mobius coefficients,
//! built at init by `emit_cached_select_init`), checked to carry the site's `C`; and the
//! emitter's CSE caches, through which the builders find the shared vectors they use.

use std::collections::BTreeMap;

use crate::types::{Engine, MemLocation, MemRegion, Opcode, RegionKind, VReg, VLEN};

use super::super::config::{level_last_index, AccessEncoding, CachedSelect, IndexForm, STAGE5_XOR_CONST};
use super::super::tick_tock_ssa::{SsaTickTockKernelBuilder, UNROLL};
use super::graph::SsaGraph;
use super::rule::{AlternativeSpec, Fact, FactSource, Proposal, Rejection, Rule};

/// Rule id.
pub const TREE_ACCESS_RULE: &str = "tree-access";

/// Alternatives of a tree-access site, in plan spelling, with their proofs.
pub const TREE_ACCESS_ALTERNATIVES: [AlternativeSpec; 4] = [
    AlternativeSpec { name: "gather", proofs: &["R10-mirrored-index-uses", "R11-mirrored-index-range"] },
    AlternativeSpec { name: "bits", proofs: &["R13-select-by-path-bits", "R4-fma-select", "R15-bits-from-index"] },
    AlternativeSpec { name: "bits-flow", proofs: &["R13-select-by-path-bits", "R4-fma-select", "R15-bits-from-index"] },
    AlternativeSpec { name: "mobius", proofs: &["R14-mobius-select", "R15-bits-from-index"] },
];

/// Plan spelling of an encoding.
pub fn encoding_name(e: AccessEncoding) -> &'static str {
    match e {
        AccessEncoding::Gather => "gather",
        AccessEncoding::Standard => "standard",
        AccessEncoding::Bits => "bits",
        AccessEncoding::BitsFlow => "bits-flow",
        AccessEncoding::Mobius => "mobius",
    }
}

/// The bit-driven encodings, in plan order.
pub const BIT_DRIVEN: [AccessEncoding; 3] = [AccessEncoding::Bits, AccessEncoding::BitsFlow, AccessEncoding::Mobius];

/// The index chain of a site: how the index at depth `d` was built from branch bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexChain {
    /// Branch bits `b_0 .. b_(d-1)`, oldest first (each `value & ones`).
    pub bits: Vec<VReg>,
    /// Index before the last update (`None` at depth 1).
    pub prev: Option<VReg>,
    /// The vector of ones the bits are masked with.
    pub ones: VReg,
}

/// A matched tree-access site (gather form).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeAccessSite {
    /// Round label (from the tag).
    pub round: u32,
    /// Group label (from the tag).
    pub group: u32,
    /// Tree depth the index is at.
    pub depth: usize,
    /// The index vector (mirrored `n` or zero-based `idx`).
    pub index: VReg,
    /// The site's value: node ^ C, read by the node XOR.
    pub output: VReg,
    /// The gather's ops: address op, 8 loads, XOR (sorted indices).
    pub ops: Vec<usize>,
    /// The scalar deferred hash constant `C` the node is XORed with.
    pub hash_const: VReg,
    /// Branch bits for the bit-driven alternatives, when the chain matched.
    pub chain: Result<IndexChain, Rejection>,
    /// Facts the match relies on, with their sources.
    pub facts: Vec<Fact>,
}

/// The tree-access rule.
#[derive(Clone, Copy, Debug)]
pub struct TreeAccessRule {
    /// Index form of the program (a program fact).
    pub index_form: IndexForm,
}

impl TreeAccessRule {
    /// `Some(ones)` if `v = x & ones` with `ones` a vector of 1 (a 0/1 branch bit).
    fn bit_mask(graph: &SsaGraph, v: VReg) -> Option<VReg> {
        let op = graph.def_op(v)?;
        if op.engine != Engine::Valu || op.opcode != Opcode::And || op.srcs.len() != 2 {
            return None;
        }
        let ones = op.srcs[1].as_vreg()?;
        (graph.broadcast_const(ones) == Some(1)).then_some(ones)
    }

    /// Walk a mirrored index back to the depth-0 bit: `n = n' * 2 + b`, ..., `n_1 = b_0`.
    fn match_chain(graph: &SsaGraph, index: VReg, depth: usize) -> Result<IndexChain, Rejection> {
        let mut bits = Vec::new();
        let mut cur = index;
        let mut prev = None;
        let ones = loop {
            let op = graph
                .def_op(cur)
                .ok_or_else(|| Rejection(format!("index chain: {:?} has no definition", cur)))?;
            if op.opcode == Opcode::MultiplyAdd && op.srcs.len() == 3 {
                let (n, two, b) = (op.srcs[0].as_vreg(), op.srcs[1].as_vreg(), op.srcs[2].as_vreg());
                let (Some(n), Some(two), Some(b)) = (n, two, b) else {
                    return Err(Rejection("index chain: multiply_add with an immediate".into()));
                };
                if graph.broadcast_const(two) != Some(2) || Self::bit_mask(graph, b).is_none() {
                    return Err(Rejection("index chain: update is not n * 2 + (v & 1)".into()));
                }
                if prev.is_none() {
                    prev = Some(n);
                }
                bits.push(b);
                cur = n;
            } else if let Some(ones) = Self::bit_mask(graph, cur) {
                bits.push(cur);
                break ones;
            } else {
                return Err(Rejection(format!("index chain: {:?} is neither an update nor a branch bit", op.opcode)));
            }
        };
        bits.reverse();
        if bits.len() != depth {
            return Err(Rejection(format!("index chain has {} bits, the address says depth {}", bits.len(), depth)));
        }
        Ok(IndexChain { bits, prev: if depth >= 2 { prev } else { None }, ones })
    }

    /// True if `v` is the tree pointer: the scalar loaded from header word 4.
    fn is_forest_pointer(graph: &SsaGraph, v: VReg) -> bool {
        graph.def_op(v).map_or(false, |op| {
            op.opcode == Opcode::Load && op.mem_loc == Some(MemLocation::word(MemRegion::Header, 4))
        })
    }
}

impl Rule for TreeAccessRule {
    type Site = TreeAccessSite;

    fn id(&self) -> &'static str {
        TREE_ACCESS_RULE
    }

    fn alternatives(&self) -> &'static [AlternativeSpec] {
        &TREE_ACCESS_ALTERNATIVES
    }

    /// One proposal per tagged per-group gather region.
    fn propose(&self, graph: &SsaGraph) -> Vec<Proposal> {
        let mut by_id: BTreeMap<u32, Proposal> = BTreeMap::new();
        for (i, op) in graph.ops.iter().enumerate() {
            let Some(tag) = op.region else { continue };
            if tag.kind != RegionKind::TreeAccess(AccessEncoding::Gather) || tag.group.is_none() {
                continue;
            }
            by_id
                .entry(tag.id)
                .or_insert_with(|| Proposal {
                    region_id: Some(tag.id),
                    round: tag.round,
                    group: tag.group,
                    depth: tag.depth,
                    ops: Vec::new(),
                })
                .ops
                .push(i);
        }
        by_id.into_values().collect()
    }

    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<TreeAccessSite, Rejection> {
        let reject = |s: &str| Err(Rejection(s.to_string()));
        let (Some(round), Some(group)) = (proposal.round, proposal.group) else {
            return reject("a site needs round and group labels");
        };
        // Anchor: the XOR of a vector assembled lane by lane with a broadcast constant.
        let anchors: Vec<usize> = proposal
            .ops
            .iter()
            .copied()
            .filter(|&i| {
                let op = &graph.ops[i];
                op.engine == Engine::Valu
                    && op.opcode == Opcode::Xor
                    && op.srcs.first().and_then(|s| s.as_vreg()).map_or(false, |v| {
                        graph.def_index(v).is_none() && graph.group_lanes(v).map_or(false, |l| l.len() == VLEN)
                    })
            })
            .collect();
        let anchor = match anchors.as_slice() {
            [a] => *a,
            [] => return reject("no XOR of a gathered vector among the proposed ops"),
            _ => return reject("more than one gathered XOR among the proposed ops"),
        };
        let xor = &graph.ops[anchor];
        let gathered = xor.srcs[0].as_vreg().unwrap();
        let output = xor.dest.ok_or_else(|| Rejection("XOR without a result".into()))?;
        let c_vec = xor.srcs.get(1).and_then(|s| s.as_vreg()).ok_or_else(|| Rejection("XOR operand".into()))?;
        let hash_const = graph
            .broadcast_source(c_vec)
            .filter(|&c| graph.const_value(c) == Some(STAGE5_XOR_CONST))
            .ok_or_else(|| Rejection("the gathered vector is not XORed with the deferred hash constant".into()))?;

        // Each lane k: a scalar load from lane k of one decomposed address vector.
        let mut ops = vec![anchor];
        let mut address: Option<VReg> = None;
        for (k, &lane) in graph.group_lanes(gathered).unwrap().iter().enumerate() {
            let load_idx = graph.def_index(lane).ok_or_else(|| Rejection("lane without a load".into()))?;
            let load = &graph.ops[load_idx];
            if load.opcode != Opcode::Load {
                return reject("a lane is not written by a scalar load");
            }
            let a = load.srcs.first().and_then(|s| s.as_vreg()).ok_or_else(|| Rejection("load address".into()))?;
            match graph.element_of(a) {
                Some((vec, lane_k)) if lane_k == k && address.map_or(true, |v| v == vec) => address = Some(vec),
                _ => return reject("loads do not read the lanes of one address vector in order"),
            }
            ops.push(load_idx);
        }
        let address = address.unwrap();
        let addr_idx = graph.def_index(address).unwrap();
        let addr_op = &graph.ops[addr_idx];
        ops.push(addr_idx);
        let operands: Vec<Option<VReg>> = addr_op.srcs.iter().map(|s| s.as_vreg()).collect();
        let mirrored = self.index_form == IndexForm::Mirrored;
        let (base, index) = match (mirrored, addr_op.engine, addr_op.opcode, operands.as_slice()) {
            (true, Engine::Valu, Opcode::Sub, [Some(b), Some(n)]) => (*b, *n),
            (false, Engine::Valu, Opcode::Add, [Some(n), Some(b)]) => (*b, *n),
            _ => return reject("address is not base - n (mirrored) / idx + base (zero-based)"),
        };
        let pointer = graph.broadcast_source(base).ok_or_else(|| Rejection("address base is not a broadcast".into()))?;

        let mut facts = vec![
            Fact { name: "round", value: round.to_string(), source: FactSource::Tag },
            Fact { name: "group", value: group.to_string(), source: FactSource::Tag },
            Fact { name: "index_form", value: format!("{:?}", self.index_form), source: FactSource::Program },
            Fact { name: "tree_at_header_word_4", value: "forest_values_p".into(), source: FactSource::Program },
            Fact { name: "hash_const", value: STAGE5_XOR_CONST.to_string(), source: FactSource::Structure },
        ];
        let depth = if mirrored {
            // base = forest_p + L_d.
            let add = graph.def_op(pointer).ok_or_else(|| Rejection("address base pointer".into()))?;
            let parts: Vec<Option<VReg>> = add.srcs.iter().map(|s| s.as_vreg()).collect();
            let (Opcode::Add, [Some(p), Some(k)]) = (add.opcode, parts.as_slice()) else {
                return reject("mirrored address base is not forest_p + L_d");
            };
            if !Self::is_forest_pointer(graph, *p) {
                return reject("address base does not start at the tree pointer");
            }
            let l = graph.const_value(*k).ok_or_else(|| Rejection("level-end offset is not a constant".into()))?;
            let d = (0..63usize)
                .find(|&d| level_last_index(d) == l)
                .ok_or_else(|| Rejection(format!("{} is not a level-end index 2^(d+1) - 2", l)))?;
            if let Some(claimed) = proposal.depth {
                if claimed as usize != d {
                    return Err(Rejection(format!("tag claims depth {}, the address base is level {}", claimed, d)));
                }
            }
            facts.push(Fact { name: "depth", value: d.to_string(), source: FactSource::Structure });
            d
        } else {
            if !Self::is_forest_pointer(graph, pointer) {
                return reject("address base is not the tree pointer");
            }
            let d = proposal.depth.ok_or_else(|| Rejection("zero-based index: the depth comes only from the tag".into()))?;
            facts.push(Fact { name: "depth", value: d.to_string(), source: FactSource::Tag });
            d as usize
        };
        if depth == 0 {
            return reject("no gather at depth 0 (every lane is at the root)");
        }
        let chain = if mirrored {
            Self::match_chain(graph, index, depth)
        } else {
            Err(Rejection("zero-based index: branch bits are not matched structurally yet".into()))
        };
        if chain.is_ok() {
            facts.push(Fact { name: "branch_bits", value: format!("{} from the index chain", depth), source: FactSource::Structure });
        }
        ops.sort_unstable();
        Ok(TreeAccessSite { round, group, depth, index, output, ops, hash_const, chain, facts })
    }
}

/// One instantiated alternative of a site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Index into `TreeAccessInstantiation::sites`.
    pub site: usize,
    /// The alternative.
    pub encoding: AccessEncoding,
    /// Indices of the ops it emitted (the site's own ops for the canonical gather).
    pub ops: Vec<usize>,
    /// Its value (node ^ C).
    pub output: VReg,
}

/// Every choice site of a canonical program with every alternative built next to it
/// (`SsaTickTockKernelBuilder::new_instantiating`). The bit-driven candidates are dead
/// code in the program; the cost analysis and the apply pass choose among them.
#[derive(Clone, Debug, Default)]
pub struct TreeAccessInstantiation {
    /// Matched sites, in emission order.
    pub sites: Vec<TreeAccessSite>,
    /// Candidates, grouped by site in emission order (gather first).
    pub candidates: Vec<Candidate>,
    /// Proposals at choice depths that did not match, or alternatives that could not be built.
    pub rejections: Vec<(Proposal, Rejection)>,
}

impl SsaTickTockKernelBuilder {
    /// Build a bit-driven alternative of a site from the operands the match bound.
    ///
    /// # Arguments
    /// * `site` - The matched site.
    /// * `mode` - `Bits`, `BitsFlow` or `Mobius`.
    ///
    /// # Returns
    /// The alternative's value (node ^ C), or why it cannot be built here (no index chain,
    /// no cached table for the level, or a table built with another hash constant).
    pub(crate) fn build_bit_driven_alternative(
        &mut self,
        site: &TreeAccessSite,
        mode: CachedSelect,
    ) -> Result<VReg, Rejection> {
        let chain = site.chain.clone()?;
        // Builder state: the level's select constants, built at init.
        let table = self
            .ctx
            .cached_tables
            .get(site.depth)
            .cloned()
            .flatten()
            .ok_or_else(|| Rejection(format!("no cached table for level {}", site.depth)))?;
        if self.ctx.stage5_xor_s != Some(site.hash_const) {
            return Err(Rejection("the cached table carries another hash constant".into()));
        }
        let usable = match mode {
            CachedSelect::Bits => table.pairs.is_some(),
            CachedSelect::BitsFlow => table.pairs.as_ref().map_or(false, |p| p.iter().all(|x| x.high.is_some())),
            CachedSelect::Mobius => table.coefficients.is_some(),
            CachedSelect::Standard => false,
        };
        if !usable {
            return Err(Rejection(format!("the level-{} table has no {:?} constants", site.depth, mode)));
        }
        let bits = self.emit_select_bits_from(&chain.bits, chain.prev, chain.ones, site.depth);
        Ok(self.emit_select_from_table(mode, &bits, &table, site.depth))
    }

    /// Match the gather sites of round `rnd` in the ops emitted so far and build every
    /// bit-driven alternative of each next to it, tagged as its own region. Called by
    /// `emit_round` right after the round's tree access when instantiating.
    ///
    /// # Arguments
    /// * `rnd` - The round just emitted.
    pub(crate) fn instantiate_tree_access_round(&mut self, rnd: usize) {
        let rule = TreeAccessRule { index_form: self.config.spec.index_form };
        let matched: Vec<(Proposal, Result<TreeAccessSite, Rejection>)> = {
            let graph = SsaGraph::new(&self.emit.ops, &self.emit.vector_group_lanes);
            rule.propose(&graph)
                .into_iter()
                .filter(|p| p.round == Some(rnd as u32))
                .map(|p| {
                    let site = rule.match_site(&graph, &p);
                    (p, site)
                })
                .collect()
        };
        let mut inst = self.instantiation.take().expect("instantiating");
        for (proposal, site) in matched {
            let site = match site {
                Ok(site) => site,
                Err(e) => {
                    inst.rejections.push((proposal, e));
                    continue;
                }
            };
            let site_idx = inst.sites.len();
            inst.candidates.push(Candidate {
                site: site_idx,
                encoding: AccessEncoding::Gather,
                ops: site.ops.clone(),
                output: site.output,
            });
            let g = site.group as usize;
            for encoding in BIT_DRIVEN {
                // The emitter's per-group context, so the candidate's ops are those the
                // emitter would write for this group (symmetry hints, source info).
                self.emit.set_region(RegionKind::TreeAccess(encoding), None);
                self.emit.set_iter_group(g / UNROLL, g);
                let first = self.emit.ops.len();
                match self.build_bit_driven_alternative(&site, encoding.cached_select().unwrap()) {
                    Ok(output) => inst.candidates.push(Candidate {
                        site: site_idx,
                        encoding,
                        ops: (first..self.emit.ops.len()).collect(),
                        output,
                    }),
                    Err(e) => inst.rejections.push((proposal.clone(), e)),
                }
            }
            inst.sites.push(site);
        }
        self.emit.clear_iter_group();
        self.emit.clear_region();
        self.instantiation = Some(inst);
    }
}
