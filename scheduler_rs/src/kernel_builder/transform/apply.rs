//! Apply a tree-access selection: rewrite the canonical program into the planned one.
//!
//! Input: the canonical program with every alternative instantiated next to each site
//! (`instantiate_tree_access`) and a selection, one encoding per (round, group) (the same
//! information as an emission plan). Steps:
//!
//! 1. Rewire: the readers of each site's value read the chosen candidate's value instead.
//! 2. Order: within each round's tree access, ops are put in the order the emitter writes a
//!    mixed round (encodings in order of the first group using each, groups in order).
//! 3. Lazy constants: a constant of a gather's per-round base that the emitter creates on
//!    first use (the level-end index `L_d`) moves to the first round that still gathers at
//!    that depth, and its scheduling alternatives are recomputed against the constants that
//!    exist there.
//! 4. Dead code elimination removes the unchosen candidates and unused setup.
//!
//! The result is op for op the program the emitter writes for the plan, up to VReg and op
//! numbering (tested against `--emission-plan` emission in `tests.rs`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::types::{compute_const_encodings_alu_ready, ConstEncoding, Opcode, Operand, RegionKind, SsaOp, VReg};

use super::super::config::{AccessEncoding, ProblemSpec, MAX_TREE_REDUCE_DEPTH};
use super::super::emit_ssa::eliminate_ssa_dead_code;
use super::super::SsaProgram;
use super::cost::{canonical_plan, instantiate_tree_access};
use super::graph::SsaGraph;

/// Rewrite the canonical program of `spec` into the program of `plan`.
///
/// # Arguments
/// * `spec` - Emission options (the context; no tree-access options).
/// * `plan` - `[round][group]` encodings. Rounds outside depths 1 to `MAX_TREE_REDUCE_DEPTH`
///   must be canonical (bits at the root, gather below level 4).
///
/// # Returns
/// The planned program after dead code elimination, ops renumbered in order.
pub fn apply_tree_access_plan(spec: &ProblemSpec, plan: &[Vec<AccessEncoding>]) -> Result<SsaProgram, String> {
    let canonical = canonical_plan(spec);
    if plan.len() != spec.rounds || plan.iter().any(|row| row.len() != spec.n_groups()) {
        return Err("plan shape does not match the problem".into());
    }
    let choice = |r: usize| (1..=MAX_TREE_REDUCE_DEPTH as usize).contains(&spec.round_depth(r));
    for r in 0..spec.rounds {
        // At the root every bit-driven encoding is the one shared root vector.
        let root_alias = spec.round_depth(r) == 0 && plan[r].iter().all(|e| e.cached_select().is_some());
        if !choice(r) && plan[r] != canonical[r] && !root_alias {
            return Err(format!("round {} (depth {}) has no choice: it must be {:?}", r, spec.round_depth(r), canonical[r][0]));
        }
    }
    let (mut program, inst) = instantiate_tree_access(spec)?;
    if let Some((p, e)) = inst.rejections.first() {
        return Err(format!("round {:?} group {:?}: {}", p.round, p.group, e.0));
    }

    // 1. Rewire.
    let by_choice: BTreeMap<(u32, u32, AccessEncoding), usize> = inst
        .candidates
        .iter()
        .enumerate()
        .map(|(c, cand)| {
            let s = &inst.sites[cand.site];
            ((s.round, s.group, cand.encoding), c)
        })
        .collect();
    let mut rename: HashMap<VReg, VReg> = HashMap::new();
    for site in &inst.sites {
        let chosen = plan[site.round as usize][site.group as usize];
        let c = by_choice
            .get(&(site.round, site.group, chosen))
            .ok_or_else(|| format!("round {} group {}: no {:?} candidate", site.round, site.group, chosen))?;
        let out = inst.candidates[*c].output;
        if out != site.output {
            rename.insert(site.output, out);
        }
    }
    for op in &mut program.ops {
        for src in &mut op.srcs {
            if let Operand::VReg(v) = src {
                if let Some(&to) = rename.get(v) {
                    *v = to;
                }
            }
        }
        if let Some(keep) = op.keep_live.as_mut() {
            for v in keep.iter_mut() {
                if let Some(&to) = rename.get(v) {
                    *v = to;
                }
            }
        }
    }

    // 2. Order each round's tree access as the emitter does for a mixed round.
    let rank_in_round = |r: usize| -> Vec<AccessEncoding> {
        let mut order = Vec::new();
        for &e in &plan[r] {
            if !order.contains(&e) {
                order.push(e);
            }
        }
        order
    };
    let access_round = |op: &SsaOp| -> Option<(u32, AccessEncoding, Option<u32>)> {
        let t = op.region?;
        match t.kind {
            RegionKind::TreeAccess(e) => Some((t.round?, e, t.group)),
            _ => None,
        }
    };
    let mut ops = std::mem::take(&mut program.ops);
    let mut start = 0;
    while start < ops.len() {
        let Some((r, _, _)) = access_round(&ops[start]) else {
            start += 1;
            continue;
        };
        let mut end = start;
        while end < ops.len() && access_round(&ops[end]).map_or(false, |(rr, _, _)| rr == r) {
            end += 1;
        }
        if choice(r as usize) {
            let order = rank_in_round(r as usize);
            let key = |op: &SsaOp| {
                let (_, e, g) = access_round(op).unwrap();
                // Unused encodings of the round sort last; they are dead.
                let rank = order.iter().position(|&x| x == e).unwrap_or(order.len());
                (rank, g.map_or(0, |g| g as usize + 1))
            };
            ops[start..end].sort_by_key(|op| key(op)); // stable: emission order within a key
        }
        start = end;
    }

    // 3. Lazy constants of gather bases whose round no longer gathers.
    let gathers = |r: usize| plan[r].contains(&AccessEncoding::Gather);
    let is_base = |op: &SsaOp, r: u32| {
        op.region.map_or(false, |t| {
            t.kind == RegionKind::TreeAccess(AccessEncoding::Gather) && t.group.is_none() && t.round == Some(r)
        })
    };
    let lanes = std::mem::take(&mut program.vector_group_lanes);
    let mut moves: Vec<(usize, usize)> = Vec::new(); // (op index, index of the op to precede)
    let mut phantoms: Vec<usize> = Vec::new(); // lazy constants the plan never creates
    {
        let graph = SsaGraph::new(&ops, &lanes);
        let users = graph.users();
        for (i, op) in ops.iter().enumerate() {
            let Some(r) = op.region.and_then(|t| t.round) else { continue };
            if op.opcode != Opcode::ConstChoice || !is_base(op, r) || gathers(r as usize) || !choice(r as usize) {
                continue;
            }
            let target = users[i].iter().copied().find(|&u| {
                ops[u].region.and_then(|ut| ut.round).map_or(false, |ur| is_base(&ops[u], ur) && gathers(ur as usize))
            });
            match target {
                Some(u) => moves.push((i, u)),
                None => phantoms.push(i),
            }
        }

        // 3b. The broadcast of a level's first table entry is shared (CSE) by the leaf pairs
        // and the Mobius coefficients. The emitter creates it with the first pair; at a level
        // with Mobius but no pairs, with the coefficients: just before the next live broadcast
        // of the level (coefficient 1).
        let mut live = vec![false; ops.len()];
        for i in (0..ops.len()).rev() {
            let root = matches!(
                ops[i].opcode,
                Opcode::Store | Opcode::Vstore | Opcode::Halt | Opcode::Jump | Opcode::CondJump | Opcode::Barrier
            );
            live[i] = root || users[i].iter().any(|&u| live[u]);
        }
        let pairs_used = |d: usize| {
            (0..spec.rounds).any(|r| {
                spec.round_depth(r) == d
                    && plan[r].iter().any(|&e| e == AccessEncoding::Bits || e == AccessEncoding::BitsFlow)
            })
        };
        for d in 1..=MAX_TREE_REDUCE_DEPTH as usize {
            if pairs_used(d) {
                continue;
            }
            let broadcasts: Vec<usize> = (0..ops.len())
                .filter(|&i| {
                    live[i]
                        && ops[i].opcode == Opcode::Vbroadcast
                        && ops[i].region.map_or(false, |t| t.kind == RegionKind::CachedLevelSetup && t.depth == Some(d as u32))
                })
                .collect();
            if let [first, next, ..] = broadcasts.as_slice() {
                moves.push((*first, *next));
            }
        }
    }
    let moved: HashSet<usize> = moves.iter().map(|&(i, _)| i).collect();
    let before: BTreeMap<usize, Vec<usize>> = moves.iter().fold(BTreeMap::new(), |mut m, &(i, u)| {
        m.entry(u).or_insert_with(Vec::new).push(i);
        m
    });
    let phantom_set: HashSet<usize> = phantoms.into_iter().collect();
    let mut old: Vec<Option<(SsaOp, bool)>> =
        ops.into_iter().enumerate().map(|(i, op)| Some((op, phantom_set.contains(&i)))).collect();
    let mut reordered: Vec<(SsaOp, bool)> = Vec::with_capacity(old.len());
    for idx in 0..old.len() {
        if let Some(firsts) = before.get(&idx) {
            let anchor = old[idx].as_ref().unwrap().0.clone();
            for &i in firsts {
                let (mut op, phantom) = old[i].take().unwrap();
                // It is now created in the anchor's round.
                op.source_info = anchor.source_info.clone();
                op.region = anchor.region;
                op.symmetry_hint = anchor.symmetry_hint.clone();
                reordered.push((op, phantom));
            }
        }
        if !moved.contains(&idx) {
            reordered.push(old[idx].take().unwrap());
        }
    }
    // A ConstChoice's ALU-derived alternatives are those available from the constants created
    // before it (`emit_const_choice`); recompute them for the planned order, in which the
    // phantoms do not exist.
    let mut available: HashSet<i64> = HashSet::new();
    if program.zero_vreg.is_some() {
        available.insert(0);
    }
    let mut ops: Vec<SsaOp> = Vec::with_capacity(reordered.len());
    for (mut op, phantom) in reordered {
        if op.opcode == Opcode::ConstChoice && !phantom {
            let value = op.produces_constant.expect("ConstChoice carries its value");
            let mut alts = vec![ConstEncoding::Load, ConstEncoding::AddImm { value }];
            alts.extend(compute_const_encodings_alu_ready(value, &available, 2));
            op.const_alternatives = Some(alts);
            available.insert(value);
        }
        ops.push(op);
    }

    // Renumber ids in the new order, then dead code elimination.
    let mut vreg_def: HashMap<VReg, usize> = HashMap::new();
    for (i, op) in ops.iter_mut().enumerate() {
        op.id = i;
        if let Some(d) = op.dest {
            vreg_def.entry(d).or_insert(i);
        }
    }
    let (ops, vreg_def, const_alternatives) = eliminate_ssa_dead_code(
        ops,
        &mut vreg_def,
        &mut program.const_alternatives,
        program.zero_vreg,
        Some(&lanes),
    );
    Ok(SsaProgram {
        ops,
        vreg_def,
        vreg_count: program.vreg_count,
        const_alternatives,
        zero_vreg: program.zero_vreg,
        vector_group_lanes: lanes,
    })
}
