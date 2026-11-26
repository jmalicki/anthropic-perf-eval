//! Graph surgery on a finished SSA program: insert ops, rewire readers, remove what the
//! rewrite made dead, and keep the constant set's scheduling alternatives consistent.
//!
//! A rule's replacement builder records edits against the original op indices; `finish`
//! applies them in one pass:
//!
//! 1. Splice: ops inserted (or moved) before / after an original op appear there, in the
//!    order the edits were made.
//! 2. Rewire: every renamed VReg is replaced in `srcs`, `keep_live` and `delta_vreg`.
//! 3. Remove: the ops the rule removed, then every op the rewrite made dead. "Made dead" is
//!    exact: an op goes if dead code elimination (`eliminate_ssa_dead_code`) keeps it in the
//!    original program and not in the rewritten one. Ops that were already dead stay (the
//!    pipeline's DCE removes them later), so the constants the emitter created, used or not,
//!    still exist where they were created; only an already-dead op that reads a removed
//!    value is dropped with it.
//! 4. Constants (catalogue coupling C13): a `ConstChoice`'s ALU-derived alternatives are the
//!    derivations whose operand values are constants created before it
//!    (`SsaEmitContext::emit_const_choice`). Creating or removing a constant can change the
//!    alternatives of every later constant. `new` marks the constants whose alternatives are
//!    this function of the ordered constant set (all of them in emitted programs); `finish`
//!    recomputes those for the rewritten order and reports each change. Any other
//!    ("foreign") alternatives are kept as given, less the derivations whose operands no
//!    longer exist before the constant, so no alternative ever names a missing constant.
//! 5. Check: nothing reads a removed value, every value is defined before it is read
//!    (`delta_vreg` and `keep_live` included).
//! 6. Renumber op ids in order and rebuild `vreg_def`.
//!
//! Preconditions (`Surgery::check`, also run by `new`): single assignment (every VReg
//! defined by at most one op), def before use, no dependencies computed yet, no legacy
//! dual-op constants. A program that fails them is refused as a whole; rules turn shapes
//! they cannot rewrite safely into per-site rejections before getting here.
//!
//! New constants and broadcasts go through `constant` / `broadcast`, which reuse an existing
//! op with the same value (or source) created at or before the insertion point, as the
//! emitter's CSE does, so a rewrite never duplicates a constant the program already has.
//! Maps are used for lookup only; edits are kept in `BTreeMap`s, so output order is
//! deterministic.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::types::{compute_const_encodings_alu_ready, ConstEncoding, Engine, Opcode, Operand, RegionTag, SsaOp, SymmetryHint, VReg, VarSource, VarType};

use super::super::emit_ssa::eliminate_ssa_dead_code;
use super::super::SsaProgram;
use super::graph::{op_reads, SsaGraph};

/// Metadata a new op takes over from an existing one: scheduling hints and accounting labels,
/// never semantics.
#[derive(Clone, Debug, Default)]
pub struct Stamp {
    /// `SsaOp::source_info`.
    pub source_info: Option<VarSource>,
    /// `SsaOp::region`.
    pub region: Option<RegionTag>,
    /// `SsaOp::symmetry_hint`.
    pub symmetry_hint: Option<SymmetryHint>,
    /// `SsaOp::var_type`.
    pub var_type: Option<VarType>,
}

impl Stamp {
    /// The metadata of `op`.
    ///
    /// # Arguments
    /// * `op` - The op to copy from.
    ///
    /// # Returns
    /// Its source info, region tag, symmetry hint and variable type.
    pub fn of(op: &SsaOp) -> Self {
        Stamp {
            source_info: op.source_info.clone(),
            region: op.region,
            symmetry_hint: op.symmetry_hint.clone(),
            var_type: op.var_type,
        }
    }

    /// The same stamp with another variable type.
    pub fn with_var_type(mut self, var_type: Option<VarType>) -> Self {
        self.var_type = var_type;
        self
    }

    fn apply(&self, op: &mut SsaOp) {
        op.source_info = self.source_info.clone();
        op.region = self.region;
        op.symmetry_hint = self.symmetry_hint.clone();
        op.var_type = self.var_type;
    }
}

/// A `ConstChoice` whose alternatives the rewrite changed (coupling C13).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlternativesChange {
    /// The constant.
    pub value: i64,
    /// Alternatives before the rewrite (`None` for a constant the rewrite created).
    pub before: Option<Vec<ConstEncoding>>,
    /// Alternatives after.
    pub after: Vec<ConstEncoding>,
}

/// What `Surgery::finish` did.
#[derive(Clone, Debug, Default)]
pub struct SurgeryReport {
    /// Ops inserted (and still live).
    pub inserted: usize,
    /// Ops removed because the rule removed them.
    pub removed: usize,
    /// Original ops moved (`move_after`).
    pub moved: usize,
    /// Ops removed because the rewrite made them dead, as `(original index, opcode, constant)`.
    pub cascaded: Vec<(usize, Opcode, Option<i64>)>,
    /// New ops that ended up dead and were dropped.
    pub dropped_new: usize,
    /// Ops that were already dead and read a value the rewrite removed, dropped with it.
    pub dropped_dead: usize,
    /// `ConstChoice` ops whose alternatives changed, in program order.
    pub alternatives: Vec<AlternativesChange>,
    /// `ConstChoice` ops whose input alternatives were not the emitter's function of the
    /// constants before them: kept as given, less derivations from constants that no longer
    /// exist before them.
    pub foreign_alternatives: usize,
}

/// An op spliced next to an original op: a new one, or an original one moved there.
enum Slot {
    New(SsaOp),
    Moved(usize),
}

/// Edits against a finished program; applied by `finish`.
pub struct Surgery {
    program: SsaProgram,
    removed: BTreeSet<usize>,
    before: BTreeMap<usize, Vec<Slot>>,
    after: BTreeMap<usize, Vec<Slot>>,
    /// Original ops placed elsewhere (`move_after`): op index -> the original op it now follows.
    moved: BTreeMap<usize, usize>,
    rename: BTreeMap<VReg, VReg>,
    /// Value -> indices of `ConstChoice` / `Const` ops producing it, increasing (lookup only).
    const_ops: HashMap<i64, Vec<usize>>,
    /// Scalar -> indices of `vbroadcast` ops of it, increasing (lookup only).
    broadcast_ops: HashMap<VReg, Vec<usize>>,
    /// `ConstChoice` ops whose alternatives are not the emitter's function (kept as given).
    foreign: BTreeSet<usize>,
    /// Constants and broadcasts created so far: key -> (anchor index, VReg).
    new_consts: BTreeMap<i64, (usize, VReg)>,
    new_broadcasts: BTreeMap<VReg, (usize, VReg)>,
}

/// Alternatives of every `ConstChoice` for this order.
///
/// An op with `foreign(i)` false gets them as `emit_const_choice` computes them: Load,
/// AddImm, then up to two ALU derivations from constants created before it (and 0 when the
/// program reserves the zero register). A foreign op keeps its own alternatives, less those
/// not ready from the constants before it. Ops without alternatives keep `None`.
///
/// # Arguments
/// * `ops` - The ops in order.
/// * `zero` - The program reserves the zero register.
/// * `foreign` - Whether op `i` keeps its own alternatives.
///
/// # Returns
/// One entry per op.
pub(super) fn recompute_alternatives(ops: &[SsaOp], zero: bool, foreign: impl Fn(usize) -> bool) -> Vec<Option<Vec<ConstEncoding>>> {
    let mut available: HashSet<i64> = HashSet::new();
    if zero {
        available.insert(0);
    }
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if op.opcode != Opcode::ConstChoice {
            out.push(None);
            continue;
        }
        let value = op.produces_constant.unwrap_or(0);
        out.push(op.const_alternatives.as_ref().map(|own| {
            if foreign(i) {
                return own.iter().filter(|e| e.is_ready(&available)).cloned().collect();
            }
            let mut alts = vec![ConstEncoding::Load, ConstEncoding::AddImm { value }];
            alts.extend(compute_const_encodings_alu_ready(value, &available, 2));
            alts
        }));
        if let Some(v) = op.produces_constant {
            available.insert(v);
        }
    }
    out
}

/// Which ops dead code elimination keeps, by index.
///
/// A value named in a kept op's `keep_live` or `delta_vreg` is kept too (with what it reads):
/// the emitter's dead code elimination follows `srcs` only, so on this private copy those
/// entries are appended to `srcs`.
fn dce_live(ops: &[SsaOp], program: &SsaProgram) -> Vec<bool> {
    let mut ops: Vec<SsaOp> = ops.to_vec();
    let mut vreg_def: HashMap<VReg, usize> = HashMap::new();
    for (i, op) in ops.iter_mut().enumerate() {
        op.id = i;
        if let Some(keep) = op.keep_live.clone() {
            op.srcs.extend(keep.into_iter().map(Operand::VReg));
        }
        if let Some(delta) = op.delta_vreg {
            op.srcs.push(Operand::VReg(delta));
        }
        if let Some(d) = op.dest {
            vreg_def.entry(d).or_insert(i);
        }
    }
    let n = ops.len();
    let mut legacy = Vec::new();
    let (kept, _, _) = eliminate_ssa_dead_code(
        ops,
        &mut vreg_def,
        &mut legacy,
        program.zero_vreg,
        Some(&program.vector_group_lanes),
    );
    let mut live = vec![false; n];
    for op in &kept {
        live[op.id] = true;
    }
    live
}

/// Check that every VReg an op reads (or keeps live) is defined by an earlier op: the zero
/// register, a vector group alias whose lanes are all defined, or a lane of a defined vector.
///
/// # Arguments
/// * `ops` - The ops in order.
/// * `program` - Supplies the zero register and the vector groups.
///
/// # Returns
/// The first violation.
fn check_def_before_use(ops: &[SsaOp], program: &SsaProgram) -> Result<(), String> {
    let graph = SsaGraph::new(ops, &program.vector_group_lanes);
    let mut defined: HashSet<VReg> = HashSet::new();
    for (i, op) in ops.iter().enumerate() {
        for v in op_reads(op).chain(op.keep_live.iter().flatten().copied()) {
            if Some(v) == program.zero_vreg {
                continue;
            }
            let ok = defined.contains(&v)
                || graph.group_lanes(v).map_or(false, |ls| ls.iter().all(|l| defined.contains(l)))
                || graph.element_of(v).map_or(false, |(vec, _)| defined.contains(&vec));
            if !ok {
                return Err(match graph.def_index(v) {
                    None if graph.group_lanes(v).is_none() && graph.element_of(v).is_none() => {
                        format!("op {} ({:?}) reads {:?}, which nothing defines", i, op.opcode, v)
                    }
                    d => format!("op {} ({:?}) reads {:?} before its definition (op {:?})", i, op.opcode, v, d),
                });
            }
        }
        if let Some(d) = op.dest {
            defined.insert(d);
        }
    }
    Ok(())
}

impl Surgery {
    /// Whether `program` can be edited: single assignment, def before use, no dependencies
    /// computed yet, no legacy dual-op constants. `new` runs it; call it first to keep the
    /// program when it fails (`new` consumes it).
    ///
    /// # Arguments
    /// * `program` - The program.
    ///
    /// # Returns
    /// Why it cannot be edited, if so.
    pub fn check(program: &SsaProgram) -> Result<(), String> {
        if !program.const_alternatives.is_empty() {
            return Err("legacy dual-op constants (emit_const) are not supported".into());
        }
        if let Some(op) = program.ops.iter().find(|o| !o.dep_ids.is_empty() || !o.mem_deps.is_empty()) {
            return Err(format!("op {} already has dependencies: run transforms before ComputeDependencies", op.id));
        }
        let mut defined: HashSet<VReg> = HashSet::new();
        for (i, op) in program.ops.iter().enumerate() {
            if let Some(d) = op.dest {
                if !defined.insert(d) {
                    return Err(format!("{:?} is defined more than once (again by op {}): not single assignment", d, i));
                }
            }
        }
        check_def_before_use(&program.ops, program)
    }

    /// Start editing `program`.
    ///
    /// # Arguments
    /// * `program` - A finished program that passes `check`.
    ///
    /// # Returns
    /// The editor, or why the program cannot be edited safely (`check`).
    pub fn new(program: SsaProgram) -> Result<Self, String> {
        Self::check(&program)?;
        let zero = program.zero_vreg.is_some();
        let recomputed = recompute_alternatives(&program.ops, zero, |_| false);
        let foreign: BTreeSet<usize> = program
            .ops
            .iter()
            .zip(&recomputed)
            .enumerate()
            .filter(|(_, (op, alts))| op.opcode == Opcode::ConstChoice && op.const_alternatives != **alts)
            .map(|(i, _)| i)
            .collect();
        let mut const_ops: HashMap<i64, Vec<usize>> = HashMap::new();
        let mut broadcast_ops: HashMap<VReg, Vec<usize>> = HashMap::new();
        for (i, op) in program.ops.iter().enumerate() {
            match op.opcode {
                Opcode::ConstChoice | Opcode::Const => {
                    // Only a constant whose immediate agrees with its annotation is reused.
                    if let (Some(v), Some(Operand::Imm(imm))) = (op.produces_constant, op.srcs.first()) {
                        if *imm == v {
                            const_ops.entry(v).or_default().push(i);
                        }
                    }
                }
                Opcode::Vbroadcast => {
                    if let Some(s) = op.srcs.first().and_then(|s| s.as_vreg()) {
                        broadcast_ops.entry(s).or_default().push(i);
                    }
                }
                _ => {}
            }
        }
        Ok(Surgery {
            program,
            removed: BTreeSet::new(),
            before: BTreeMap::new(),
            after: BTreeMap::new(),
            moved: BTreeMap::new(),
            rename: BTreeMap::new(),
            const_ops,
            broadcast_ops,
            foreign,
            new_consts: BTreeMap::new(),
            new_broadcasts: BTreeMap::new(),
        })
    }

    /// The original ops (indices are the ones edits refer to).
    pub fn ops(&self) -> &[SsaOp] {
        &self.program.ops
    }

    /// The original program's vector groups.
    pub fn lanes(&self) -> &HashMap<VReg, Vec<VReg>> {
        &self.program.vector_group_lanes
    }

    /// A VReg never used before.
    ///
    /// # Arguments
    /// * `vector` - Vector (8 words) or scalar.
    ///
    /// # Returns
    /// The new VReg.
    pub fn fresh(&mut self, vector: bool) -> VReg {
        let id = self.program.vreg_count;
        self.program.vreg_count += 1;
        if vector {
            VReg::vector(id)
        } else {
            VReg::scalar(id)
        }
    }

    /// Insert `op` right before original op `at` (after earlier insertions there).
    ///
    /// # Arguments
    /// * `at` - Original op index.
    /// * `op` - The op; its id is assigned by `finish`.
    /// * `stamp` - Metadata it takes.
    pub fn insert_before(&mut self, at: usize, mut op: SsaOp, stamp: &Stamp) {
        stamp.apply(&mut op);
        self.before.entry(at).or_default().push(Slot::New(op));
    }

    /// Insert `op` right after original op `at` (after earlier insertions there).
    ///
    /// # Arguments
    /// * `at` - Original op index.
    /// * `op` - The op.
    /// * `stamp` - Metadata it takes.
    pub fn insert_after(&mut self, at: usize, mut op: SsaOp, stamp: &Stamp) {
        stamp.apply(&mut op);
        self.after.entry(at).or_default().push(Slot::New(op));
    }

    /// Move original ops `idxs` to right after original op `at` (after earlier insertions
    /// there), in increasing index order, keeping their metadata.
    ///
    /// # Arguments
    /// * `idxs` - The ops to move; each must come before `at`.
    /// * `at` - Original op index they go after.
    ///
    /// # Returns
    /// An error if an op outside the group that is not removed reads one of them, or names its
    /// value in `keep_live`, and is placed at or before `at` (its own position, or the op an
    /// earlier `move_after` put it after), if one is not before `at`, or if one was already
    /// moved or removed (an op has one placement); then nothing is recorded.
    pub fn move_after(&mut self, idxs: &[usize], at: usize) -> Result<(), String> {
        let group: BTreeSet<usize> = idxs.iter().copied().collect();
        let graph = SsaGraph::new(&self.program.ops, &self.program.vector_group_lanes);
        // The op defining `v`, or the vector `v` is a lane of.
        let definer = |v: VReg| graph.def_index(v).or_else(|| graph.element_of(v).and_then(|(vec, _)| graph.def_index(vec)));
        for &idx in &group {
            if idx >= at {
                return Err(format!("op {} is not before op {}", idx, at));
            }
            if self.moved.contains_key(&idx) {
                return Err(format!("op {} is already moved", idx));
            }
            if self.removed.contains(&idx) {
                return Err(format!("op {} is removed", idx));
            }
            // A reader outside the group that is placed at or before `at`.
            let before_at = |r: &usize| {
                !group.contains(r) && !self.removed.contains(r) && self.moved.get(r).copied().unwrap_or(*r) <= at
            };
            if let Some(&r) = graph.readers(idx).iter().find(|&r| before_at(r)) {
                return Err(format!("op {} is read by op {} before op {}", idx, r, at));
            }
            let keeper = (0..self.program.ops.len()).find(|&r| {
                before_at(&r) && self.program.ops[r].keep_live.iter().flatten().any(|&v| definer(v) == Some(idx))
            });
            if let Some(r) = keeper {
                return Err(format!("op {} is kept live by op {} before op {}", idx, r, at));
            }
            // A new op queued next to an original op at or before `at` that reads it (or keeps
            // it live).
            let queued = self.before.range(..=at).chain(self.after.range(..=at));
            let reads_idx = |op: &SsaOp| op_reads(op).chain(op.keep_live.iter().flatten().copied()).any(|v| definer(v) == Some(idx));
            if let Some((&k, _)) = queued.into_iter().find(|(_, slots)| slots.iter().any(|s| matches!(s, Slot::New(op) if reads_idx(op)))) {
                return Err(format!("op {} is read by an op inserted next to op {} before op {}", idx, k, at));
            }
        }
        for idx in group {
            self.moved.insert(idx, at);
            self.after.entry(at).or_default().push(Slot::Moved(idx));
        }
        Ok(())
    }

    /// Whether original op `i` is still in the program and placed at or before original op
    /// `after` (its own position, or the op it was moved after).
    fn visible_at(&self, i: usize, after: usize) -> bool {
        !self.removed.contains(&i) && self.moved.get(&i).copied().unwrap_or(i) <= after
    }

    /// A new vector op `dest = opcode(srcs)` (VALU), inserted before `at`.
    ///
    /// # Arguments
    /// * `at` - Original op index.
    /// * `opcode` - Vector opcode.
    /// * `srcs` - Vector operands.
    /// * `stamp` - Metadata.
    ///
    /// # Returns
    /// Its destination VReg.
    pub fn valu_before(&mut self, at: usize, opcode: Opcode, srcs: &[VReg], stamp: &Stamp) -> VReg {
        let dest = self.fresh(true);
        let op = SsaOp::new(0, Engine::Valu, opcode, Some(dest), srcs.iter().map(|&v| Operand::VReg(v)).collect());
        self.insert_before(at, op, stamp);
        dest
    }

    /// Remove original op `idx`. Its value must not be read after `finish` (rename it first).
    pub fn remove(&mut self, idx: usize) {
        self.removed.insert(idx);
    }

    /// Every reader of `old` reads `new` instead.
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        if old != new {
            self.rename.insert(old, new);
        }
    }

    /// The scalar constant `value` visible right after original op `after`: an existing
    /// `ConstChoice` / `Const` placed at or before it (the first such, as the emitter's CSE
    /// returns; a moved op counts at the op it was moved after), one this surgery created at an anchor at or before it, the zero register for
    /// 0, or else a new `ConstChoice` inserted after `after` (alternatives set by `finish`).
    ///
    /// # Arguments
    /// * `value` - The constant (a 32-bit word).
    /// * `after` - Original op index of the insertion point.
    /// * `stamp` - Metadata of a new op.
    ///
    /// # Returns
    /// The scalar VReg holding `value`.
    pub fn constant(&mut self, value: i64, after: usize, stamp: &Stamp) -> VReg {
        if value == 0 {
            if let Some(z) = self.program.zero_vreg {
                return z;
            }
        }
        if let Some(&i) = self.const_ops.get(&value).and_then(|v| v.iter().find(|&&i| self.visible_at(i, after))) {
            return self.program.ops[i].dest.expect("constant op has a dest");
        }
        if let Some(&(a, v)) = self.new_consts.get(&value) {
            if a <= after {
                return v;
            }
        }
        let dest = self.fresh(false);
        let mut op = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(dest), vec![Operand::Imm(value)]);
        op.produces_constant = Some(value);
        op.const_alternatives = Some(Vec::new()); // recomputed by `finish`
        self.insert_after(after, op, stamp);
        self.new_consts.insert(value, (after, dest));
        dest
    }

    /// `vbroadcast(scalar)` visible right after original op `after`: an existing broadcast of
    /// `scalar` placed at or before it (a moved one counts at the op it was moved after), one this surgery created, or a new one inserted after
    /// `after`.
    ///
    /// # Arguments
    /// * `scalar` - The scalar VReg.
    /// * `after` - Original op index of the insertion point.
    /// * `stamp` - Metadata of a new op.
    ///
    /// # Returns
    /// The vector VReg.
    pub fn broadcast(&mut self, scalar: VReg, after: usize, stamp: &Stamp) -> VReg {
        if let Some(&i) = self.broadcast_ops.get(&scalar).and_then(|v| v.iter().find(|&&i| self.visible_at(i, after))) {
            return self.program.ops[i].dest.expect("broadcast has a dest");
        }
        if let Some(&(a, v)) = self.new_broadcasts.get(&scalar) {
            if a <= after {
                return v;
            }
        }
        let dest = self.fresh(true);
        let mut op = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(dest), vec![Operand::VReg(scalar)]);
        op.produces_constant = self.constant_value(scalar);
        self.insert_after(after, op, stamp);
        self.new_broadcasts.insert(scalar, (after, dest));
        dest
    }

    /// The vector constant `value` (`vbroadcast` of the scalar constant) visible after
    /// `after`, reusing what exists.
    ///
    /// # Arguments
    /// * `value` - The constant.
    /// * `after` - Original op index of the insertion point.
    /// * `scalar_stamp` - Metadata of a new scalar constant.
    /// * `vector_stamp` - Metadata of a new broadcast.
    ///
    /// # Returns
    /// The vector VReg.
    pub fn vector_constant(&mut self, value: i64, after: usize, scalar_stamp: &Stamp, vector_stamp: &Stamp) -> VReg {
        let s = self.constant(value, after, scalar_stamp);
        self.broadcast(s, after, vector_stamp)
    }

    fn constant_value(&self, v: VReg) -> Option<i64> {
        if Some(v) == self.program.zero_vreg {
            return Some(0);
        }
        if let Some((&value, _)) = self.new_consts.iter().find(|(_, &(_, nv))| nv == v) {
            return Some(value);
        }
        self.program
            .ops
            .iter()
            .find(|o| o.dest == Some(v) && matches!(o.opcode, Opcode::ConstChoice | Opcode::Const))
            .and_then(|o| o.produces_constant)
    }

    /// Apply the edits.
    ///
    /// # Returns
    /// The rewritten program (op ids renumbered in order, `vreg_def` rebuilt, not dead code
    /// eliminated beyond what the rewrite killed) and the report, or an error if a removed
    /// value is still read.
    pub fn finish(self) -> Result<(SsaProgram, SurgeryReport), String> {
        let Surgery { program, removed, mut before, mut after, moved, rename, foreign, .. } = self;
        // Every rename chain must end (resolve below then terminates).
        for &start in rename.keys() {
            let mut seen: HashSet<VReg> = HashSet::from([start]);
            let mut v = start;
            while let Some(&to) = rename.get(&v) {
                if !seen.insert(to) {
                    return Err(format!("rename cycle through {:?}", start));
                }
                v = to;
            }
        }
        let resolve = |mut v: VReg| {
            while let Some(&to) = rename.get(&v) {
                v = to;
            }
            v
        };
        let live_before = dce_live(&program.ops, &program);

        // 1. Splice, with the origin of each op.
        let mut seq: Vec<(SsaOp, Option<usize>)> = Vec::with_capacity(program.ops.len());
        // A moved op that was removed afterwards is not placed anywhere.
        let slot = |s: Slot| match s {
            Slot::New(op) => Some((op, None)),
            Slot::Moved(j) if removed.contains(&j) => None,
            Slot::Moved(j) => Some((program.ops[j].clone(), Some(j))),
        };
        for (i, op) in program.ops.iter().enumerate() {
            seq.extend(before.remove(&i).unwrap_or_default().into_iter().filter_map(slot));
            if !removed.contains(&i) && !moved.contains_key(&i) {
                seq.push((op.clone(), Some(i)));
            }
            seq.extend(after.remove(&i).unwrap_or_default().into_iter().filter_map(slot));
        }
        // 2. Rewire.
        if !rename.is_empty() {
            for (op, _) in &mut seq {
                for s in &mut op.srcs {
                    if let Operand::VReg(v) = s {
                        *v = resolve(*v);
                    }
                }
                if let Some(keep) = op.keep_live.as_mut() {
                    for v in keep.iter_mut() {
                        *v = resolve(*v);
                    }
                }
                if let Some(d) = op.delta_vreg.as_mut() {
                    *d = resolve(*d);
                }
            }
        }
        // Single assignment again, before dead ops are dropped: a rule that inserts a second
        // definer of a VReg (e.g. two replacements of one op) must fail here. Checked later, the
        // second definer would already be gone as "dead" (liveness follows the first one), and
        // the program would silently use whichever came first.
        {
            let mut defined: HashSet<VReg> = HashSet::new();
            for (op, _) in &seq {
                if let Some(d) = op.dest {
                    if !defined.insert(d) {
                        return Err(format!("after the rewrite: {:?} is defined more than once", d));
                    }
                }
            }
        }
        // 3. Remove what the rewrite made dead.
        let ops_only: Vec<SsaOp> = seq.iter().map(|(o, _)| o.clone()).collect();
        let live_after = dce_live(&ops_only, &program);
        drop(ops_only);
        let mut report = SurgeryReport { removed: removed.len(), moved: moved.len(), ..Default::default() };
        let mut kept: Vec<(SsaOp, Option<usize>)> = Vec::with_capacity(seq.len());
        for (k, (op, origin)) in seq.into_iter().enumerate() {
            let dead_now = !live_after[k];
            match origin {
                Some(i) if dead_now && live_before[i] => report.cascaded.push((i, op.opcode, op.produces_constant)),
                None if dead_now => report.dropped_new += 1,
                None => {
                    report.inserted += 1;
                    kept.push((op, origin));
                }
                Some(_) => kept.push((op, origin)),
            }
        }
        // An op that was already dead may read a value the rewrite removed (e.g. an unused
        // broadcast of a constant whose last live reader was a removed op): it goes too.
        loop {
            let ops_only: Vec<SsaOp> = kept.iter().map(|(o, _)| o.clone()).collect();
            let graph = SsaGraph::new(&ops_only, &program.vector_group_lanes);
            let dangling = |op: &SsaOp| {
                op_reads(op).chain(op.keep_live.iter().flatten().copied()).any(|v| {
                    Some(v) != program.zero_vreg
                        && graph.def_index(v).is_none()
                        && graph.group_lanes(v).is_none()
                        && graph.element_of(v).is_none()
                })
            };
            let before_len = kept.len();
            let mut next = Vec::with_capacity(kept.len());
            for (op, origin) in kept {
                if matches!(origin, Some(i) if !live_before[i]) && dangling(&op) {
                    report.dropped_dead += 1;
                } else {
                    next.push((op, origin));
                }
            }
            kept = next;
            if kept.len() == before_len {
                break;
            }
        }
        let (mut ops, origins): (Vec<SsaOp>, Vec<Option<usize>>) = kept.into_iter().unzip();

        // Nothing may read a value that no longer exists, and every value is defined before
        // it is read (the input's order, which the splice must keep).
        check_def_before_use(&ops, &program).map_err(|e| format!("after the rewrite: {}", e))?;

        // 4. Constants: alternatives for the new order (C13).
        let is_foreign = |k: usize| origins[k].map_or(false, |i| foreign.contains(&i));
        report.foreign_alternatives = (0..ops.len()).filter(|&k| is_foreign(k)).count();
        let recomputed = recompute_alternatives(&ops, program.zero_vreg.is_some(), is_foreign);
        for (op, alts) in ops.iter_mut().zip(recomputed) {
            if op.opcode != Opcode::ConstChoice || op.const_alternatives.is_none() {
                continue;
            }
            let alts = alts.expect("ConstChoice with alternatives");
            let created = op.const_alternatives.as_ref().map_or(false, |a| a.is_empty());
            if op.const_alternatives.as_ref() != Some(&alts) {
                if !created {
                    report.alternatives.push(AlternativesChange {
                        value: op.produces_constant.unwrap_or(0),
                        before: op.const_alternatives.clone(),
                        after: alts.clone(),
                    });
                }
                op.const_alternatives = Some(alts);
            }
        }

        // 5. Renumber.
        let mut vreg_def: HashMap<VReg, usize> = HashMap::new();
        for (i, op) in ops.iter_mut().enumerate() {
            op.id = i;
            if let Some(d) = op.dest {
                vreg_def.entry(d).or_insert(i);
            }
        }
        Ok((
            SsaProgram {
                ops,
                vreg_def,
                vreg_count: program.vreg_count,
                const_alternatives: Vec::new(),
                zero_vreg: program.zero_vreg,
                vector_group_lanes: program.vector_group_lanes,
            },
            report,
        ))
    }
}
