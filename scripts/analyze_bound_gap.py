#!/usr/bin/env python3
"""Why is our capacity bound (about 1,271) far above the public leaderboard (863)?

Reproduces every computed number in docs/BOUND_GAP_ANALYSIS.md. Read-only: it reads a
scheduler output JSON, runs arithmetic, and (optionally) Z3 bit-vector checks.

    uv run python scripts/analyze_bound_gap.py                      # sections 1-3, 5
    uv run --extra verify python scripts/analyze_bound_gap.py --z3  # plus section 4
    uv run python scripts/analyze_bound_gap.py --external-kernel /path/to/perf_takehome.py

--external-kernel never imports or executes the file. It only reads the author's
"# name: N ENGINE computation(s)" comments with a regular expression.
"""

from __future__ import annotations

import os
import sys

# scripts/bisect.py shadows the standard library's bisect (needed by random); drop this
# directory from the import path before importing anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAP = {"valu": 6, "alu": 12, "load": 2, "store": 2, "flow": 1}
VEQ_PER_CYCLE = CAP["valu"] + CAP["alu"] / 8  # 7.5 vector-op equivalents per cycle
N_GROUPS = 32
ROUNDS = 16
LEVELS = [r % 11 for r in range(ROUNDS)]  # tree level visited in each round (height 10)


def hr(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 1. Bound recomputed from the schedule JSON
# ----------------------------------------------------------------------------
def load_schedule(path):
    with open(path) as f:
        d = json.load(f)
    ssa_by_id = {o["id"]: o for o in d["ssa_ops"]}
    return d, ssa_by_id


def classify(d, ssa_by_id):
    """Per lowered op: (ssa op, engine class). A vector op shifted to scalar appears as 8
    lowered ALU ops with id = ssa_id*1000 + lane."""
    out = []
    for o in d["ops"]:
        if o["id"] in ssa_by_id:
            s, cls = ssa_by_id[o["id"]], ("alu_native" if o["engine"] == "alu" else o["engine"])
        else:
            s, cls = ssa_by_id[o["id"] // 1000], "alu_shift"
            assert o["engine"] == "alu"
        out.append((o, s, cls))
    return out


def section_bound(d, ssa_by_id):
    hr("1. Capacity bound recomputed from the JSON")
    ops = classify(d, ssa_by_id)
    c = collections.Counter(cls for _, _, cls in ops)
    assert c["alu_shift"] % 8 == 0
    shifted = c["alu_shift"] // 8
    valu, alu_native = c["valu"], c["alu_native"]
    load, store, flow = c["load"], c["store"], c["flow"]
    print(f"cycles in this schedule       : {d['total_cycles']}")
    print(f"vector ALU ops                : {valu}")
    print(f"vector ops run as 8 scalar ops: {shifted}")
    print(f"native scalar ALU ops         : {alu_native}")
    print(f"load / flow / store           : {load} / {flow} / {store}")

    ssa = d["ssa_ops"]
    ssa_valu = [o for o in ssa if o["engine"] == "valu"]
    fixed = sum(1 for o in ssa_valu if o["opcode"] in ("multiply_add", "vbroadcast"))
    shiftable = len(ssa_valu) - fixed
    const_choice = sum(1 for o in ssa if o["opcode"] == "const_choice")
    select_choice = sum(1 for o in ssa if o["opcode"] == "select_choice")
    V = len(ssa_valu)
    assert V == valu + shifted
    veq = V + alu_native / 8
    print(f"vector-op equivalents         : {V} + {alu_native}/8 = {veq:.2f}")
    print(f"continuous joint floor        : {veq:.2f} / {VEQ_PER_CYCLE} = {veq / VEQ_PER_CYCLE:.2f}")
    print(f"non-shiftable vector ops      : {fixed} (multiply_add, vbroadcast) -> {fixed / 6:.1f} cycles alone")

    # Same search as compute_capacity_lower_bound_global (bounds.rs), z FMA selects cost 2 VALU.
    load_fixed = load + sum(1 for o, s, _ in ops if o["engine"] == "flow" and s["opcode"] == "const_choice") - const_choice
    flow_fixed = flow - sum(1 for o, s, _ in ops if o["engine"] == "flow" and s["opcode"] == "const_choice") - select_choice
    best = None
    for z in (0, select_choice):  # extremes; VALU term is monotone in z, flow term anti-monotone
        for y in (0, const_choice):
            ld = math.ceil((load_fixed + const_choice - y) / 2)
            fl = flow_fixed + y + (select_choice - z)
            for x in range(shiftable + 1):  # x = shiftable ops left on the vector ALU
                v = math.ceil((fixed + x + 2 * z) / 6)
                a = math.ceil((alu_native + 8 * (shiftable - x)) / 12)
                lb = max(v, a, ld, fl, math.ceil(store / 2))
                if best is None or lb < best[0]:
                    best = (lb, x, y, z, v, a, ld, fl)
    lb, x, y, z, v, a, ld, fl = best
    print(f"integer bound (min over shift/const/select choices): {lb}")
    print(f"  at: {shiftable - x} ops shifted, {y} consts as add_imm, {z} selects as FMA")
    print(f"  per-engine terms there: valu={v} alu={a} load={ld} flow={fl} store={math.ceil(store / 2)}")

    # Dependency (critical path) bound: +1 per SSA edge.
    depth = {}
    for o in sorted(ssa, key=lambda o: o["id"]):
        depth[o["id"]] = 1 + max((depth.get(p, 0) for p in o["dep_ids"]), default=0)
    print(f"critical path (ops, +1 per SSA edge)               : {max(depth.values())}")
    print(f"peak live scratch words / limit                    : {d['peak_live_words']} / 1536")
    return ops


# ----------------------------------------------------------------------------
# 2. Per-round, per-phase budget of the emitted kernel
# ----------------------------------------------------------------------------
def section_budget(ops):
    hr("2. Emitted op budget (vector-op equivalents = valu + scalar/8)")
    per = collections.defaultdict(collections.Counter)
    for o, s, cls in ops:
        si = s["source_info"]
        per[(si["round"], si["phase"])][cls] += 1

    def veq(c):
        return c["valu"] + (c["alu_shift"] + c["alu_native"]) / 8

    kinds = {}
    for r in range(ROUNDS):
        lvl = LEVELS[r]
        kinds[r] = "gather" if per[(r, "tree_access")]["load"] else f"select L{lvl}"
    print("| Round | Level | Kind | xor | hash | branch | tree access | total veq | per group | loads | flow |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for r in range(ROUNDS):
        row = {p: veq(per[(r, p)]) for p in ("xor", "hash", "branch", "tree_access")}
        t = sum(row.values())
        ld = sum(per[(r, p)]["load"] for p in row)
        fl = sum(per[(r, p)]["flow"] for p in row)
        tot["veq"] += t
        tot["load"] += ld
        tot["flow"] += fl
        for p in row:
            tot[p] += row[p]
        print(f"| {r} | {LEVELS[r]} | {kinds[r]} | {row['xor']:.0f} | {row['hash']:.0f} | {row['branch']:.0f} | "
              f"{row['tree_access']:.0f} | {t:.0f} | {t / N_GROUPS:.2f} | {ld} | {fl} |")
    other = collections.Counter()
    for (r, p), c in per.items():
        if r is None:
            other["veq"] += veq(c)
            other["load"] += c["load"]
            other["flow"] += c["flow"]
            other["store"] += c["store"]
            print(f"| - | - | {p} | | | | | {veq(c):.1f} | | {c['load']} | {c['flow']} |")
    print(f"| all rounds | | | {tot['xor']:.0f} | {tot['hash']:.0f} | {tot['branch']:.0f} | {tot['tree_access']:.0f} | "
          f"{tot['veq']:.0f} | {tot['veq'] / N_GROUPS / ROUNDS:.2f} per group-round | {tot['load']} | {tot['flow']} |")
    grand = tot["veq"] + other["veq"]
    print(f"\nTOTAL veq = {grand:.2f}   loads = {tot['load'] + other['load']}   flow = {tot['flow'] + other['flow']}")
    for T in (1100, 1000, 863):
        print(f"budget at {T:>4} cycles: veq <= {T * VEQ_PER_CYCLE:.0f} ({T * VEQ_PER_CYCLE / 512:.2f} per group-round), "
              f"loads <= {2 * T}, flow <= {T};  our excess veq = {grand - T * VEQ_PER_CYCLE:.0f}, "
              f"excess loads = {tot['load'] + other['load'] - 2 * T}")


# ----------------------------------------------------------------------------
# 3. Design model: op counts and capacity floors for alternative emissions
# ----------------------------------------------------------------------------
def design(name, gather_rounds, hash_ops=10, overhead_veq=60, overhead_loads=40, overhead_flow=10,
           addr_ops=1, c5_in_gather=1, verbose=True):
    """Per-group op model for a kernel that keeps one scalar load per lane per gathered round.

    gather_rounds: dict round -> number of groups (0..32) that gather in that round; the other
    groups select among broadcast node vectors (2^level - 1 two-way selects).
    Costs per group (all vector ops):
      node xor 1/round; hash `hash_ops`/round; 1 final xor (deferred stage-5 constant);
      parity bit 1/round except rounds 10 and 15 (index is dead there);
      gather round: `addr_ops` address add + `c5_in_gather` xor + 8 loads;
      index chain: 1 multiply_add per round from round 1 (resp. 12) up to the round before the
        last gather of that pass, for groups that gather;
      select round at level d: 2^d - 1 selects, each either a flow vselect or one vector op.
    """
    veq = loads = selects = 0.0
    for r in range(ROUNDS):
        g = gather_rounds.get(r, 0)
        s = N_GROUPS - g
        veq += N_GROUPS * (1 + hash_ops) + (N_GROUPS if r not in (10, 15) else 0)
        veq += g * (addr_ops + c5_in_gather)
        loads += 8 * g
        selects += s * (2 ** LEVELS[r] - 1)
    veq += N_GROUPS  # final xor with the deferred constant
    # index chains
    g1 = max((g for r, g in gather_rounds.items() if r <= 10), default=0)
    last1 = max((r for r, g in gather_rounds.items() if r <= 10 and g), default=None)
    if last1 is not None:
        veq += g1 * max(0, last1 - 1)
    g2 = max((g for r, g in gather_rounds.items() if r > 10), default=0)
    last2 = max((r for r, g in gather_rounds.items() if r > 10 and g), default=None)
    if last2 is not None:
        veq += g2 * max(0, last2 - 12)
    veq += overhead_veq
    loads += overhead_loads
    # Smallest T with: loads <= 2T, flow_used <= T, veq + (selects - flow_used) <= 7.5 T
    T = 1
    while True:
        f = min(selects, T - overhead_flow)
        if loads <= 2 * T and veq + selects - f <= VEQ_PER_CYCLE * T:
            break
        T += 1
    if verbose:
        print(f"| {name} | {veq:.0f} | {selects:.0f} | {loads:.0f} | {math.ceil(loads / 2)} | "
              f"{math.ceil((veq + selects) / (VEQ_PER_CYCLE + 1))} | **{T}** |")
    return T, veq, selects, loads


def section_designs():
    hr("3. Composed designs (capacity floors, not achieved schedules)")
    print("| Design | vector ops excl. selects | 2-way selects | loads | load floor | ALU+flow floor | capacity floor |")
    print("|---|---|---|---|---|---|---|")
    full = N_GROUPS
    deep = {r: full for r in range(5, 11)}
    design("B0: today's structure, cleaned (cache 0-3, gather 4-10 and round 15)", {**deep, 4: full, 15: full})
    design("B1: as B0 but round 15 selects among 16 nodes", {**deep, 4: full})
    design("C0: cache levels 0-4, gather 5-10", deep)
    best = None
    for g4 in range(0, full + 1):
        for g15 in range(0, full + 1):
            T, *_ = design("", {**deep, 4: g4, 15: g15}, verbose=False)
            if best is None or T < best[0]:
                best = (T, g4, g15)
    T, g4, g15 = best
    design(f"C1: best mix (level 4 gathered by {g4} groups in round 4, {g15} in round 15)", {**deep, 4: g4, 15: g15})
    design("C2: C0 with selection free of charge (not achievable; shows the residual)", deep,
           overhead_flow=-10**6)
    design("C3: C0 with hypothetical 9-op hash (no identity known)", deep, hash_ops=9)
    print("\nEvery row keeps one scalar `load` per lane per gathered round. None reaches 863:")
    print(f"  863 * 7.5 = {863 * 7.5:.0f} vector-op equivalents; fixed work alone (xor+hash+bits+final) is "
          f"{N_GROUPS * (16 * 11 + 14 + 1)} -> {N_GROUPS * (16 * 11 + 14 + 1) / 7.5:.0f} cycles.")


# ----------------------------------------------------------------------------
# 4. Z3 identities (32-bit) and randomised check against problem.myhash
# ----------------------------------------------------------------------------
C0, C1, C2, C3, C4, C5 = 0x7ED55D16, 0xC761C23C, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09
M32 = 0xFFFFFFFF


def hash10(x):
    """The 10-op hash (lazy form): returns myhash(x) ^ C5."""
    a = (4097 * x + C0) & M32
    b = (a >> 19) ^ a ^ C1
    c = (33 * b + ((C2 + C3) & M32)) & M32
    dd = (16896 * b + ((C2 << 9) & M32)) & M32
    e = (9 * (c ^ dd) + C4) & M32
    return e ^ (e >> 16)


def section_random():
    hr("4a. Randomised check of the 10-op hash against problem.myhash")
    sys.path.insert(0, ROOT)
    from problem import myhash  # noqa: E402

    rng = random.Random(1)
    xs = [rng.getrandbits(32) for _ in range(200_000)] + [0, 1, M32, 0x80000000, 0x7FFFFFFF]
    bad = sum(1 for x in xs if hash10(x) ^ C5 != myhash(x))
    par = sum(1 for x in xs if (hash10(x) & 1) == (myhash(x) & 1))
    print(f"hash10(x) ^ C5 == myhash(x): {len(xs) - bad}/{len(xs)} agree")
    print(f"parity of lazy value equals parity of true value in {par} cases (expected 0: C5 is odd)")


def section_vload():
    hr("4c. Simulator check: one vload at the 1-based index puts the node in a fixed word")
    sys.path.insert(0, ROOT)
    from problem import DebugInfo, Input, Machine, Tree, build_mem_image  # noqa: E402

    random.seed(5)
    t = Tree.generate(10)
    mem = build_mem_image(t, Input.generate(t, 256, 16))
    base = mem[4]
    ok = True
    for idx in [0, 1, 30, 31, 254, 255, 1022, 1023, 2046] + [random.randrange(2047) for _ in range(40)]:
        m = Machine(mem, [{"load": [("const", 8, idx + 1)]}, {"load": [("vload", 16, 8)]}], DebugInfo(scratch_map={}))
        m.run()
        ok &= m.cores[0].scratch[16 + base - 1] == t.values[idx]
    print(f"forest_values_p = {base}: vload(dest, j) with j = idx+1 leaves tree[idx] in word {base - 1} of dest: {ok}")
    print("(upstream has forest_values_p = 7, so the word is 6; one load slot, no address add)")


# Checks that moved to verification/rewrites.py when the emission started to rely on them.
MOVED_TO_REWRITES = (
    "R8-hash-stage-merge",  # stages 2+3 in 3 ops (--merge-hash-stages)
)


def section_z3():
    hr("4b. Z3 bit-vector identities (UNSAT of the negation = proved for all 2^32 inputs)")
    import z3

    x, j, v, F = z3.BitVecs("x j v F", 32)
    K = lambda n: z3.BitVecVal(n & M32, 32)  # noqa: E731
    sh = lambda a, n: z3.LShR(a, n)  # noqa: E731

    def ref_stage(a, i):
        if i == 0:
            return (a + K(C0)) + (a << 12)
        if i == 1:
            return (a ^ K(C1)) ^ sh(a, 19)
        if i == 2:
            return (a + K(C2)) + (a << 5)
        if i == 3:
            return (a + K(C3)) ^ (a << 9)
        if i == 4:
            return (a + K(C4)) + (a << 3)
        return (a ^ K(C5)) ^ sh(a, 16)

    def ref_hash(a):
        for i in range(6):
            a = ref_stage(a, i)
        return a

    def lazy(a):
        a0 = K(4097) * a + K(C0)
        b = sh(a0, 19) ^ a0 ^ K(C1)
        c = K(33) * b + K(C2 + C3)
        d = K(16896) * b + K(C2 << 9)
        e = K(9) * (c ^ d) + K(C4)
        return e ^ sh(e, 16), c ^ d, e

    lz, st4, e = lazy(x)
    a, b, c, d, s, t = z3.BitVecs("a b c d s t", 32)
    # Identities the emission relies on live in verification/rewrites.py (one copy); run them
    # from there. The checks below are candidates and negative results that nothing emits yet.
    sys.path.insert(0, ROOT)
    from verification import rewrites  # noqa: E402

    for name, desc, fn in rewrites.REWRITE_CHECKS:
        if name in MOVED_TO_REWRITES:
            _, result, _, _ = fn()
            want = "sat" if name in rewrites.EXPECT_SAT else "unsat"
            print(f"  [{'PROVED' if result == want else 'NOT PROVED: ' + result}] {name}: {desc}  "
                  f"(verification/rewrites.py)", flush=True)

    checks = [
        # The whole-hash equivalence in one query times out (three nested multiplies), so it is
        # proved stage by stage; stages 1 and 5 are used unchanged, so composing is substitution.
        ("stage 0 == 4097a + C0  [1 multiply_add]", ref_stage(x, 0) != K(4097) * x + K(C0), []),
        ("stage 4 == 9a + C4  [1 multiply_add]", ref_stage(x, 4) != K(9) * x + K(C4), []),
        ("stage 5 == (a ^ (a>>16)) ^ C5  [constant can be applied later]",
         ref_stage(x, 5) != (x ^ sh(x, 16)) ^ K(C5), []),
        ("parity(reference hash) == 1 - parity(lazy hash)", (ref_hash(x) & 1) != 1 - (lz & 1), []),
        ("1-based index: j=idx+1, next j == 2j + bit  [2 ops: and, multiply_add]",
         (2 * (j - 1) + 1 + (v & 1)) + 1 != 2 * j + (v & 1), []),
        ("mirrored 1-based index m (j = c - m, c = 3*2^d - 1): next m == 2m + lazy bit, c' = 2c+1"
         "  [2 ops with the inverted parity of the lazy hash]",
         2 * (F - j) + (1 - (v & 1)) != (2 * F + 1) - (2 * j + (v & 1)), []),
        ("pointer form: p=idx+F, next p == 2p + bit + (1-F)  [3 ops, no address add]",
         (2 * (j - F) + 1 + (v & 1)) + F != 2 * j + (v & 1) + (1 - F), []),
        ("v | 0xFFFFFFFE == (v & 1) - 2  [bit plus constant in one op, only for -2]",
         (v | K(0xFFFFFFFE)) != (v & 1) - 2, []),
        ("arithmetic 2-way select: f + c*(t-f) == If(c==1,t,f) for c in {0,1}",
         b + s * (a - b) != z3.If(s == 1, a, b), [z3.ULE(s, 1)]),
        ("4-way select in 3 multiply_adds (Mobius form)",
         (a + s * (b - a)) + t * ((c - a) + s * (d - c - b + a))
         != z3.If(t == 1, z3.If(s == 1, d, c), z3.If(s == 1, b, a)), [z3.ULE(s, 1), z3.ULE(t, 1)]),
        ("early branch bit: bit16(65545*y + C4) == NOT bit0(lazy hash), y = stage-4 input",
         z3.Extract(16, 16, K(65545) * st4 + K(C4)) != ~z3.Extract(0, 0, lz), []),
    ]
    for name, neg, assume in checks:
        sv = z3.Solver()
        sv.set("timeout", 120_000)
        sv.add(*assume)
        sv.add(neg)
        r = sv.check()
        print(f"  [{'PROVED' if r == z3.unsat else 'NOT PROVED: ' + str(r)}] {name}", flush=True)

    # Negative results: rewrites that would save an op but are false. Each is refuted by
    # asking for a constant that works on just a few concrete inputs (UNSAT there implies
    # no constant works for all inputs).
    kk = z3.BitVec("kk", 32)
    pts = [0, 1, 2, 3, 4, 5, 6, 7, 0x80000000, M32]
    for cname, cval in (("C1", C1), ("C5", C5)):
        sv = z3.Solver()
        sv.add(*[(K(p) ^ K(cval)) == K(p) + kk for p in pts])
        print(f"  [{'NO SUCH CONSTANT' if sv.check() == z3.unsat else 'EXISTS?'}] "
              f"k with (a ^ {cname}) == a + k for all a  (would fold that xor into the preceding multiply_add)")
    m = z3.BitVec("m", 32)
    for base in (7, 8):
        for opname, f in (("&", lambda q: q & m), ("|", lambda q: q | m), ("^", lambda q: q ^ m),
                          ("+", lambda q: q + m), ("*", lambda q: q * m), ("%", lambda q: z3.URem(q, m))):
            sv = z3.Solver()
            sv.add(*[f(K(p)) == K((p & 1) + 1 - base) for p in pts])
            print(f"  [{'NO SUCH MASK' if sv.check() == z3.unsat else 'EXISTS?'}] "
                  f"m with (v {opname} m) == (v & 1) + 1 - {base} for all v  (2-op pointer update, tree base {base})")


# ----------------------------------------------------------------------------
# 5. Static tally of an external kernel's own op-count comments (never executed)
# ----------------------------------------------------------------------------
def section_external(path):
    hr(f"5. Static tally of {path} (regex over comments; file is not imported or run)")
    pat = re.compile(r"\s*# (.*): (\d+) (ALU|VALU|LOAD|STORE|FLOW) computation")
    put = re.compile(r"emit\.put\([^,]+, '(\w+)', '([^']+)'")
    fn = re.compile(r"^def (_emit_\w+)")
    lines = open(path, encoding="utf-8").read().split("\n")
    cur = None
    eng = collections.Counter()
    by = collections.Counter()
    for i, line in enumerate(lines):
        m = fn.match(line)
        if m:
            cur = m.group(1)
        m = pat.match(line)
        if m:
            op = next((put.search(lines[k]).group(2) for k in range(i + 1, min(i + 6, len(lines)))
                       if put.search(lines[k])), "?")
            eng[m.group(3)] += int(m.group(2))
            by[(cur, m.group(1), m.group(3), op)] += int(m.group(2))
    m = re.search(r"range\((\d+)\)\]", "\n".join(lines[:60]))
    cycles = int(m.group(1)) if m else None
    print(f"declared cycles: {cycles}; ops by engine: {dict(eng)}; total {sum(eng.values())}")
    veq = eng["VALU"] + eng["ALU"] / 8
    print(f"vector-op equivalents: {eng['VALU']} + {eng['ALU']}/8 = {veq:.2f} -> {veq / VEQ_PER_CYCLE:.1f} cycles")
    print(f"load floor {math.ceil(eng['LOAD'] / 2)}, flow floor {eng['FLOW']}, store floor {math.ceil(eng['STORE'] / 2)}")
    if cycles:
        for e, capn in (("VALU", 6), ("ALU", 12), ("LOAD", 2), ("STORE", 2), ("FLOW", 1)):
            print(f"  {e:5s} utilisation {eng[e] / (capn * cycles):.1%}")
    grp = collections.Counter()
    for (f, name, e, op), n in by.items():
        grp[(f, e)] += n / (8 if e == "ALU" else 1) if e in ("ALU", "VALU") else 0
    sec = collections.Counter()
    for (f, e), n in grp.items():
        sec[f] += n
    print("vector-op equivalents by emitter section:")
    for f, n in sec.items():
        print(f"  {f:28s} {n:8.1f}  ({n / 512:.2f} per group-round)")
    print("largest items (section, name, engine, opcode, count):")
    for k, n in sorted(by.items(), key=lambda kv: -kv[1])[:45]:
        print(f"  {n:5d}  {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(ROOT, "scheduler_output_g16.json"))
    ap.add_argument("--z3", action="store_true", help="run the Z3 identities (needs the 'verify' extra)")
    ap.add_argument("--external-kernel", help="statically tally op-count comments of another kernel source")
    args = ap.parse_args()
    d, ssa_by_id = load_schedule(args.json)
    ops = section_bound(d, ssa_by_id)
    section_budget(ops)
    section_designs()
    section_random()
    section_vload()
    if args.z3:
        section_z3()
    if args.external_kernel:
        section_external(args.external_kernel)


if __name__ == "__main__":
    main()
