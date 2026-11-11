# Why the leaderboard (863) is far below our bound (1,271)

November 2026. Numbers marked **V** were computed or checked in this session, **I** are
inferences from verified facts, **S** are speculation. Everything marked V that comes from
this repo is reproduced by `scripts/analyze_bound_gap.py` (section 7).

## 1. Short answer

**The 1,271 bound is computed correctly, but it is a bound on scheduling the ops we emit,
not on the problem.** It says nothing about kernels that emit different ops, and the top
kernels emit far fewer.

| | Vector-op equivalents | Loads | Capacity floor | Achieved |
|---|---|---|---|---|
| Our 16-round emission | 9,526 | 2,136 | 1,271 | 1,994 to 2,007 |
| Public 880-cycle kernel (Sakauma, static tally) | 6,541 | 1,724 | 873 | 880 |
| Budget of an 863-cycle kernel | at most 6,472 | at most 1,726 | | 863 |

Three assumptions are baked into our op set, and the public 880-cycle kernel breaks all three:

1. **One scalar `load` per lane per gathered level.** A `vload` takes one load slot, reads
   8 consecutive memory words, and its address can be data-dependent. The 880 kernel
   rewrites tree levels 3 to 7 in memory as 3-node and 7-node subtree packets (stores are
   nearly free), so one load slot per lane serves 2 or 3 rounds. It visits levels 3 to 10
   and 3 to 4 again (10 level-visits) with 6 loads per lane. We spend 8 loads per lane on
   8 gathered rounds and would need 10 to cover the same levels.
2. **Node selection and address arithmetic must be paid for on the ALUs.** In the 880
   kernel all work outside the hash costs 2.58 vector-op equivalents per group-round. Ours
   costs 7.39. With `vload`, selection moves into the load address and into which loaded
   word is read, and the `+ forest_values_p` add disappears.
3. **An 11-op hash.** Stages 2 and 3 merge into three ops, giving 10. Proved here with Z3.

Within assumption 1 (scalar gathers only) the best design I can construct has a capacity
floor of about 945 cycles, and no such design can reach 863: the fixed work plus six
unavoidable gather rounds already needs 913 cycles with selection free of charge (section
5). So 863 is not reachable by tidying our current emission. It requires the packet/`vload`
approach or something equivalent.

What I could not determine: what the 863 to 866 entries do beyond the 880 kernel. No
write-up exists. The 880 kernel's floor is 873, so 863 needs about 70 fewer vector-op
equivalents (1%) on the same design, which is plausible but not shown.

## 2. Is the bound right for our op set?

**Yes (V).** Recomputed independently from `scheduler_output_g16.json`:

| Quantity | Value |
|---|---|
| Vector ALU ops | 8,783 |
| Vector ops executed as 8 scalar ops | 723 |
| Native scalar ALU ops | 162 |
| Load / flow / store | 2,136 / 488 / 64 |
| Vector-op equivalents | 9,506 + 162/8 = 9,526.25 |
| Continuous joint floor | 9,526.25 / 7.5 = 1,270.17 |
| Integer bound, min over shift, const and select choices | **1,271** (1,886 ops shifted) |
| Per-engine terms at the optimum | valu 1,270, alu 1,271, load 1,065 to 1,088, flow 448, store 32 |
| Critical path (+1 per SSA edge) | 264 |
| Peak live scratch | 1,268 of 1,536 |

Reading of the code (`scheduler_rs/src/cpsat_scheduler/bounds.rs`,
`compute_capacity_lower_bound_global`, and `model.rs` section 9 onward):

| Item | Finding | Status |
|---|---|---|
| Global bound | `min over (x,y,z) of max(ceil(valu/6), ceil(alu/12), ceil(load/2), flow, ceil(store/2))`. A valid lower bound for the given ops. My independent recomputation gives the same 1,271. | V |
| BUGS.md "ALU direct capacity constraint wrong" | **Stale.** At HEAD the constraint is `12*makespan - 8*s >= alu_count`, which is the corrected form the entry asks for. Only the comment two blocks above it (`ALU ops = alu_count + 8*(shiftable_count - s)`) still describes the old formula. The entry should move to Fixed. | V (read at HEAD with `git show`) |
| `capacity_valu` | `6*makespan + s - select_contrib >= valu_count`: consistent with the cumulative. | V (by reading) |
| Looseness | The bound ignores precedence (dependency floor is passed as 0 on purpose), scratch, and ramp-up and tail effects. Critical path is 264, so precedence alone is far from binding. The 720-cycle gap between 1,271 and the 1,994 schedule is schedule structure (rounds run back to back), as the pickup assessment says. | V for the numbers, I for the attribution |
| Other engines | Load floor 1,065 to 1,088, flow 448, store 32: none binds today. Load becomes the binding engine as soon as ALU work drops below about 8,000 equivalents. | V |
| `n_nodes=1023` in `run_options` | The CLI derived `n_nodes = 2^height - 1` (fixed since: 2^(height+1) - 1 = 2,047, as the harness passes). Emission keys off `forest_height`, so the JSON still shows 11 levels and 8 gather rounds, and the counts above are unaffected. | V (round/level table below) |

So the bound is not wrong. It is conditional, and the condition is where the gap lives.

## 3. Where our ops go

Vector-op equivalents (vector ALU ops plus scalar ALU ops / 8), from the JSON (V).

| Rounds | Level | Kind | xor | hash | branch | tree access | Per group | Loads | Flow |
|---|---|---|---|---|---|---|---|---|---|
| 0, 11 | 0 | root | 1 | 11 | 3 | 2 | 17.00 | 0 | 0 |
| 1, 12 | 1 | select | 1 | 11 | 3 | 3 | 18.00 | 0 | 32 each |
| 2, 13 | 2 | select | 1 | 11 | 3 | 5 | 20.00 | 0 | 96 each |
| 3, 14 | 3 | select | 1 | 11 | 3 | 10.25 / 10 | 25.25 / 25 | 0 | 96 each |
| 4 to 9, 15 | 4 to 9, 4 | gather | 1 | 11 | 3 | 2 | 17.00 | 256 each | 0 |
| 10 | 10 | gather, wrap | 1 | 11 | 1 | 2 | 15.00 | 256 | 0 |
| Init, load, store | | | | | | | 109 total | 88 | 40 |
| **Total** | | | 512 | 5,632 | 1,472 | 1,801 | **9,526** | **2,136** | **488** |

Average 18.39 per group-round. Budgets:

| Target | Vector-op equivalents | Per group-round | Loads | Our excess (equivalents / loads) |
|---|---|---|---|---|
| 1,100 | 8,250 | 16.11 | 2,200 | 1,276 / none |
| 1,000 | 7,500 | 14.65 | 2,000 | 2,026 / 136 |
| 863 | 6,472 | 12.64 | 1,726 | 3,054 / 410 |

Side by side with the 880 kernel (its counts are a static tally of the author's per-item
comments; they sum to the 19,815 the file asserts; I did not execute the file):

| Phase | Ours | 880 kernel | Difference |
|---|---|---|---|
| Hash | 5,632 (11 per group-round) | 5,121 (10.00) | 511 |
| Node xor, branch bit, index, address, selection | 3,785 (7.39) | 1,322 (2.58) | 2,463 |
| Setup, packets, output | 109 | 98 | 11 |
| **Total** | **9,526** | **6,541** | **2,985** |
| Loads | 2,136 | 1,724 | 412 |
| Flow | 488 | 843 | |
| Stores | 64 | 1,591 | |
| Engine use at its cycle count | | valu 99.2%, alu 98.7%, load 98.0%, flow 95.8%, store 90.4% | |

The 880 kernel is saturated on four engines at once. Its floors are 873 (ALUs), 862
(load) and 843 (flow).

## 4. Candidate reductions

Savings are against our 9,526 equivalents; cycles are savings / 7.5. "Public" means found
in a write-up or public source; the source is named.

### 4.1 Within the current structure (scalar gathers)

| Change | Saving (equivalents) | Bound cycles | Source | Check |
|---|---|---|---|---|
| Merge hash stages 2 and 3: `(33a+C2+C3) ^ (16896a+(C2<<9))` | 512 | 68 | Public (Battistela write-up; 880 kernel) | **Z3 proved** |
| Index update in 2 ops instead of 3: 1-based index `j' = 2j + bit`; with the deferred stage-5 constant the parity is inverted, so track the mirrored index `m' = 2m + lazy_bit` | 480 | 64 | Public (975 repo: "one-based and mirrored indices") | **Z3 proved** (both) |
| No index chain where nothing reads it: rounds 0, 10, 11, 15 and the step into round 11 | 160 | 21 | Public (skip last-round update); the rest I | Arithmetic; dataflow checker confirms dead ops |
| Tree access on cached levels: no per-group root broadcast, condition is the branch bit itself, stage-5 constant folded into the node constants, 2^d - 1 selects per level, all on flow | 1,257 | 168 | Public (several) | Arithmetic: 1,801 becomes 8 x 64 gather ops + 32; flow rises from 448 to 704 |
| Gather rounds stay at 2 (address add, stage-5 xor) | 0 | 0 | | See 4.3 for why neither folds away |
| Final xor and output unchanged; index `vload`/`vstore` and their pointer adds dropped | 49, and 48 loads | 7 | Public (values only) | `tests/submission_tests.py` compares values only |
| **Sum (design B0 in section 5)** | **2,458** | | | Floor becomes load-bound at 1,044 |

After these the ALUs would allow about 942 but 2,088 loads force 1,044. Loads are then the
problem.

### 4.2 Fewer loads

| Change | Effect | Source | Status |
|---|---|---|---|
| Cache level 4 too (16 nodes, 15 selects per group per visit) | Loads 1,576, but 1,664 selects exceed the flow slot; overflow goes to the ALUs. Floor 1,003 | Public (Vogel thread, 975 repo "cached depth-4 lookups") | V (model) |
| Cache level 4 only where it pays (round 4 gathered, round 15 mostly selected) | Floor 945 | Public in spirit (975 repo searches cache sites) | V (model, exhaustive over group counts) |
| "Load each distinct node once" for levels with fewer nodes than lanes | This is caching. It cuts loads to 2^d (or 2^d/8 with `vload`) but needs 2^d - 1 selects per group: 31, 63, 127 for levels 5 to 7. Not viable beyond level 4 with two-way selects | Public claim ("839 loads") | I. 839 = 768 + 32 + 32 + 7 fits "gather levels 8 to 10, `vload` 255 nodes and the inputs, 7 header words". I found no kernel that does this |
| Arithmetic select `f + c*(t-f)`, 4-way select in 3 `multiply_add`s | Same op count as a select tree (2^d - 1); only moves work from flow to the vector ALU | Public (975 repo wiki) | **Z3 proved**; the count argument is I |
| Select on the scalar ALU | 8 scalar ops per vector select. Same price as any shifted op | | V (arithmetic) |
| **`vload` with a data-dependent address** | One load slot returns 8 consecutive words. See 4.4 | Public (880 kernel source) | V on the simulator for the basic form |

### 4.3 Things I checked that do not work

| Idea | Result | Check |
|---|---|---|
| Fold the stage-1 or stage-5 xor constant into the preceding `multiply_add` | No constant k has `(a ^ C) == a + k` | **Z3**: no such k for C1 or C5 |
| Pointer-form index with a 2-op update on the in-place tree: needs one op giving `(v & 1) + 1 - base` | No mask works for `& \| ^ + * %` with tree base 7 (upstream) or 8 (this repo). `v \| 0xFFFFFFFE = (v&1) - 2` exists but needs base 3 | **Z3**: no such mask; the -2 identity proved |
| Pointer form in general | Proved, but it is 3 ops (bit, `multiply_add`, constant) against 2 + 1 address add. No saving for scalar gathers | **Z3 proved**; count is arithmetic |
| A hash below 10 ops | I found no identity. Stage 1 needs shift, xor, xor; stages 2+3 need two multiplies and an xor; stage 5 needs shift and xor. The 880 kernel also uses exactly 10.00 per group-round | I. A superoptimiser search over 2-op programs for stage 1 would settle it |
| Combining adjacent stages other than 2+3 | Stages 0+1 and 4+5 are separated by a right shift and an xor; no fold | I |
| Sharing work across lanes or groups | Only the level-0 node is common (rounds 0 and 11). Values differ per lane from the first hash on. We currently broadcast the root once per group (64 ops); it should be once | V for our waste; I for "nothing else" |
| `load_offset` | Same as `load` with lane addressing. No address arithmetic | V (read `problem.py`) |
| Unused scalar ops (`%`, `//`, `cdiv`, `<`, `==`) | None changes the arithmetic. `v % 2` is the same price as `v & 1` | I |
| Critical path as the binding limit | 9 dependent levels per round (xor + 8 hash levels), about 150 cycles in all | V (ours is 264 with the 11-op hash and gathers) |

### 4.4 What the 880 kernel does (public source, read not run)

Repo `Sakauma/my_original_performance_takehome`, 880 cycles, README states 9/9 official
tests at upstream commit `5452f74`. Mechanisms, from its docstrings and op tally:

| Mechanism | Evidence | Status |
|---|---|---|
| Levels 0 to 2 cached, selected with flow `vselect` (8 per group in total) | "cache select": 256 flow ops | V (tally) |
| Levels 3+4 as 8 packets of 3 words, levels 5 to 7 as 32 packets of 7 words, written over consumed header and tree memory with 31 `vload` + 31 `vstore`, pre-xored | `_emit_tree_packets` docstring; 248 scalar xors | V (tally); behaviour not run |
| One `vload` per lane per packet: 3 packet loads per lane (levels 3+4, 5 to 7, 3+4 again) | "packet": 768 `vload` | V (tally) |
| Levels 8 to 10 read in place with one `vload` per lane per level at the 1-based index; the wanted node is word 6, so there is no address add and the index update is 2 ops | "deep packet": 768 `vload`; docstring "VLOAD pointer is 511-q8; the desired tree word is lane 6" | **V on our simulator**: `vload(dest, idx+1)` leaves `tree[idx]` in word `forest_values_p - 1` |
| Node xor done as scalar ops, which also moves the word from the lane's load window into the group vector | "deep xor" 768, "packet xor" 984 scalar ops | V (tally) |
| Idle store slots plus `vload` used to assemble selected lane values through memory | "packet": 1,448 scalar stores | V (tally); mechanism I |
| Constants broadcast by `vload` from consumed memory and by all-false `vselect`, not by vector ALU | setup docstring | V (read) |
| Branch bit one op early: bit 16 of `65545*y + C4` is the complement of bit 0 of the lazy hash | hash docstring | **Z3 proved** (latency trick, no op saving) |
| Final indices not written; initial indices assumed 0; tree and index memory overwritten | header docstring | V (read). Allowed by both harnesses, see section 6 |

Loads: 1,536 per-lane packet loads + 32 inputs + 31 tree blocks + 31 bank reads + 94
others (constants, header, a few extra node loads) = 1,724 (V, tally).

## 5. Composed designs

Capacity floors from an op model (`design()` in the script). Each two-way select may use the
flow slot or one vector op. Overheads: 60 equivalents, 40 loads, 10 flow. These are floors,
not schedules; the 880 kernel lands 7 cycles above its floor, the public 975 kernel 8
above its vector-ALU floor of 967.

| Design | Equivalents excl. selects | Selects | Loads | Load floor | **Capacity floor** |
|---|---|---|---|---|---|
| A. Today's emission | 9,526 | (448 on flow) | 2,136 | 1,068 | **1,271** |
| B0. Today's structure with 4.1 applied (cache 0 to 3, gather 4 to 10 and round 15) | 7,068 | 704 | 2,088 | 1,044 | **1,044** |
| B1. B0, but round 15 selects among 16 cached nodes | 6,908 | 1,184 | 1,832 | 916 | **954** |
| C0. Cache 0 to 4, gather 5 to 10 | 6,844 | 1,664 | 1,576 | 788 | **1,003** |
| C1. Best mix: level 4 gathered in round 4, and by 7 of 32 groups in round 15 | 6,943 | 1,079 | 1,888 | 944 | **945** |
| C2. C0 with selection free (not achievable) | 6,844 | 0 | 1,576 | 788 | **913** |
| C3. C0 with a hypothetical 9-op hash | 6,332 | 1,664 | 1,576 | 788 | **942** |
| D. Packets and per-lane `vload` (the 880 kernel, measured) | 6,541 | (843 flow) | 1,724 | 862 | **873** |

Which design reaches which target:

| Target | Reachable by | Margin needed |
|---|---|---|
| 1,100 | B0 (floor 1,044). Needs a schedule within 5% of the floor, so round barriers must go | Public kernels of this shape reach 1,023 to 1,160 |
| 1,000 | B1 or C1 (floors 954, 945). Needs a schedule within 5% | The public 975 kernel is this family: 5,800 vector ops, floor 967 |
| 863 | Only D, and only with about 70 fewer equivalents than the 880 kernel (6,472 against 6,541) at the same load count | Not demonstrated publicly |

Why no scalar-gather design reaches 863 (V, arithmetic): levels 5 to 10 have 32 to 1,024
nodes and must be gathered, so there are at least 6 gather rounds. Per group that is
16 x 11 (xor and hash) + 1 (final constant) + 14 (branch bits) + 9 (index chain) + 12
(address add and stage-5 xor) = 212 ops, or 6,784 + 60 overhead. That alone is 913 cycles
before a single select is paid for. Getting under 863 means removing about 12 ops per
group from the index, address and constant work, which is what `vload` at the 1-based
index and pre-xored packets do.

What remains unexplained: the last 2% from D's floor of 873 to 863. Candidates (S): fewer
scalar xors by covering more level-visits with packets; a cheaper bit for packet arms; a
tighter setup. Each can be tested by counting ops in a modified emission and recomputing
the floor.

## 6. Is 863 comparable with our numbers?

Leaderboard rules as reported by its about page (fetched through a summarising tool, so
treat wording as paraphrase): submit a complete `perf_takehome.py` (at most 1,000 KB); the
submitted Python runs only to generate instructions, which are checked against an engine
whitelist and simulated by trusted code; shape is height 10, 16 rounds, batch 256; nine
fresh random seeds; score is the worst cycle count; output values required, indices
optional; harness exploits rejected.

| Point | Leaderboard | This repo | Effect |
|---|---|---|---|
| What is compared | Values only | Values only (`tests/submission_tests.py`) | Same |
| Problem shape | 10 / 16 / 256 | Same | Same |
| Simulator | Upstream, run server-side | `frozen_problem.py` re-exports the edited `problem.py`: header 8 words instead of 7, `forest_values_p` 8 instead of 7, rewritten `reference_kernel2` | **Not the same oracle.** A kernel that hardcodes pointers for one layout fails on the other. Slot limits and op semantics are unchanged, so cycle counts of correct straight-line kernels are comparable |
| Seeds | Nine fresh seeds, worst cycle count | 8 unseeded runs | A straight-line kernel has one cycle count for all inputs, so "worst of nine" only matters for correctness |
| Initial indices | Always 0 from `Input.generate` | Same | Kernels that assume root start are valid in both |
| Clobbering tree and index memory | Not checked | Not checked | Valid in both. A stricter reading ("leave inputs intact") would exclude the 880 kernel's packet rewrite; neither harness takes that reading |
| Precomputed answers | Impossible with fresh seeds | | 863 cannot be a lookup of results (I) |
| Submission form | Self-contained Python | Needs the Rust extension | We would have to embed the instruction list |

I found nothing that makes 863 invalid by our own test. The only non-comparability is on
our side: our oracle is an edited simulator.

## 7. Reproduction

```bash
# Sections 2, 3, 5 and the randomised hash check
uv run python scripts/analyze_bound_gap.py

# Plus the Z3 identities
uv run --extra verify python scripts/analyze_bound_gap.py --z3

# Static tally of the public 880 kernel (the file is read with a regex, never executed)
gh api repos/Sakauma/my_original_performance_takehome/contents/perf_takehome.py \
  -H 'Accept: application/vnd.github.raw' > /tmp/sakauma_perf_takehome.txt
uv run python scripts/analyze_bound_gap.py --external-kernel /tmp/sakauma_perf_takehome.txt
```

Checks done, by kind:

| Kind | What |
|---|---|
| Z3, all 2^32 inputs | Stage 0, 2+3, 4, 5 rewrites; parity inversion; 1-based and mirrored index; pointer form; `v \| ~1`; 2-way and 4-way arithmetic select; early branch bit; non-existence of the folds in 4.3 |
| Randomised, 200,005 inputs against `problem.myhash` | The 10-op hash end to end (the single whole-hash Z3 query timed out at 120 s, so it is proved stage by stage instead) |
| Simulator (`problem.Machine`) | `vload` at the 1-based index returns the node in a fixed word |
| Arithmetic from the JSON | Section 2 and 3 |
| Static read of third-party source | Section 4.4 and design D. Not executed here; to confirm 880, follow that repo's README against upstream |

## 8. Sources

- Leaderboard and rules: <https://www.paradigm.xyz/puzzles/anthropic-challenge>,
  <https://www.paradigm.xyz/puzzles/anthropic-challenge/about>
- 880 cycles, source with op tallies: <https://github.com/Sakauma/my_original_performance_takehome>
- 975 cycles, technique wiki: <https://github.com/Trav55555/original_performance_takehome>
- 1,158 cycles: <https://lorenzobattistela.github.io/posts/beating-claude>
- 1,218 cycles: <https://sambhavg.github.io/blog/performance>
