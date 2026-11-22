# Scratch allocation conjectures

**The per-cycle scratch check is not enough.** The schedulers guarantee that the live words at every cycle fit the budget, but that does not guarantee the address allocator can lay the values out. A 16-word instance with scalars and 8-word vectors is a counterexample, and it stays one for every budget from 16 up, including 1,536. This holds whether or not vectors are 8-aligned, and under our real same-cycle-reuse liveness rules. No polynomial allocator can therefore always succeed after the per-cycle check. Deciding whether a layout exists is NP-hard (Dynamic Storage Allocation, Garey and Johnson SR2). What the proof does give: a two-pass allocator always succeeds when `8 * max_v + max_s <= B` (C3 below). Every claim on this page has a check in [`verification/allocation_conjectures.py`](../verification/allocation_conjectures.py); the counterexamples are data in [`verification/allocation_counterexamples.json`](../verification/allocation_counterexamples.json), ready for allocator unit tests.

## Model

After scheduling, each value has a fixed live interval and a size: 1 word for a scalar, 8 for a vector or vector group. [`address_assign.rs`](../scheduler_rs/src/address_assign.rs) gives each value one contiguous block for its whole interval. Two values live at the same time must not share a word.

**Abstract model** (used by all checks):
- A value is a closed interval `[start, end]` of steps and a size.
- Two values *overlap* when their intervals share a step.
- A *layout* gives each value an address `0 <= a <= B - size` such that overlapping values get disjoint blocks.
- An optional alignment rule (`a % align == offset`) applies to values wider than one word.
- The *load* at a step is the total size of the values live at that step.
- The *per-cycle check* is `load <= B` at every step.

This is Dynamic Storage Allocation (DSA): rectangles with fixed time extent, packed in the address dimension by sliding them vertically.

**Real semantics.** The allocator uses closed `[def, last_read]` intervals. A value whose last read is at cycle C may share words with one written at C, because reads happen before writes within a cycle. The exception is a point interval `[C, C]`, which shares with nothing live at C. This is the abstract model on a doubled time line: phase 2C reads and phase 2C+1 writes.
- A non-point `[s, e]` occupies phases `[2s+1, 2e]`.
- A point `[C, C]` occupies phases `[2C, 2C+1]`.

The checks prove:
- `A3-real-conflict-is-overlap`: the expiry rule (in Python, cross-checked against a port of the Rust rule in `tests/test_allocation_conjectures.py`) forbids sharing exactly when the doubled intervals overlap.
- `A3-real-load-bound`: a closed per-cycle count bounds the doubled-time load. This is the count the CP-SAT scratch cumulative uses, `[def, last_read + 1)` half-open.
- `A3-real-embedding`: mapping abstract `[s, e]` to real `[2s, 2e+1]` keeps both conflicts and per-cycle loads.

So every abstract result below holds for the real allocator. Every abstract counterexample becomes a real one with cycles doubled.

**Alignment.** Our allocator does not need vector alignment. `address_assign.rs` places vectors at any address, and the ISA only requires a vload/vstore block to be contiguous ([ISA_REFERENCE.md](ISA_REFERENCE.md), `problem.py`). Aligned variants are reported because an allocator might choose 8-aligned slots. A reserved address 0 is the same problem as `B - 1` words starting at 0, with every address shifted by one; slots then sit at 1, 9, 17, .... The `offset` of `Alignment` only sets the slot phase. It does not reserve word 0, so scalars may still use word 0.

Every check below follows the repo convention ([verification/README.md](../verification/README.md)). A soundness check encodes a violation, so **UNSAT means the statement holds**. A witness check wants **SAT**.

## C1: does the per-cycle check suffice? No.

**Statement.** For sizes {1, k}, if the load is at most B at every step, a layout exists.

**Refuted** for every k >= 2:
- with vectors at any offset, for every B >= 2k;
- with k-aligned vectors, for every multiple of k from 2k;
- under the real semantics.

**Construction** (`pinning_counterexample`). Unaligned, B = 2k, k = 8:

| step | live | load |
|---|---|---|
| 0 | vectors V1 `[0,0]`, V2 `[0,1]` | 16 |
| 1 | V2; scalars: 7 x `[1,2]`, pin `[1,3]` | 16 |
| 2 | 7 x `[1,2]`, pin, 1 x `[2,3]` | 9 |
| 3 | pin, `[2,3]`; vector V3 `[3,4]` | 10 |
| 4 | V3, V4 `[4,4]` | 16 |

Why no layout exists:
1. At steps 0 and 4, two vectors fill all 16 words, so V2 and V3 each sit at address 0 or 8.
2. So every scalar live at step 1 is in the 8-word half that V2 leaves free. Likewise, every scalar live at step 3 is in the half that V3 leaves free.
3. The pin is live at steps 1 and 3 at one fixed address, so those two halves are the same half.
4. At step 2 all 9 scalars must then fit in that one half of 8 words: contradiction.

With 8-aligned slots the slots are the halves, so V1 and V4 are not needed: 3 steps, 2 vectors, 9 scalars.

**Larger budgets.** For B > 2k, add B - 2k filler scalars that are live for the whole horizon. Take any layout of the larger instance and delete the filler words while keeping the order of the remaining words. Every other block stays contiguous, so the result is a layout of the 2k-word instance, which does not exist. For aligned budgets that are multiples of k, use whole filler vectors instead, which keeps alignment. The same argument works for sizes {a, b} whenever b >= 2a, so it also covers sizes that do not divide each other, such as {2, 5}.

**Checks.** Rerun with `uv run python -m verification.allocation_conjectures`, part of `./scripts/test.sh proofs`.

| Check | Result | Meaning |
|---|---|---|
| `A1-c1-refuted-k8-B16-unaligned` | UNSAT | the table above: per-cycle check passes, Z3 proves no layout |
| `A1-c1-refuted-k8-B16-aligned` | UNSAT | the 3-step aligned version |
| `A1-c1-refuted-k8-B20-unaligned`, `-k8-B24-aligned` | UNSAT | the filler argument, checked directly |
| `A1-c1-refuted-k8-B16-unaligned-real` | UNSAT | the doubled-time real-semantics instance |
| `A1-c1-refuted-k{2,3,4}-B{2k}-unaligned`, `-s2k5-B10-unaligned` | UNSAT | other k; sizes {2, 5} |
| `A1-stored` | UNSAT | every entry in `c1_counterexamples` of the JSON |
| `A2-c1-tight-k8-B17-{unaligned,aligned}` | SAT | one more word suffices for both witnesses |

The general statement (every k >= 2 and every B) rests on the hand proof above. The checks confirm it for the listed k and B.

### Bounded search: where C1 holds and how small counterexamples are

`exhaustive_c1` enumerates every load-maximal instance. Within a bound of T steps and sizes S, that means every multiset of values with load <= B to which no value can be added. Adding values never creates a layout, so this decides C1 over **all instances with at most T steps, with any number of values**. Each counterexample found is minimised and re-proved by Z3. `cegis_c1` decides the same question by CEGIS: Z3 proposes instance counts, a layout search refutes them, and the saturated layout and its time reversal become lemmas. It is a cross-check.

Its lemmas generalise poorly here: each learned layout covers about one maximal instance, so it does about as much work as the enumeration. The exhaustive search is the workhorse.

| Check | Sizes | B | Steps <= | Alignment | Result | Instances / iterations | Time | Suite |
|---|---|---|---|---|---|---|---|---|
| `A6-c1-bounded-sizes1-2-B4-T4-unaligned` | {1,2} | 4 | 4 | any | holds | 3,335 | 0.4 s | fast |
| `A6-c1-cegis-sizes1-2-B4-T4-unaligned` | {1,2} | 4 | 4 | any | holds | 1,759 it. | 6 s | `--slow` |
| `A6-c1-bounded-sizes1-2-B4-T5-unaligned` | {1,2} | 4 | 5 | any | **fails** (the pinning shape) | 35,838 | 8 s | `--slow` |
| `A6-c1-bounded-sizes1-2-B4-T2-aligned` (and `-cegis-`) | {1,2} | 4 | 2 | 2-aligned | holds | 26 / 21 it. | <0.1 s | fast |
| `A6-c1-bounded-sizes1-2-B4-T3-aligned` (and `-cegis-`) | {1,2} | 4 | 3 | 2-aligned | **fails** | 224 / 79 it. | <0.2 s | fast |
| `A6-c1-bounded-sizes1-2-B3-T6-unaligned` | {1,2} | 3 | 6 | any | holds | 30,870 | 13 s | `--slow` |
| `A6-c1-bounded-sizes1-3-B4-T5-unaligned` | {1,3} | 4 | 5 | any | holds | 8,578 | 3 s | `--slow` |
| `A6-c1-bounded-sizes1-3-B5-T5-unaligned` | {1,3} | 5 | 5 | any | holds | 46,035 | 28 s | `--slow` |
| `A6-c1-bounded-sizes1-3-B6-T4-unaligned` | {1,3} | 6 | 4 | any | holds | 10,209 | 3 s | `--slow` |
| `A6-c1-bounded-sizes1-4-B8-T4-unaligned` | {1,4} | 8 | 4 | any | holds | 27,781 | 17 s | `--slow` |
| `A6-c1-bounded-sizes1-4-B8-T3-aligned` | {1,4} | 8 | 3 | 4-aligned | **fails** | 488 | 0.1 s | `--slow` |
| `A6-c1-bounded-sizes2-3-B6-T6-unaligned` | {2,3} | 6 | 6 | any | holds | 84,941 | 20 s | `--slow` |
| `A6-c1-bounded-sizes2-3-B7-T5-unaligned` | {2,3} | 7 | 5 | any | **fails** | 25,223 | 4 s | `--slow` |
| `A6-c1-bounded-sizes1-2-B4-T5-unaligned-slack1` | {1,2} | 4 (+1 word) | 5 | any | holds: one extra word always suffices | 41,734 | 41 s | `--slow` |
| `A6-c1-bounded-sizes1-3-B6-T5-unaligned-slack1` | {1,3} | 6 (+1 word) | 5 | any | holds: one extra word always suffices | 232,769 | 168 s | `--slow` |

Run the slow rows with `uv run python -m verification.allocation_conjectures --slow`. The whole slow run takes about 5 minutes. The times above are from one run on an M-series Mac.

What the bounded results say:
- **The pinning witnesses are the shortest possible** in the bounds checked. Unaligned, no counterexample has 4 steps or fewer ({1,2} at B=4, {1,3} at B=6, {1,4} at B=8), and the witness has 5. Aligned, none has 2 steps, and the witness has 3.
- **Two steps never fail, for any k and B, aligned or not** (hand proof). Write `B = m*k + r` (m slots, r < k spare words). Lay out the values live at both steps first: their v01 vectors in slots 0 .. v01-1, their s01 scalars at the top. Those scalars cover the r spare words first, then `ceil(max(0, s01 - r) / k)` slots. At step 0, the load bound `k*(v01 + v0) + s01 + s0 <= m*k + r` gives `v0 <= m - v01 - (s01 - r)/k`. Since v0 is an integer, v0 is at most the number of slots still whole, and the step-0 scalars fit in the words left. The same holds for step 1, independently. An aligned layout is also an unaligned one.
- **Below B = 2k, C1 held everywhere we looked** (at most one vector is live at a time): {1,2} at B=3 up to 6 steps, {1,3} at B=4 and 5 up to 5 steps. This is bounded evidence only. It is irrelevant at B = 1536.
- **The true theorem is about equal sizes.** The per-cycle check suffices for every instance only when all values have the same size. For equal sizes, start-order first fit is optimal on interval graphs: see A4 and the C3 proof with no scalars. Two different sizes fail: by the pinning construction when b >= 2a ({1, k}, {2, 5}, ...), and by exhaustive search for {2, 3} (B = 7; the stored `sizes-2-3-B7`). Three sizes such as {1, 2, 3} contain {1, 2}, so they fail too. We have not proved the pair case a < b < 2a in general.
- **Gap.** Every counterexample found needs exactly one extra word. One extra word always sufficed in the bounds checked: {1,2} at B = 4 and {1,3} at B = 6, up to 5 steps each. We have no general additive bound for {1, 8} (see Literature).

## C2: is there a polynomial allocator that always succeeds? No.

- **Under the per-cycle check:** no allocator of any kind always succeeds, because by C1 a layout may not exist.
- **Whenever a layout exists:** this is DSA, NP-complete in the strong sense even with sizes {1, 2} (SR2). It stays NP-complete with aligned register pairs on straight-line code (Lee, Palsberg and Pereira). So there is no polynomial complete allocator unless P = NP.
- Each candidate greedy fails on a small instance that has a layout.

Each instance is stored in `greedy_failures` of the JSON with a one-line explanation, and `A7-greedy-witnesses` re-checks them all: the per-cycle check passes, the greedy fails, and Z3 finds a layout. They are 1-minimal: dropping any value makes the greedy succeed or the layout disappear. They were found by a seeded search over k = 8 and B from 8 upward (`--search-greedy` reproduces them):

| Allocator (`GREEDY_ALLOCATORS`) | Unaligned | 8-aligned | B >= 16 example |
|---|---|---|---|
| `start_order_lowest` (first fit) | B=9, 3 values | B=9, 2 values | B=16, 4 values: a scalar reuses a dead vector's word 0 and splits the space 7 + 7 |
| `start_order_most_recently_freed` | B=9 (same as first fit) | B=9 | B=16, 4 values |
| `longest_lifetime_first` (vectors bottom-up, scalars top-down) | B=9, 3 values | B=9 | B=16, 10 values (unaligned), 4 (aligned) |
| `vectors_then_scalars` (C3's allocator, outside its hypothesis) | n/a | B=16, 4 values | same |

The original random search behind this work found failure rates of about 0.1-1% for these orders. The minimal witnesses show why: a short-lived scalar placed where a later vector needs contiguous room.

## C3: a sufficient condition for a simple allocator. True.

**Statement.** Let max_v be the most vectors live at any one step, and max_s the most scalars. If `k * max_v + max_s <= B`, the following always succeeds:
- **Pass 1:** vectors in start order, each into the lowest k-aligned slot that is free over its interval.
- **Pass 2:** scalars in start order, each into any word that is free over its interval.

**Lemma (A4-start-order-lemma, UNSAT).** Suppose values are placed in start order. Then a value placed earlier overlaps the new value x exactly when it is live at x's start: earlier values start no later than x, so they overlap x iff they have not ended by x's start. So at most `(number live at x.start) - 1` placed values can block x.

**Proof.**
- *Vectors.* When vector x is placed, at most max_v - 1 other vectors are live at its start (x is one of the live vectors). By the lemma, at most max_v - 1 of the slots 0 .. max_v-1 are blocked, so the lowest free slot exists and is below max_v. All vectors stay in words `[0, k*max_v)`.
- *Scalars.* The words `[k*max_v, B)`, at least max_s of them, never hold a vector. When scalar x is placed, at most max_s - 1 scalars placed earlier are live at its start. By the lemma, those are the only scalars that can block x, so some word in that region is free over x's whole interval. The allocator may also pick a word inside the vector region if one is free there.

This also proves the equal-size case: with no scalars, start-order first fit uses max_v slots.

**Necessity of start order.** The order matters: scalars in arbitrary order can fail. Interval-graph first fit is only optimal in start order. Vectors still need slots in the proof (`vectors_then_scalars` always uses k-aligned slots).

**Checks.**
- `A5-c3-exhaustive-k2-B6-T3-n5` (fast): exhaustive over every instance with sizes {1, 2}, B = 6, at most 3 steps and at most 5 values that satisfies the hypothesis. No failure.
- `A5-c3-exhaustive-k3-B8-T3-n6` (`--slow`): the same for k = 3.
- `test_c3_two_pass_bound_on_random_instances`: 500 seeded random k = 8 instances with B set exactly to the bound.

## Consequences for the allocator

1. **The per-cycle check in the schedulers does not guarantee an address layout.** Counterexamples need a specific pinning shape, so they are rare in random instances; none of the 58,000 random instances in the original exploration was one. But they exist at every budget, so an allocator failure after a passing schedule is not by itself a bug.
2. **The current order (start order, first fit or most-recently-freed reuse) can fail even when a layout exists.** The 4-value B = 16 witnesses above make good unit tests for `address_assign.rs`, and so do the `c1_counterexamples`, which should be reported as "no layout" rather than crash. The `pinning-k8-B16-real` entry is already in real `[def, last_read]` form.
3. **Choosing an allocator.** No polynomial algorithm is complete, so use a fast heuristic with a complete fallback:
   - Keep a start-order greedy as the fast path.
   - On failure, run an exact search: CP-SAT `NoOverlap2D` over (time, address), seeded with the greedy's partial layout. Or retry with a different order.
   - Report a genuine failure only when the exact search proves there is no layout. Then the scheduler must shorten some lifetimes or free more words.
   - If a guarantee is wanted up front instead, the schedulers can enforce C3's condition `8 * max_v + max_s <= B`, and then the two-pass allocator is guaranteed. That condition is much stronger than the per-cycle check, because max_v and max_s may come from different cycles. Whether it is affordable is an empirical question for the 16-round kernel.
4. **The scan frees less than the pairwise rule allows.** A3 verifies the pairwise rule, which says when two intervals may share a word. The heap scan in `assign_addresses` stops expiring at the first heap entry it may not free. So a point interval at the head of the heap keeps the words of later-ending entries busy, even when the rule would let them be reused. Example with two words: the one-word intervals `[0,0]`, `[0,1]`, `[1,1]`, `[1,2]`, allocated in that order. `[1,1]` takes word 0 and becomes the heap head. When `[1,2]` is allocated, the scan stops at `[1,1]` and never frees `[0,1]`'s word 1, although `[1,2]` could use it. This loses layouts but never produces a wrong one. Every "no layout" result above is judged by the pairwise rule, so it holds for the scan too.
5. **Side note on peak counting.** `compute_peak_liveness` in `address_assign.rs` expires an interval with `end <= start` even when that interval is a point `[C, C]`. The allocator does not, since a point never shares with a value written at C. So the reported peak can undercount by the point intervals that end at a cycle where another value starts. The checks here do not depend on it.

## Literature

- M. R. Garey, D. S. Johnson, *Computers and Intractability* (1979), problem [SR2] Dynamic Storage Allocation. NP-complete in the strong sense even when every size is 1 or 2, by a reduction from 3-PARTITION credited to L. Stockmeyer. The proof is reproduced in, for example, [arXiv cs/0605102](https://arxiv.org/abs/cs/0605102).
- J. K. Lee, J. Palsberg, F. M. Q. Pereira, "Aliased register allocation for straight-line programs is NP-complete", ICALP 2007; *Theoretical Computer Science* 407 (2008) 258-273. Covers aligned register pairs on interval (straight-line) lifetimes, which is our aligned variant with k = 2.
- M. Chrobak, M. Ślusarek, "On some packing problem related to dynamic storage allocation", *RAIRO Inform. Théor. Appl.* 22 (1988) 487-499. Studies the two-size case.
- J. Gergov, "Approximation algorithms for dynamic storage allocation", ESA 1996, LNCS 1136, 52-61: a 5-approximation, and a 2-approximation for two sizes. J. Gergov, "Algorithms for compile-time memory optimization", SODA 1999: OPT <= 3 LOAD.
- H. A. Kierstead, "The linearity of first-fit coloring of interval graphs", *SIAM J. Discrete Math.* 1 (1988), and "A polynomial time approximation algorithm for dynamic storage allocation", *Discrete Math.* 88 (1991) 231-237: constant-factor bounds (80, then 6) via on-line interval colouring.
- A. L. Buchsbaum, H. Karloff, C. Kenyon, N. Reingold, M. Thorup, "OPT versus LOAD in dynamic storage allocation", STOC 2003; *SIAM J. Comput.* 33(3) (2004) 632-646. OPT = (1 + o(1)) LOAD when the largest size is small relative to LOAD, and a (2 + ε)-approximation. Their additive term is asymptotic and gives no useful explicit headroom at LOAD = 1536 with size 8.
- Equal sizes: interval graphs are perfect, and start-order first fit colours them optimally (e.g. M. C. Golumbic, *Algorithmic Graph Theory and Perfect Graphs*, 1980).

The citation details for Gergov and Kierstead are from memory and secondary sources, not checked against the papers.

## Files

- [`verification/allocation_conjectures.py`](../verification/allocation_conjectures.py): the model, the layout search (Z3 and an exact backtracking search cross-checked against it), the pinning construction, the real-semantics lemmas, the candidate allocators, the bounded searches (exhaustive and CEGIS), and the check list. Fast checks are registered in `verification/verify.py`.
- [`verification/allocation_counterexamples.json`](../verification/allocation_counterexamples.json): the C1 and C2 counterexamples, one `demonstrates` line each.
- [`tests/test_allocation_conjectures.py`](../tests/test_allocation_conjectures.py): runs the fast checks, and cross-checks the layout search against Z3, `real_conflict` against a port of the Rust expiry rule, greedy layouts for validity, and C3 on random k = 8 instances (`./scripts/test.sh python`).
