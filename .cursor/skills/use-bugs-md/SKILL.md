---
name: use-bugs-md
description: Read and update the project BUGS.md file for bug tracking. Use when fixing bugs, reporting new bugs, triaging issues, or when the user asks about known bugs or the bug list.
---

# Using BUGS.md

This project tracks bugs in **BUGS.md** at the repo root. Use it so the agent and humans share one source of truth.

## Location

- **Path:** `BUGS.md` (project root)
- If the file is missing, create it using the format below.

## When to Use

1. **Before fixing a bug** — Read BUGS.md to see if the issue is listed and to load context (repro, area).
2. **When a bug is fixed** — Move the entry from "Open" to "Fixed" and add a brief note (e.g. PR or cause).
3. **When a new bug is found or reported** — Add an entry under "Open" with title, location, and optional repro.
4. **When asked "what are the known bugs?"** — Read BUGS.md and summarize.

## Format

Keep entries short. One line per bug when possible.

```markdown
# BUGS.md

## Open

- [ ] **Short title** – One-line description. Optional: `area` or `path` (e.g. `scheduler_rs/`, `solver/`). Optional: repro or doc link.
- [ ] **Another bug** – Description.

## Fixed

- [x] **Fixed bug title** – How it was fixed (e.g. "Fixed in PR #N" or "Root cause: …").
```

- Use `- [ ]` for open and `- [x]` for fixed.
- **Bold** the title; after the em dash add location/repro/docs as needed.
- Prefer linking to docs (e.g. `docs/CP-SAT-minimal-failing-case.md`) or file paths instead of long prose in BUGS.md.

## Agent Workflow

1. **Read** — Use the read_file tool on `BUGS.md` when starting bug-related work or when the user asks about bugs.
2. **Update** — Use search_replace or the write tool to add, complete, or move entries. Preserve existing content and section order (Open, then Fixed).
3. **Don’t invent** — Only add bugs the user or the codebase clearly indicate; don’t populate BUGS.md with speculative items.

## Example Entry

```markdown
- [ ] **CP-SAT slot limit violation** – If a CP-SAT solution ever violates per-cycle slot limits, treat as constraint bug. See docs/CPSAT_SCHEDULER_IMPLEMENTATION.md.
```
