#!/usr/bin/env sh
# Run generate_ssa_kernel.py, rebuilding scheduler_rs first if Rust sources changed.
# Usage: from project root, ./scripts/run_generate_ssa_kernel.sh [args...]
# Example: ./scripts/run_generate_ssa_kernel.sh --enable-valu-shift --cpsat-time-limit 7200 --validate --rounds 5
set -e
cd "$(dirname "$0")/.."

# Compare newest scheduler_rs Rust source/Cargo.toml mtime vs installed extension mtime.
# If Rust is newer, rebuild so the script uses the updated extension.
need_rebuild() {
  [ -d "scheduler_rs/src" ] || return 1
  out=$(uv run python -c '
import os
import sys
try:
    import scheduler_rs
    ext_path = scheduler_rs.__file__
except Exception:
    sys.exit(1)
if not ext_path or not os.path.isfile(ext_path):
    sys.exit(1)
ext_mtime = int(os.path.getmtime(ext_path))
rust_newest = 0
for dirpath, _, filenames in os.walk("scheduler_rs"):
    for f in filenames:
        if f.endswith(".rs") or f == "Cargo.toml":
            path = os.path.join(dirpath, f)
            rust_newest = max(rust_newest, int(os.path.getmtime(path)))
print(ext_mtime, rust_newest)
' 2>/dev/null) || return 1
  ext_mtime="${out%% *}"
  rust_newest="${out##* }"
  [ -n "$ext_mtime" ] && [ -n "$rust_newest" ] && [ "$rust_newest" -gt "$ext_mtime" ]
}

if need_rebuild; then
  echo "scheduler_rs Rust sources newer than installed extension; rebuilding..."
  ./scripts/build_scheduler_rs.sh
fi

exec uv run generate_ssa_kernel.py "$@"
