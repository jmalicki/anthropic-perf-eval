"""
Slot utilization analysis and SVG visualization for VLIW instruction bundles.

Workflow:
1. analyze_utilization() - Extract raw utilization data from instructions
2. save_json() - Save data to JSON file (human-readable intermediate)
3. load_json() - Load data from JSON file
4. generate_svg() - Generate SVG visualization from utilization data

Usage:
    # Full pipeline
    visualize_kernel(instrs, "output")  # Creates output.json and output.svg

    # Or step-by-step
    data = analyze_utilization(instrs)
    save_json(data, "output.json")
    generate_svg_from_json("output.json", "output.svg")
"""

import json
from problem import SLOT_LIMITS

# Engines to visualize (exclude debug)
ENGINES = ["alu", "valu", "load", "store", "flow"]

# Color scheme for each engine
ENGINE_COLORS = {
    "alu": "#3498db",    # Blue
    "valu": "#e74c3c",   # Red
    "load": "#2ecc71",   # Green
    "store": "#f39c12",  # Orange
    "flow": "#9b59b6",   # Purple
}


def compute_utilization(instr: dict, engine: str) -> float:
    """
    Compute utilization percentage for an engine in a bundle.

    Args:
        instr: Instruction bundle dict mapping engine names to slot lists
        engine: Engine name to compute utilization for

    Returns:
        Utilization as a percentage (0.0 to 100.0)
    """
    slots_used = len(instr.get(engine, []))
    return (slots_used / SLOT_LIMITS[engine]) * 100.0


def analyze_utilization(instrs: list[dict]) -> dict:
    """
    Analyze slot utilization across all bundles.

    Args:
        instrs: List of instruction bundles

    Returns:
        Dict with metadata, summary stats, and per-bundle utilization
    """
    # Build slot_limits dict for metadata (exclude debug)
    slot_limits = {engine: SLOT_LIMITS[engine] for engine in ENGINES}

    # Initialize utilization arrays
    utilization = {engine: [] for engine in ENGINES}

    # Compute per-bundle utilization
    for instr in instrs:
        for engine in ENGINES:
            util = compute_utilization(instr, engine)
            utilization[engine].append(util)

    # Compute summary statistics
    summary = {}
    for engine in ENGINES:
        utils = utilization[engine]
        if utils:
            summary[engine] = {
                "mean": sum(utils) / len(utils),
                "max": max(utils),
                "min": min(utils),
            }
        else:
            summary[engine] = {"mean": 0.0, "max": 0.0, "min": 0.0}

    return {
        "metadata": {
            "total_bundles": len(instrs),
            "slot_limits": slot_limits,
        },
        "summary": summary,
        "utilization": utilization,
    }


def save_json(data: dict, output_path: str) -> None:
    """Save utilization data to JSON file (pretty-printed)."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(input_path: str) -> dict:
    """Load utilization data from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def generate_svg(
    data: dict, output_path: str, width: int = 1200, height: int = 400
) -> None:
    """
    Generate SVG line graph from utilization data dict.

    Args:
        data: Utilization data dict from analyze_utilization()
        output_path: Path to write SVG file
        width: SVG width in pixels
        height: SVG height in pixels
    """
    # Layout constants
    margin_left = 80
    margin_right = 150  # Extra space for legend
    margin_top = 50
    margin_bottom = 50

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    utilization = data["utilization"]
    total_bundles = data["metadata"]["total_bundles"]

    # Handle empty data case
    if total_bundles == 0:
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width // 2}" y="{height // 2}" text-anchor="middle" font-family="sans-serif" font-size="14">No data to display</text>
</svg>"""
        with open(output_path, "w") as f:
            f.write(svg_content)
        return

    # Build SVG elements
    svg_parts = []

    # SVG header and background
    svg_parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    svg_parts.append('  <rect width="100%" height="100%" fill="white"/>')

    # Title
    svg_parts.append(
        f'  <text x="{width // 2}" y="25" text-anchor="middle" '
        f'font-family="sans-serif" font-size="16" font-weight="bold">'
        f"Slot Utilization by Engine</text>"
    )

    # Y-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + plot_height}" '
        f'stroke="black" stroke-width="1"/>'
    )

    # Y-axis label
    y_label_x = 25
    y_label_y = margin_top + plot_height // 2
    svg_parts.append(
        f'  <text x="{y_label_x}" y="{y_label_y}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" '
        f'transform="rotate(-90,{y_label_x},{y_label_y})">Utilization %</text>'
    )

    # Y-axis tick marks and labels (0%, 25%, 50%, 75%, 100%)
    for pct in [0, 25, 50, 75, 100]:
        y = margin_top + plot_height - (pct / 100.0) * plot_height
        # Tick mark
        svg_parts.append(
            f'  <line x1="{margin_left - 5}" y1="{y}" '
            f'x2="{margin_left}" y2="{y}" stroke="black" stroke-width="1"/>'
        )
        # Label
        svg_parts.append(
            f'  <text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10">{pct}%</text>'
        )
        # Grid line
        if pct > 0:
            svg_parts.append(
                f'  <line x1="{margin_left}" y1="{y}" '
                f'x2="{margin_left + plot_width}" y2="{y}" '
                f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
            )

    # X-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
        f'stroke="black" stroke-width="1"/>'
    )

    # X-axis label
    svg_parts.append(
        f'  <text x="{margin_left + plot_width // 2}" y="{height - 10}" '
        f'text-anchor="middle" font-family="sans-serif" font-size="12">Bundle Index</text>'
    )

    # X-axis tick marks (show a few markers)
    num_x_ticks = min(10, total_bundles)
    if num_x_ticks > 1:
        for i in range(num_x_ticks + 1):
            bundle_idx = int(i * (total_bundles - 1) / num_x_ticks)
            x = margin_left + (bundle_idx / (total_bundles - 1)) * plot_width
            y = margin_top + plot_height
            # Tick mark
            svg_parts.append(
                f'  <line x1="{x}" y1="{y}" x2="{x}" y2="{y + 5}" '
                f'stroke="black" stroke-width="1"/>'
            )
            # Label
            svg_parts.append(
                f'  <text x="{x}" y="{y + 18}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10">{bundle_idx}</text>'
            )

    # Data lines (one per engine)
    for engine in ENGINES:
        utils = utilization[engine]
        if not utils:
            continue

        color = ENGINE_COLORS[engine]
        points = []

        for i, util in enumerate(utils):
            if total_bundles > 1:
                x = margin_left + (i / (total_bundles - 1)) * plot_width
            else:
                x = margin_left + plot_width / 2
            y = margin_top + plot_height - (util / 100.0) * plot_height
            points.append(f"{x:.1f},{y:.1f}")

        points_str = " ".join(points)
        svg_parts.append(
            f'  <polyline points="{points_str}" stroke="{color}" '
            f'fill="none" stroke-width="1.5" stroke-opacity="0.8"/>'
        )

    # Legend
    legend_x = margin_left + plot_width + 20
    legend_y = margin_top + 20

    for i, engine in enumerate(ENGINES):
        color = ENGINE_COLORS[engine]
        y = legend_y + i * 25

        # Color line
        svg_parts.append(
            f'  <line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        # Label
        svg_parts.append(
            f'  <text x="{legend_x + 40}" y="{y + 4}" '
            f'font-family="sans-serif" font-size="12">{engine}</text>'
        )

    # Close SVG
    svg_parts.append("</svg>")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(svg_parts))


def generate_svg_from_json(
    json_path: str, svg_path: str, width: int = 1200, height: int = 400
) -> None:
    """Generate SVG from a JSON data file."""
    data = load_json(json_path)
    generate_svg(data, svg_path, width, height)


def visualize_kernel(instrs: list[dict], output_base: str) -> None:
    """
    Main entry point: analyze, save JSON, and generate SVG.

    Args:
        instrs: List of instruction bundles
        output_base: Base path (will create output_base.json and output_base.svg)
    """
    data = analyze_utilization(instrs)
    save_json(data, f"{output_base}.json")
    generate_svg(data, f"{output_base}.svg")


# =============================================================================
# Memory Lifecycle Analysis
# =============================================================================

# Colors for memory tracking
MEMORY_COLORS = {
    "scratch_active": "#3498db",  # Blue - active scratch words
    "scratch_peak": "#e74c3c",    # Red - peak scratch usage
    "mem_loads": "#2ecc71",       # Green - cumulative loads
    "mem_stores": "#f39c12",      # Orange - cumulative stores
}


def analyze_memory_lifecycle(
    instrs: list[dict], scratch_size: int = 1536
) -> dict:
    """
    Analyze memory lifecycle from debug instructions.

    Parses debug events (scratch_alloc, scratch_retire, mem_load, mem_store)
    to track scratch memory usage and main memory accesses over time.

    Args:
        instrs: List of instruction bundles (with debug events)
        scratch_size: Total scratch memory size for utilization calculation

    Returns:
        Dict with metadata, summary, and per-bundle memory data:
        - scratch_active: Number of active scratch words at each bundle
        - scratch_allocs: Cumulative allocations at each bundle
        - scratch_retires: Cumulative retirements at each bundle
        - mem_loads: Cumulative load operations at each bundle
        - mem_stores: Cumulative store operations at each bundle
    """
    total_bundles = len(instrs)

    # Track state
    active_scratch = set()  # Currently live scratch addresses
    cumulative_allocs = 0
    cumulative_retires = 0
    cumulative_loads = 0
    cumulative_stores = 0

    # Per-bundle data
    scratch_active = []
    scratch_allocs = []
    scratch_retires = []
    mem_loads = []
    mem_stores = []
    scratch_active_pct = []

    peak_scratch = 0

    for instr in instrs:
        debug_slots = instr.get("debug", [])

        bundle_allocs = 0
        bundle_retires = 0
        bundle_loads = 0
        bundle_stores = 0

        for slot in debug_slots:
            if slot[0] == "scratch_alloc":
                addr, length = slot[1], slot[2]
                for a in range(addr, addr + length):
                    active_scratch.add(a)
                bundle_allocs += length
            elif slot[0] == "scratch_retire":
                addr, length = slot[1], slot[2]
                for a in range(addr, addr + length):
                    active_scratch.discard(a)
                bundle_retires += length
            elif slot[0] == "mem_load":
                bundle_loads += 1
            elif slot[0] == "mem_store":
                bundle_stores += 1

        cumulative_allocs += bundle_allocs
        cumulative_retires += bundle_retires
        cumulative_loads += bundle_loads
        cumulative_stores += bundle_stores

        current_active = len(active_scratch)
        peak_scratch = max(peak_scratch, current_active)

        scratch_active.append(current_active)
        scratch_allocs.append(cumulative_allocs)
        scratch_retires.append(cumulative_retires)
        mem_loads.append(cumulative_loads)
        mem_stores.append(cumulative_stores)
        scratch_active_pct.append((current_active / scratch_size) * 100.0 if scratch_size > 0 else 0.0)

    # Summary statistics
    summary = {
        "scratch": {
            "peak_words": peak_scratch,
            "peak_pct": (peak_scratch / scratch_size) * 100.0 if scratch_size > 0 else 0.0,
            "total_allocs": cumulative_allocs,
            "total_retires": cumulative_retires,
        },
        "memory": {
            "total_loads": cumulative_loads,
            "total_stores": cumulative_stores,
        },
    }

    # Compute mean scratch utilization
    if scratch_active_pct:
        summary["scratch"]["mean_pct"] = sum(scratch_active_pct) / len(scratch_active_pct)
    else:
        summary["scratch"]["mean_pct"] = 0.0

    return {
        "metadata": {
            "total_bundles": total_bundles,
            "scratch_size": scratch_size,
        },
        "summary": summary,
        "scratch_active": scratch_active,
        "scratch_active_pct": scratch_active_pct,
        "scratch_allocs": scratch_allocs,
        "scratch_retires": scratch_retires,
        "mem_loads": mem_loads,
        "mem_stores": mem_stores,
    }


def generate_memory_svg(
    data: dict, output_path: str, width: int = 1200, height: int = 500
) -> None:
    """
    Generate SVG visualization for memory lifecycle.

    Creates a two-panel visualization:
    - Top panel: Scratch memory utilization over time (area chart)
    - Bottom panel: Cumulative memory operations (line chart)

    Args:
        data: Memory lifecycle data dict from analyze_memory_lifecycle()
        output_path: Path to write SVG file
        width: SVG width in pixels
        height: SVG height in pixels
    """
    total_bundles = data["metadata"]["total_bundles"]
    scratch_size = data["metadata"]["scratch_size"]

    # Handle empty data
    if total_bundles == 0:
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width // 2}" y="{height // 2}" text-anchor="middle" font-family="sans-serif" font-size="14">No data to display</text>
</svg>"""
        with open(output_path, "w") as f:
            f.write(svg_content)
        return

    # Layout constants
    margin_left = 80
    margin_right = 150
    margin_top = 40
    margin_bottom = 40
    panel_gap = 40

    panel_height = (height - margin_top - margin_bottom - panel_gap) // 2
    plot_width = width - margin_left - margin_right

    svg_parts = []
    svg_parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    svg_parts.append('  <rect width="100%" height="100%" fill="white"/>')

    # Title
    svg_parts.append(
        f'  <text x="{width // 2}" y="20" text-anchor="middle" '
        f'font-family="sans-serif" font-size="16" font-weight="bold">'
        f"Memory Lifecycle</text>"
    )

    # =========================================================================
    # Top Panel: Scratch Utilization
    # =========================================================================
    panel1_top = margin_top
    panel1_bottom = panel1_top + panel_height

    # Panel title
    svg_parts.append(
        f'  <text x="{margin_left}" y="{panel1_top - 5}" '
        f'font-family="sans-serif" font-size="12" font-weight="bold">'
        f"Scratch Memory Utilization</text>"
    )

    # Y-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{panel1_top}" '
        f'x2="{margin_left}" y2="{panel1_bottom}" stroke="black" stroke-width="1"/>'
    )

    # Y-axis label
    y_label_y = panel1_top + panel_height // 2
    svg_parts.append(
        f'  <text x="25" y="{y_label_y}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="10" '
        f'transform="rotate(-90,25,{y_label_y})">Active Words</text>'
    )

    # Y-axis ticks (0 to scratch_size)
    peak_scratch = data["summary"]["scratch"]["peak_words"]
    y_max = max(peak_scratch, scratch_size // 2)  # At least show half capacity

    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = int(frac * y_max)
        y = panel1_bottom - frac * panel_height
        svg_parts.append(
            f'  <line x1="{margin_left - 5}" y1="{y}" x2="{margin_left}" y2="{y}" '
            f'stroke="black" stroke-width="1"/>'
        )
        svg_parts.append(
            f'  <text x="{margin_left - 8}" y="{y + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="9">{val}</text>'
        )
        if frac > 0:
            svg_parts.append(
                f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" '
                f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
            )

    # X-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{panel1_bottom}" '
        f'x2="{margin_left + plot_width}" y2="{panel1_bottom}" stroke="black" stroke-width="1"/>'
    )

    # Draw scratch active as area chart
    scratch_active = data["scratch_active"]
    if scratch_active and y_max > 0:
        points = [f"{margin_left},{panel1_bottom}"]  # Start at bottom-left
        for i, val in enumerate(scratch_active):
            if total_bundles > 1:
                x = margin_left + (i / (total_bundles - 1)) * plot_width
            else:
                x = margin_left + plot_width / 2
            y = panel1_bottom - (val / y_max) * panel_height
            points.append(f"{x:.1f},{y:.1f}")
        points.append(f"{margin_left + plot_width},{panel1_bottom}")  # End at bottom-right

        svg_parts.append(
            f'  <polygon points="{" ".join(points)}" '
            f'fill="{MEMORY_COLORS["scratch_active"]}" fill-opacity="0.3" '
            f'stroke="{MEMORY_COLORS["scratch_active"]}" stroke-width="1.5"/>'
        )

    # Draw peak line
    if y_max > 0:
        peak_y = panel1_bottom - (peak_scratch / y_max) * panel_height
        svg_parts.append(
            f'  <line x1="{margin_left}" y1="{peak_y}" '
            f'x2="{margin_left + plot_width}" y2="{peak_y}" '
            f'stroke="{MEMORY_COLORS["scratch_peak"]}" stroke-width="1" stroke-dasharray="6,3"/>'
        )
        svg_parts.append(
            f'  <text x="{margin_left + plot_width + 5}" y="{peak_y + 4}" '
            f'font-family="sans-serif" font-size="10" fill="{MEMORY_COLORS["scratch_peak"]}">'
            f'Peak: {peak_scratch}</text>'
        )

    # =========================================================================
    # Bottom Panel: Memory Operations
    # =========================================================================
    panel2_top = panel1_bottom + panel_gap
    panel2_bottom = panel2_top + panel_height

    # Panel title
    svg_parts.append(
        f'  <text x="{margin_left}" y="{panel2_top - 5}" '
        f'font-family="sans-serif" font-size="12" font-weight="bold">'
        f"Cumulative Memory Operations</text>"
    )

    # Y-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{panel2_top}" '
        f'x2="{margin_left}" y2="{panel2_bottom}" stroke="black" stroke-width="1"/>'
    )

    # Y-axis label
    y_label_y = panel2_top + panel_height // 2
    svg_parts.append(
        f'  <text x="25" y="{y_label_y}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="10" '
        f'transform="rotate(-90,25,{y_label_y})">Operations</text>'
    )

    # Compute Y-max for memory ops
    mem_loads = data["mem_loads"]
    mem_stores = data["mem_stores"]
    max_ops = max(max(mem_loads) if mem_loads else 0, max(mem_stores) if mem_stores else 0, 1)

    # Y-axis ticks
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = int(frac * max_ops)
        y = panel2_bottom - frac * panel_height
        svg_parts.append(
            f'  <line x1="{margin_left - 5}" y1="{y}" x2="{margin_left}" y2="{y}" '
            f'stroke="black" stroke-width="1"/>'
        )
        svg_parts.append(
            f'  <text x="{margin_left - 8}" y="{y + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="9">{val}</text>'
        )
        if frac > 0:
            svg_parts.append(
                f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" '
                f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
            )

    # X-axis
    svg_parts.append(
        f'  <line x1="{margin_left}" y1="{panel2_bottom}" '
        f'x2="{margin_left + plot_width}" y2="{panel2_bottom}" stroke="black" stroke-width="1"/>'
    )

    # X-axis label
    svg_parts.append(
        f'  <text x="{margin_left + plot_width // 2}" y="{height - 10}" '
        f'text-anchor="middle" font-family="sans-serif" font-size="12">Bundle Index</text>'
    )

    # Draw loads line
    if mem_loads and max_ops > 0:
        points = []
        for i, val in enumerate(mem_loads):
            if total_bundles > 1:
                x = margin_left + (i / (total_bundles - 1)) * plot_width
            else:
                x = margin_left + plot_width / 2
            y = panel2_bottom - (val / max_ops) * panel_height
            points.append(f"{x:.1f},{y:.1f}")
        svg_parts.append(
            f'  <polyline points="{" ".join(points)}" '
            f'stroke="{MEMORY_COLORS["mem_loads"]}" fill="none" stroke-width="1.5"/>'
        )

    # Draw stores line
    if mem_stores and max_ops > 0:
        points = []
        for i, val in enumerate(mem_stores):
            if total_bundles > 1:
                x = margin_left + (i / (total_bundles - 1)) * plot_width
            else:
                x = margin_left + plot_width / 2
            y = panel2_bottom - (val / max_ops) * panel_height
            points.append(f"{x:.1f},{y:.1f}")
        svg_parts.append(
            f'  <polyline points="{" ".join(points)}" '
            f'stroke="{MEMORY_COLORS["mem_stores"]}" fill="none" stroke-width="1.5"/>'
        )

    # Legend for bottom panel
    legend_x = margin_left + plot_width + 10
    legend_y = panel2_top + 20

    for i, (label, color) in enumerate([("loads", MEMORY_COLORS["mem_loads"]),
                                         ("stores", MEMORY_COLORS["mem_stores"])]):
        y = legend_y + i * 20
        svg_parts.append(
            f'  <line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        svg_parts.append(
            f'  <text x="{legend_x + 30}" y="{y + 4}" '
            f'font-family="sans-serif" font-size="10">{label}</text>'
        )

    svg_parts.append("</svg>")

    with open(output_path, "w") as f:
        f.write("\n".join(svg_parts))


def generate_memory_svg_from_json(
    json_path: str, svg_path: str, width: int = 1200, height: int = 500
) -> None:
    """Generate memory SVG from a JSON data file."""
    data = load_json(json_path)
    generate_memory_svg(data, svg_path, width, height)


def visualize_kernel_full(
    instrs: list[dict], output_base: str, scratch_size: int = 1536
) -> None:
    """
    Full visualization: slot utilization + memory lifecycle.

    Creates:
    - {output_base}_slots.json / {output_base}_slots.svg - Slot utilization
    - {output_base}_memory.json / {output_base}_memory.svg - Memory lifecycle

    Args:
        instrs: List of instruction bundles (with debug events for memory tracking)
        output_base: Base path for output files
        scratch_size: Total scratch memory size
    """
    # Slot utilization
    slot_data = analyze_utilization(instrs)
    save_json(slot_data, f"{output_base}_slots.json")
    generate_svg(slot_data, f"{output_base}_slots.svg")

    # Memory lifecycle
    mem_data = analyze_memory_lifecycle(instrs, scratch_size)
    save_json(mem_data, f"{output_base}_memory.json")
    generate_memory_svg(mem_data, f"{output_base}_memory.svg")
