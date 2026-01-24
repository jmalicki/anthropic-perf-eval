"""Unit tests for slot utilization analysis and visualization."""

import json
import os
import tempfile
import unittest

from solver.slot_utilization import (
    analyze_memory_lifecycle,
    analyze_utilization,
    compute_utilization,
    generate_memory_svg,
    generate_svg,
    generate_svg_from_json,
    load_json,
    save_json,
    visualize_kernel,
    visualize_kernel_full,
)


class TestComputeUtilization(unittest.TestCase):
    """Tests for the compute_utilization helper function."""

    def test_full_utilization(self):
        """All slots filled should return 100%."""
        instr = {"alu": [("op",)] * 12}
        self.assertEqual(compute_utilization(instr, "alu"), 100.0)

    def test_half_utilization(self):
        """Half slots filled should return 50%."""
        instr = {"valu": [("op",)] * 3}  # 3/6 = 50%
        self.assertEqual(compute_utilization(instr, "valu"), 50.0)

    def test_empty_engine(self):
        """Missing engine should return 0%."""
        instr = {"alu": [("op",)]}
        self.assertEqual(compute_utilization(instr, "valu"), 0.0)

    def test_empty_instruction(self):
        """Empty instruction should return 0%."""
        instr = {}
        self.assertEqual(compute_utilization(instr, "alu"), 0.0)


class TestAnalyzeUtilization(unittest.TestCase):
    """Tests for the analyze_utilization function."""

    def test_empty_instructions(self):
        """Empty instruction list returns empty utilization."""
        result = analyze_utilization([])
        self.assertEqual(result["metadata"]["total_bundles"], 0)
        for engine in ["alu", "valu", "load", "store", "flow"]:
            self.assertEqual(result["utilization"][engine], [])

    def test_single_bundle_full_utilization(self):
        """Bundle with all slots filled should be 100%."""
        instr = {
            "alu": [("op",)] * 12,  # 12/12 = 100%
            "valu": [("op",)] * 6,  # 6/6 = 100%
            "load": [("op",)] * 2,  # 2/2 = 100%
        }
        result = analyze_utilization([instr])
        self.assertEqual(result["utilization"]["alu"], [100.0])
        self.assertEqual(result["utilization"]["valu"], [100.0])
        self.assertEqual(result["utilization"]["load"], [100.0])
        self.assertEqual(result["utilization"]["store"], [0.0])
        self.assertEqual(result["utilization"]["flow"], [0.0])

    def test_partial_utilization(self):
        """Bundle with partial slots should calculate correctly."""
        instr = {
            "alu": [("op",)] * 6,  # 6/12 = 50%
            "valu": [("op",)] * 3,  # 3/6 = 50%
        }
        result = analyze_utilization([instr])
        self.assertEqual(result["utilization"]["alu"], [50.0])
        self.assertEqual(result["utilization"]["valu"], [50.0])

    def test_multiple_bundles(self):
        """Multiple bundles should produce list of utilizations."""
        instrs = [
            {"alu": [("op",)] * 12},  # 100%
            {"alu": [("op",)] * 6},  # 50%
            {},  # 0%
        ]
        result = analyze_utilization(instrs)
        self.assertEqual(result["utilization"]["alu"], [100.0, 50.0, 0.0])
        self.assertEqual(result["metadata"]["total_bundles"], 3)

    def test_summary_statistics(self):
        """Summary should contain mean, max, min for each engine."""
        instrs = [
            {"alu": [("op",)] * 12},  # 100%
            {"alu": [("op",)] * 6},  # 50%
            {"alu": [("op",)] * 0},  # 0%
        ]
        result = analyze_utilization(instrs)
        self.assertAlmostEqual(result["summary"]["alu"]["mean"], 50.0)
        self.assertEqual(result["summary"]["alu"]["max"], 100.0)
        self.assertEqual(result["summary"]["alu"]["min"], 0.0)

    def test_metadata_slot_limits(self):
        """Metadata should include slot limits."""
        result = analyze_utilization([{}])
        self.assertEqual(result["metadata"]["slot_limits"]["alu"], 12)
        self.assertEqual(result["metadata"]["slot_limits"]["valu"], 6)
        self.assertEqual(result["metadata"]["slot_limits"]["load"], 2)
        self.assertEqual(result["metadata"]["slot_limits"]["store"], 2)
        self.assertEqual(result["metadata"]["slot_limits"]["flow"], 1)
        # Debug should be excluded
        self.assertNotIn("debug", result["metadata"]["slot_limits"])

    def test_debug_engine_excluded(self):
        """Debug engine should be excluded from analysis."""
        instr = {
            "alu": [("op",)],
            "debug": [("compare",)] * 10,  # Should be ignored
        }
        result = analyze_utilization([instr])
        self.assertNotIn("debug", result["utilization"])


class TestJsonRoundTrip(unittest.TestCase):
    """Tests for JSON save/load functionality."""

    def test_json_round_trip(self):
        """JSON save/load should preserve data."""
        instrs = [{"alu": [("op",)] * 6, "valu": [("op",)] * 3}]
        original = analyze_utilization(instrs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            save_json(original, json_path)
            loaded = load_json(json_path)
            self.assertEqual(original, loaded)
        finally:
            os.unlink(json_path)

    def test_json_is_readable(self):
        """JSON output should be human-readable (pretty-printed)."""
        instrs = [{"alu": [("op",)] * 6}]
        data = analyze_utilization(instrs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            save_json(data, json_path)
            with open(json_path) as f:
                content = f.read()
                # Should have newlines (pretty-printed)
                self.assertIn("\n", content)
                # Should be valid JSON
                parsed = json.loads(content)
                self.assertIn("metadata", parsed)
        finally:
            os.unlink(json_path)


class TestSvgGeneration(unittest.TestCase):
    """Tests for SVG generation functionality."""

    def test_svg_generation(self):
        """SVG file should be created with valid structure."""
        data = {
            "metadata": {
                "total_bundles": 3,
                "slot_limits": {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1},
            },
            "summary": {"alu": {"mean": 50.0, "max": 100.0, "min": 0.0}},
            "utilization": {
                "alu": [50.0, 75.0, 100.0],
                "valu": [33.3, 66.6, 100.0],
                "load": [0.0, 50.0, 100.0],
                "store": [0.0, 0.0, 50.0],
                "flow": [100.0, 0.0, 100.0],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_svg(data, svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
                self.assertIn("<svg", content)
                self.assertIn("</svg>", content)
                self.assertIn("polyline", content)  # Has data lines
        finally:
            os.unlink(svg_path)

    def test_svg_from_json(self):
        """Should be able to generate SVG from JSON file."""
        data = {
            "metadata": {
                "total_bundles": 2,
                "slot_limits": {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1},
            },
            "summary": {},
            "utilization": {
                "alu": [50.0, 100.0],
                "valu": [50.0, 100.0],
                "load": [50.0, 100.0],
                "store": [0.0, 0.0],
                "flow": [0.0, 0.0],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            save_json(data, json_path)
            generate_svg_from_json(json_path, svg_path)
            self.assertTrue(os.path.exists(svg_path))
        finally:
            os.unlink(json_path)
            os.unlink(svg_path)

    def test_svg_empty_data(self):
        """SVG should handle empty data gracefully."""
        data = {
            "metadata": {
                "total_bundles": 0,
                "slot_limits": {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1},
            },
            "summary": {},
            "utilization": {
                "alu": [],
                "valu": [],
                "load": [],
                "store": [],
                "flow": [],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_svg(data, svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
                self.assertIn("<svg", content)
                self.assertIn("No data to display", content)
        finally:
            os.unlink(svg_path)

    def test_svg_single_bundle(self):
        """SVG should handle single bundle (edge case for x-axis scaling)."""
        data = {
            "metadata": {
                "total_bundles": 1,
                "slot_limits": {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1},
            },
            "summary": {},
            "utilization": {
                "alu": [50.0],
                "valu": [100.0],
                "load": [0.0],
                "store": [0.0],
                "flow": [0.0],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_svg(data, svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
                self.assertIn("<svg", content)
                self.assertIn("polyline", content)
        finally:
            os.unlink(svg_path)

    def test_svg_has_legend(self):
        """SVG should include a legend with engine names and colors."""
        data = {
            "metadata": {
                "total_bundles": 2,
                "slot_limits": {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1},
            },
            "summary": {},
            "utilization": {
                "alu": [50.0, 100.0],
                "valu": [50.0, 100.0],
                "load": [50.0, 100.0],
                "store": [0.0, 0.0],
                "flow": [0.0, 0.0],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_svg(data, svg_path)
            with open(svg_path) as f:
                content = f.read()
                # Check legend labels exist
                for engine in ["alu", "valu", "load", "store", "flow"]:
                    self.assertIn(f">{engine}<", content)
        finally:
            os.unlink(svg_path)


class TestVisualizeKernel(unittest.TestCase):
    """Tests for the visualize_kernel entry point."""

    def test_visualize_kernel_creates_both_files(self):
        """visualize_kernel should create both .json and .svg files."""
        instrs = [{"alu": [("op",)] * 6}]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output")
            visualize_kernel(instrs, base_path)

            self.assertTrue(os.path.exists(f"{base_path}.json"))
            self.assertTrue(os.path.exists(f"{base_path}.svg"))

    def test_visualize_kernel_json_content(self):
        """visualize_kernel should produce valid JSON content."""
        instrs = [
            {"alu": [("op",)] * 12, "valu": [("op",)] * 3},
            {"alu": [("op",)] * 6},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output")
            visualize_kernel(instrs, base_path)

            with open(f"{base_path}.json") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["total_bundles"], 2)
            self.assertEqual(data["utilization"]["alu"], [100.0, 50.0])
            self.assertEqual(data["utilization"]["valu"], [50.0, 0.0])


class TestMemoryLifecycle(unittest.TestCase):
    """Tests for memory lifecycle analysis."""

    def test_empty_instructions(self):
        """Empty instruction list returns empty memory data."""
        result = analyze_memory_lifecycle([])
        self.assertEqual(result["metadata"]["total_bundles"], 0)
        self.assertEqual(result["scratch_active"], [])
        self.assertEqual(result["mem_loads"], [])
        self.assertEqual(result["mem_stores"], [])

    def test_scratch_alloc_tracking(self):
        """Should track scratch allocations."""
        instrs = [
            {"debug": [("scratch_alloc", 0, 8)]},  # Allocate 8 words
            {},  # No change
            {"debug": [("scratch_alloc", 8, 4)]},  # Allocate 4 more
        ]
        result = analyze_memory_lifecycle(instrs, scratch_size=100)

        self.assertEqual(result["scratch_active"], [8, 8, 12])
        self.assertEqual(result["summary"]["scratch"]["peak_words"], 12)

    def test_scratch_retire_tracking(self):
        """Should track scratch retirements."""
        instrs = [
            {"debug": [("scratch_alloc", 0, 8)]},  # Allocate 8 words
            {"debug": [("scratch_retire", 0, 4)]},  # Retire 4 words
            {"debug": [("scratch_retire", 4, 4)]},  # Retire remaining 4
        ]
        result = analyze_memory_lifecycle(instrs, scratch_size=100)

        self.assertEqual(result["scratch_active"], [8, 4, 0])
        self.assertEqual(result["summary"]["scratch"]["peak_words"], 8)

    def test_memory_load_tracking(self):
        """Should track memory load operations."""
        instrs = [
            {"debug": [("mem_load", 10)]},
            {"debug": [("mem_load", 11), ("mem_load", 12)]},
            {},
        ]
        result = analyze_memory_lifecycle(instrs)

        self.assertEqual(result["mem_loads"], [1, 3, 3])
        self.assertEqual(result["summary"]["memory"]["total_loads"], 3)

    def test_memory_store_tracking(self):
        """Should track memory store operations."""
        instrs = [
            {},
            {"debug": [("mem_store", 20)]},
            {"debug": [("mem_store", 21)]},
        ]
        result = analyze_memory_lifecycle(instrs)

        self.assertEqual(result["mem_stores"], [0, 1, 2])
        self.assertEqual(result["summary"]["memory"]["total_stores"], 2)

    def test_combined_events(self):
        """Should handle all event types together."""
        instrs = [
            {
                "alu": [("op",)],
                "debug": [
                    ("scratch_alloc", 0, 10),
                    ("mem_load", 5),
                ],
            },
            {
                "alu": [("op",)],
                "debug": [
                    ("scratch_retire", 0, 5),
                    ("mem_store", 6),
                ],
            },
        ]
        result = analyze_memory_lifecycle(instrs, scratch_size=100)

        self.assertEqual(result["scratch_active"], [10, 5])
        self.assertEqual(result["mem_loads"], [1, 1])
        self.assertEqual(result["mem_stores"], [0, 1])

    def test_scratch_utilization_percentage(self):
        """Should compute scratch utilization as percentage."""
        instrs = [{"debug": [("scratch_alloc", 0, 50)]}]  # 50/100 = 50%
        result = analyze_memory_lifecycle(instrs, scratch_size=100)

        self.assertEqual(result["scratch_active_pct"], [50.0])
        self.assertEqual(result["summary"]["scratch"]["peak_pct"], 50.0)


class TestMemorySvgGeneration(unittest.TestCase):
    """Tests for memory SVG generation."""

    def test_memory_svg_generation(self):
        """Should generate valid memory SVG."""
        data = {
            "metadata": {"total_bundles": 3, "scratch_size": 100},
            "summary": {
                "scratch": {"peak_words": 50, "peak_pct": 50.0, "mean_pct": 30.0,
                            "total_allocs": 50, "total_retires": 30},
                "memory": {"total_loads": 10, "total_stores": 5},
            },
            "scratch_active": [20, 50, 30],
            "scratch_active_pct": [20.0, 50.0, 30.0],
            "scratch_allocs": [20, 50, 50],
            "scratch_retires": [0, 0, 20],
            "mem_loads": [3, 7, 10],
            "mem_stores": [1, 3, 5],
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_memory_svg(data, svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
                self.assertIn("<svg", content)
                self.assertIn("</svg>", content)
                self.assertIn("Memory Lifecycle", content)
                self.assertIn("Scratch Memory", content)
        finally:
            os.unlink(svg_path)

    def test_memory_svg_empty_data(self):
        """Memory SVG should handle empty data gracefully."""
        data = {
            "metadata": {"total_bundles": 0, "scratch_size": 100},
            "summary": {
                "scratch": {"peak_words": 0, "peak_pct": 0.0, "mean_pct": 0.0,
                            "total_allocs": 0, "total_retires": 0},
                "memory": {"total_loads": 0, "total_stores": 0},
            },
            "scratch_active": [],
            "scratch_active_pct": [],
            "scratch_allocs": [],
            "scratch_retires": [],
            "mem_loads": [],
            "mem_stores": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_path = f.name

        try:
            generate_memory_svg(data, svg_path)
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path) as f:
                content = f.read()
                self.assertIn("No data to display", content)
        finally:
            os.unlink(svg_path)


class TestVisualizeKernelFull(unittest.TestCase):
    """Tests for the visualize_kernel_full entry point."""

    def test_creates_all_files(self):
        """visualize_kernel_full should create all output files."""
        instrs = [
            {
                "alu": [("op",)] * 6,
                "debug": [("scratch_alloc", 0, 10), ("mem_load", 5)],
            },
            {
                "alu": [("op",)] * 3,
                "debug": [("scratch_retire", 0, 5), ("mem_store", 6)],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output")
            visualize_kernel_full(instrs, base_path)

            # Check slot files
            self.assertTrue(os.path.exists(f"{base_path}_slots.json"))
            self.assertTrue(os.path.exists(f"{base_path}_slots.svg"))

            # Check memory files
            self.assertTrue(os.path.exists(f"{base_path}_memory.json"))
            self.assertTrue(os.path.exists(f"{base_path}_memory.svg"))

    def test_json_content_valid(self):
        """visualize_kernel_full should produce valid JSON content."""
        instrs = [
            {"debug": [("scratch_alloc", 0, 20), ("mem_load", 1)]},
            {"debug": [("scratch_retire", 0, 10)]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output")
            visualize_kernel_full(instrs, base_path, scratch_size=100)

            with open(f"{base_path}_memory.json") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["total_bundles"], 2)
            self.assertEqual(data["scratch_active"], [20, 10])
            self.assertEqual(data["summary"]["scratch"]["peak_words"], 20)


if __name__ == "__main__":
    unittest.main()
