import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from s2omics.batch import (
    allocate_output_names,
    build_parser,
    config_from_row,
    defaults_from_args,
    discover_svs,
    load_manifest_rows,
    run_batch,
    run_roi_selection_pipeline,
)


class BatchTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "slide.svs"
        self.image.write_bytes(b"test")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        parser = build_parser()
        args = parser.parse_args([
            "--input-dir", str(self.root),
            "--output-root", str(self.root / "output"),
            "--pixel-size-um", "0.5",
            "--ckpt-path", str(self.checkpoint),
        ])
        self.defaults = defaults_from_args(args)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **overrides):
        row = {"image_path": str(self.image), "sample_id": "slide"}
        row.update(overrides)
        return config_from_row(row, self.defaults, self.root)

    def test_csv_and_tsv_manifest_loading(self):
        for suffix, delimiter in ((".csv", ","), (".tsv", "\t")):
            path = self.root / f"manifest{suffix}"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "roi_shape", "roi_radius_mm"],
                    delimiter=delimiter,
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": "slide.svs",
                    "roi_shape": "circle",
                    "roi_radius_mm": 1.25,
                })
            rows = load_manifest_rows(path)
            self.assertEqual(rows[0]["roi_shape"], "circle")
            self.assertEqual(str(rows[0]["roi_radius_mm"]), "1.25")

    def test_xlsx_manifest_template_loading(self):
        template = Path(__file__).parents[1] / "examples" / "roi_batch_manifest.xlsx"
        rows = load_manifest_rows(template, "Samples")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["roi_shape"], "rectangle")
        self.assertEqual(rows[1]["roi_shape"], "circle")
        self.assertEqual(int(rows[1]["num_roi"]), 2)

    def test_unknown_populated_column_is_rejected(self):
        path = self.root / "manifest.csv"
        path.write_text("image_path,n_clusterz\nslide.svs,10\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown columns"):
            load_manifest_rows(path)

    def test_shape_size_and_count_are_per_sample(self):
        circle = self.config(
            roi_shape="circle", roi_radius_mm="1.25", num_roi="3"
        )
        self.assertEqual(circle.roi_shape, "circle")
        self.assertEqual(circle.roi_radius_mm, 1.25)
        self.assertEqual(circle.num_roi, 3)

        rectangle = self.config(
            roi_shape="rectangle", roi_width_mm="5", roi_height_mm="4",
            num_roi="0",
        )
        self.assertEqual((rectangle.roi_width_mm, rectangle.roi_height_mm), (5, 4))
        self.assertEqual(rectangle.num_roi, 0)

    def test_discovery_is_case_insensitive_and_optionally_recursive(self):
        nested = self.root / "nested"
        nested.mkdir()
        (self.root / "second.SVS").write_bytes(b"test")
        (nested / "third.svs").write_bytes(b"test")
        top_level = discover_svs(self.root)
        recursive = discover_svs(self.root, recursive=True)
        self.assertEqual(len(top_level), 2)
        self.assertEqual(len(recursive), 3)

    def test_duplicate_and_existing_names_get_numeric_suffixes(self):
        output = self.root / "output"
        output.mkdir()
        (output / "slide").mkdir()
        resolved = allocate_output_names([self.config(), self.config()], output)
        self.assertEqual(
            [item.resolved_sample_id for item in resolved], ["slide-1", "slide-2"]
        )

    def test_pipeline_dispatches_rectangle_and_circle_parameters(self):
        calls = []

        def module(name, function_name, label):
            fake = types.ModuleType(name)

            def record(*args, **kwargs):
                calls.append((label, args, kwargs))

            setattr(fake, function_name, record)
            return fake

        modules = {
            "s2omics.p1_histology_preprocess": module(
                "s2omics.p1_histology_preprocess", "histology_preprocess", "preprocess"
            ),
            "s2omics.p2_superpixel_quality_control": module(
                "s2omics.p2_superpixel_quality_control", "superpixel_quality_control", "qc"
            ),
            "s2omics.p3_feature_extraction": module(
                "s2omics.p3_feature_extraction", "histology_feature_extraction", "features"
            ),
            "s2omics.single_section.p4_get_histology_segmentation": module(
                "s2omics.single_section.p4_get_histology_segmentation",
                "get_histology_segmentation", "segmentation",
            ),
            "s2omics.single_section.p5_merge_over_clusters": module(
                "s2omics.single_section.p5_merge_over_clusters",
                "merge_over_clusters", "merge",
            ),
            "s2omics.single_section.p6_roi_selection_rectangle": module(
                "s2omics.single_section.p6_roi_selection_rectangle",
                "roi_selection_for_single_section", "rectangle",
            ),
            "s2omics.single_section.p6_roi_selection_circle": module(
                "s2omics.single_section.p6_roi_selection_circle",
                "roi_selection_for_single_section", "circle",
            ),
        }

        with patch.dict(sys.modules, modules):
            rectangle = self.config(
                roi_shape="rectangle", roi_width_mm=5, roi_height_mm=4,
                rotation_seg=8, num_roi=2,
            )
            run_roi_selection_pipeline(rectangle, self.root / "rectangle")
            rectangle_call = next(call for call in calls if call[0] == "rectangle")
            self.assertEqual(rectangle_call[2]["roi_size"], [5, 4])
            self.assertEqual(rectangle_call[2]["rotation_seg"], 8)
            self.assertEqual(rectangle_call[2]["num_roi"], 2)

            calls.clear()
            circle = self.config(roi_shape="circle", roi_radius_mm=1.25, num_roi=3)
            run_roi_selection_pipeline(circle, self.root / "circle")
            circle_call = next(call for call in calls if call[0] == "circle")
            self.assertEqual(circle_call[2]["roi_size"], [1.25, 1.25])
            self.assertNotIn("rotation_seg", circle_call[2])
            self.assertEqual(circle_call[2]["num_roi"], 3)

    def test_failure_is_recorded_and_batch_continues(self):
        output = self.root / "output"
        samples = [self.config(sample_id="first"), self.config(sample_id="second")]

        def pipeline(config, _output_dir):
            if config.sample_id == "first":
                raise RuntimeError("expected failure")

        with patch("s2omics.batch.run_roi_selection_pipeline", side_effect=pipeline):
            code = run_batch(samples, output)

        self.assertEqual(code, 1)
        statuses = {
            path.parent.name: json.loads(path.read_text(encoding="utf-8"))["status"]
            for path in output.glob("*/run_status.json")
        }
        self.assertEqual(statuses, {"first": "failed", "second": "success"})
        summary = next(output.glob("batch_summary_*.csv"))
        with summary.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["status"] for row in rows], ["failed", "success"])


if __name__ == "__main__":
    unittest.main()
