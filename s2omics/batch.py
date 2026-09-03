"""Batch orchestration for independent single-section ROI selection runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs"}
SUPPORTED_MODELS = {"uni", "virchow", "gigapath"}
SUPPORTED_CLUSTERING_METHODS = {
    "kmeans", "fcm", "agglo", "bisect", "birch", "louvain", "leiden"
}

MANIFEST_COLUMNS = {
    "image_path", "sample_id", "pixel_size_um", "foundation_model",
    "ckpt_path", "device", "batch_size", "num_workers", "down_samp_step",
    "density_thresh", "clean_background", "min_size", "patch_size",
    "clustering_method", "n_clusters", "clustering_resolution",
    "target_n_clusters", "roi_shape", "roi_width_mm", "roi_height_mm",
    "roi_radius_mm", "rotation_seg", "num_roi", "optimal_roi_thres",
    "fusion_weight_scale", "fusion_weight_coverage",
    "fusion_weight_balance", "emphasize_clusters", "discard_clusters",
    "prior_preference", "notes",
}


@dataclass(frozen=True)
class SampleConfig:
    image_path: Path
    sample_id: str
    pixel_size_um: float
    foundation_model: str
    ckpt_path: Path
    device: str
    batch_size: int
    num_workers: int
    down_samp_step: int
    density_thresh: float
    clean_background: bool
    min_size: int
    patch_size: int
    clustering_method: str
    n_clusters: int
    clustering_resolution: float
    target_n_clusters: int
    roi_shape: str
    roi_width_mm: float
    roi_height_mm: float
    roi_radius_mm: float
    rotation_seg: int
    num_roi: int
    optimal_roi_thres: float
    fusion_weights: tuple[float, float, float]
    emphasize_clusters: tuple[int, ...]
    discard_clusters: tuple[int, ...]
    prior_preference: float
    resolved_sample_id: str | None = None

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["image_path"] = str(self.image_path)
        values["ckpt_path"] = str(self.ckpt_path)
        values["fusion_weights"] = list(self.fusion_weights)
        values["emphasize_clusters"] = list(self.emphasize_clusters)
        values["discard_clusters"] = list(self.discard_clusters)
        return values


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _as_int_list(value: Any, field: str) -> tuple[int, ...]:
    if _is_blank(value):
        return ()
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        text = str(value).strip()
        if text.startswith("["):
            parts = json.loads(text)
        else:
            parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    try:
        return tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a semicolon-separated list of integers") from exc


def _sample_id(value: Any, image_path: Path) -> str:
    raw = image_path.stem if _is_blank(value) else str(value).strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw)
    safe = re.sub(r"\s+", "_", safe).strip(" .")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Cannot derive a valid sample_id from {raw!r}")
    return safe


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_manifest_rows(path: Path, sheet_name: str | int = 0) -> list[dict[str, Any]]:
    """Load CSV, TSV, or XLSX rows without coercing empty cells into defaults."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    elif suffix == ".xlsx":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "XLSX manifests require pandas and openpyxl; install requirements.txt"
            ) from exc
        try:
            frame = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
        except ImportError as exc:
            raise RuntimeError(
                "XLSX manifests require openpyxl; install requirements.txt"
            ) from exc
        rows = frame.to_dict(orient="records")
    else:
        raise ValueError("Manifest must be a .csv, .tsv, or .xlsx file")

    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    normalized = []
    for row_number, row in enumerate(rows, start=2):
        cleaned = {str(key).strip(): value for key, value in row.items() if key is not None}
        unknown = sorted(
            key for key, value in cleaned.items()
            if key not in MANIFEST_COLUMNS and not _is_blank(value)
        )
        if unknown:
            raise ValueError(
                f"Manifest row {row_number} has unknown columns: {', '.join(unknown)}"
            )
        cleaned["__row_number__"] = row_number
        normalized.append(cleaned)
    return normalized


def discover_svs(input_dir: Path, recursive: bool = False) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    iterator: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() == ".svs"),
        key=lambda path: str(path).lower(),
    )
    if not images:
        scope = "recursively" if recursive else "in the directory"
        raise ValueError(f"No .svs files found {scope}: {input_dir}")
    return images


def defaults_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pixel_size_um": args.pixel_size_um,
        "foundation_model": args.foundation_model,
        "ckpt_path": args.ckpt_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "down_samp_step": args.down_samp_step,
        "density_thresh": args.density_thresh,
        "clean_background": args.clean_background,
        "min_size": args.min_size,
        "patch_size": args.patch_size,
        "clustering_method": args.clustering_method,
        "n_clusters": args.n_clusters,
        "clustering_resolution": args.clustering_resolution,
        "target_n_clusters": args.target_n_clusters,
        "roi_shape": args.roi_shape,
        "roi_width_mm": args.roi_width_mm,
        "roi_height_mm": args.roi_height_mm,
        "roi_radius_mm": args.roi_radius_mm,
        "rotation_seg": args.rotation_seg,
        "num_roi": args.num_roi,
        "optimal_roi_thres": args.optimal_roi_thres,
        "fusion_weight_scale": args.fusion_weights[0],
        "fusion_weight_coverage": args.fusion_weights[1],
        "fusion_weight_balance": args.fusion_weights[2],
        "emphasize_clusters": args.emphasize_clusters,
        "discard_clusters": args.discard_clusters,
        "prior_preference": args.prior_preference,
    }


def config_from_row(
    row: Mapping[str, Any], defaults: Mapping[str, Any], base_dir: Path
) -> SampleConfig:
    row_number = row.get("__row_number__", "?")

    def value(name: str) -> Any:
        current = row.get(name)
        return defaults.get(name) if _is_blank(current) else current

    if _is_blank(row.get("image_path")):
        raise ValueError(f"Manifest row {row_number} is missing image_path")
    image_path = _resolve_path(row["image_path"], base_dir)

    pixel_size_value = value("pixel_size_um")
    if _is_blank(pixel_size_value):
        raise ValueError(
            f"Manifest row {row_number} needs pixel_size_um or --pixel-size-um"
        )

    ckpt_value = value("ckpt_path")
    if _is_blank(ckpt_value):
        raise ValueError(f"Manifest row {row_number} needs ckpt_path or --ckpt-path")
    ckpt_base = base_dir if not _is_blank(row.get("ckpt_path")) else Path.cwd()

    fusion_weights = (
        float(value("fusion_weight_scale")),
        float(value("fusion_weight_coverage")),
        float(value("fusion_weight_balance")),
    )

    config = SampleConfig(
        image_path=image_path,
        sample_id=_sample_id(row.get("sample_id"), image_path),
        pixel_size_um=float(pixel_size_value),
        foundation_model=str(value("foundation_model")).strip().lower(),
        ckpt_path=_resolve_path(ckpt_value, ckpt_base),
        device=str(value("device")).strip(),
        batch_size=int(value("batch_size")),
        num_workers=int(value("num_workers")),
        down_samp_step=int(value("down_samp_step")),
        density_thresh=float(value("density_thresh")),
        clean_background=_as_bool(value("clean_background"), "clean_background"),
        min_size=int(value("min_size")),
        patch_size=int(value("patch_size")),
        clustering_method=str(value("clustering_method")).strip().lower(),
        n_clusters=int(value("n_clusters")),
        clustering_resolution=float(value("clustering_resolution")),
        target_n_clusters=int(value("target_n_clusters")),
        roi_shape=str(value("roi_shape")).strip().lower(),
        roi_width_mm=float(value("roi_width_mm")),
        roi_height_mm=float(value("roi_height_mm")),
        roi_radius_mm=float(value("roi_radius_mm")),
        rotation_seg=int(value("rotation_seg")),
        num_roi=int(value("num_roi")),
        optimal_roi_thres=float(value("optimal_roi_thres")),
        fusion_weights=fusion_weights,
        emphasize_clusters=_as_int_list(value("emphasize_clusters"), "emphasize_clusters"),
        discard_clusters=_as_int_list(value("discard_clusters"), "discard_clusters"),
        prior_preference=float(value("prior_preference")),
    )
    validate_config(config, row_number=row_number)
    return config


def validate_config(config: SampleConfig, row_number: Any = "?") -> None:
    label = f"row {row_number} ({config.sample_id})"
    if not config.image_path.is_file():
        raise FileNotFoundError(f"Image for {label} not found: {config.image_path}")
    if config.image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix for {label}: {config.image_path.suffix}")
    if not config.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path for {label} not found: {config.ckpt_path}")
    if config.foundation_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported foundation_model for {label}: {config.foundation_model}")
    if config.clustering_method not in SUPPORTED_CLUSTERING_METHODS:
        raise ValueError(f"Unsupported clustering_method for {label}: {config.clustering_method}")
    if config.pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be greater than 0 for {label}")
    for name in ("batch_size", "down_samp_step", "min_size", "patch_size", "n_clusters"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be greater than 0 for {label}")
    if config.num_workers < 0:
        raise ValueError(f"num_workers cannot be negative for {label}")
    if not 0 < config.target_n_clusters <= config.n_clusters:
        raise ValueError(f"target_n_clusters must be between 1 and n_clusters for {label}")
    if config.clustering_resolution <= 0:
        raise ValueError(f"clustering_resolution must be greater than 0 for {label}")
    if config.roi_shape not in {"rectangle", "circle"}:
        raise ValueError(f"roi_shape must be rectangle or circle for {label}")
    if config.roi_shape == "rectangle" and (
        config.roi_width_mm <= 0 or config.roi_height_mm <= 0
    ):
        raise ValueError(f"Rectangle width and height must be greater than 0 for {label}")
    if config.roi_shape == "circle" and config.roi_radius_mm <= 0:
        raise ValueError(f"Circle radius must be greater than 0 for {label}")
    if config.rotation_seg <= 0:
        raise ValueError(f"rotation_seg must be greater than 0 for {label}")
    if config.num_roi < 0:
        raise ValueError(f"num_roi cannot be negative for {label}")
    if config.optimal_roi_thres < 0:
        raise ValueError(f"optimal_roi_thres cannot be negative for {label}")
    if any(weight < 0 for weight in config.fusion_weights) or sum(config.fusion_weights) <= 0:
        raise ValueError(f"fusion weights must be non-negative with a positive sum for {label}")


def allocate_output_names(
    samples: Sequence[SampleConfig], output_root: Path
) -> list[SampleConfig]:
    used = {
        path.name.lower()
        for path in output_root.iterdir()
        if path.is_dir()
    } if output_root.exists() else set()
    resolved = []
    for sample in samples:
        candidate = sample.sample_id
        suffix = 0
        while candidate.lower() in used:
            suffix += 1
            candidate = f"{sample.sample_id}-{suffix}"
        used.add(candidate.lower())
        resolved.append(replace(sample, resolved_sample_id=candidate))
    return resolved


def run_roi_selection_pipeline(config: SampleConfig, output_dir: Path) -> None:
    """Run all six S2Omics stages for one independent section."""
    from .p1_histology_preprocess import histology_preprocess
    from .p2_superpixel_quality_control import superpixel_quality_control
    from .p3_feature_extraction import histology_feature_extraction
    from .single_section.p4_get_histology_segmentation import get_histology_segmentation
    from .single_section.p5_merge_over_clusters import merge_over_clusters

    prefix = str(output_dir) + os.sep
    save_folder = str(output_dir / "S2Omics_output")

    histology_preprocess(
        prefix,
        show_image=False,
        raw_image_path=str(config.image_path),
        pixel_size_raw=config.pixel_size_um,
    )
    superpixel_quality_control(
        prefix,
        save_folder,
        density_thresh=config.density_thresh,
        clean_background_flag=config.clean_background,
        min_size=config.min_size,
        patch_size=config.patch_size,
        show_image=False,
    )
    histology_feature_extraction(
        prefix,
        save_folder,
        foundation_model=config.foundation_model,
        ckpt_path=str(config.ckpt_path),
        device=config.device,
        batch_size=config.batch_size,
        down_samp_step=config.down_samp_step,
        num_workers=config.num_workers,
    )
    get_histology_segmentation(
        prefix,
        save_folder,
        foundation_model=config.foundation_model,
        down_samp_step=config.down_samp_step,
        clustering_method=config.clustering_method,
        n_clusters=config.n_clusters,
        resolution=config.clustering_resolution,
    )
    merge_over_clusters(
        prefix,
        save_folder,
        target_n_clusters=config.target_n_clusters,
    )

    roi_kwargs = dict(
        down_samp_step=config.down_samp_step,
        num_roi=config.num_roi,
        optimal_roi_thres=config.optimal_roi_thres,
        fusion_weights=list(config.fusion_weights),
        emphasize_clusters=list(config.emphasize_clusters),
        discard_clusters=list(config.discard_clusters),
        prior_preference=config.prior_preference,
    )
    if config.roi_shape == "rectangle":
        from .single_section.p6_roi_selection_rectangle import (
            roi_selection_for_single_section,
        )

        roi_selection_for_single_section(
            prefix,
            save_folder,
            roi_size=[config.roi_width_mm, config.roi_height_mm],
            rotation_seg=config.rotation_seg,
            **roi_kwargs,
        )
    else:
        from .single_section.p6_roi_selection_circle import (
            roi_selection_for_single_section,
        )

        roi_selection_for_single_section(
            prefix,
            save_folder,
            roi_size=[config.roi_radius_mm, config.roi_radius_mm],
            **roi_kwargs,
        )


class _Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _unique_batch_file(output_root: Path, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"{stem}_{timestamp}.csv"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{stem}_{timestamp}-{suffix}.csv"
    return candidate


def _manifest_record(config: SampleConfig) -> dict[str, Any]:
    values = config.serializable()
    weights = values.pop("fusion_weights")
    values["fusion_weight_scale"] = weights[0]
    values["fusion_weight_coverage"] = weights[1]
    values["fusion_weight_balance"] = weights[2]
    values["emphasize_clusters"] = ";".join(map(str, config.emphasize_clusters))
    values["discard_clusters"] = ";".join(map(str, config.discard_clusters))
    return values


def run_batch(
    samples: Sequence[SampleConfig], output_root: Path, *, dry_run: bool = False,
    fail_fast: bool = False
) -> int:
    resolved = allocate_output_names(samples, output_root)
    if dry_run:
        payload = [sample.serializable() for sample in resolved]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_manifest = _unique_batch_file(output_root, "resolved_manifest")
    summary_path = _unique_batch_file(output_root, "batch_summary")
    _write_csv(resolved_manifest, [_manifest_record(sample) for sample in resolved])

    summaries: list[dict[str, Any]] = []
    failures = 0
    for sample in resolved:
        output_dir = output_root / str(sample.resolved_sample_id)
        output_dir.mkdir(parents=False, exist_ok=False)
        config_path = output_dir / "resolved_config.json"
        status_path = output_dir / "run_status.json"
        log_path = output_dir / "run.log"
        _write_json(config_path, sample.serializable())
        started_at = _now()
        status: dict[str, Any] = {
            "status": "running",
            "sample_id": sample.sample_id,
            "resolved_sample_id": sample.resolved_sample_id,
            "started_at": started_at,
        }
        _write_json(status_path, status)

        error = ""
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                stdout_tee = _Tee(sys.stdout, log_handle)
                stderr_tee = _Tee(sys.stderr, log_handle)
                with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                    print(f"[{started_at}] Starting {sample.resolved_sample_id}")
                    run_roi_selection_pipeline(sample, output_dir)
            result = "success"
        except Exception:
            failures += 1
            result = "failed"
            error = traceback.format_exc()
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write("\n" + error)
            print(f"FAILED {sample.resolved_sample_id}: {error.splitlines()[-1]}", file=sys.stderr)

        finished_at = _now()
        status.update({"status": result, "finished_at": finished_at, "error": error})
        _write_json(status_path, status)
        summaries.append({
            "sample_id": sample.sample_id,
            "resolved_sample_id": sample.resolved_sample_id,
            "image_path": str(sample.image_path),
            "roi_shape": sample.roi_shape,
            "roi_size_mm": (
                f"{sample.roi_width_mm}x{sample.roi_height_mm}"
                if sample.roi_shape == "rectangle" else f"radius={sample.roi_radius_mm}"
            ),
            "num_roi": sample.num_roi,
            "status": result,
            "output_dir": str(output_dir),
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
        })
        _write_csv(summary_path, summaries)
        if failures and fail_fast:
            break

    print(f"Batch summary: {summary_path}")
    print(f"Completed: {len(summaries) - failures}; failed: {failures}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run S2Omics ROI selection for independent histology sections."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Directory containing .svs files")
    source.add_argument("--manifest", type=Path, help="CSV, TSV, or XLSX sample manifest")
    parser.add_argument("--manifest-sheet", default=0, help="XLSX sheet name or zero-based index")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover .svs files")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pixel-size-um", type=float)
    parser.add_argument("--foundation-model", default="uni")
    parser.add_argument("--ckpt-path", default="./checkpoints/uni/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--down-samp-step", type=int, default=10)
    parser.add_argument("--density-thresh", type=float, default=100)
    parser.add_argument("--clean-background", action="store_true")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--clustering-method", default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--clustering-resolution", type=float, default=1.0)
    parser.add_argument("--target-n-clusters", type=int, default=15)
    parser.add_argument("--roi-shape", choices=("rectangle", "circle"), default="rectangle")
    parser.add_argument("--roi-width-mm", type=float, default=6.5)
    parser.add_argument("--roi-height-mm", type=float, default=6.5)
    parser.add_argument("--roi-radius-mm", type=float, default=0.5)
    parser.add_argument("--rotation-seg", type=int, default=6)
    parser.add_argument(
        "--num-roi", type=int, default=0,
        help="Number of ROIs; 0 lets S2Omics determine the number automatically",
    )
    parser.add_argument("--optimal-roi-thres", type=float, default=0.03)
    parser.add_argument("--fusion-weights", type=float, nargs=3, default=(0.33, 0.33, 0.33))
    parser.add_argument("--emphasize-clusters", type=int, nargs="*", default=())
    parser.add_argument("--discard-clusters", type=int, nargs="*", default=())
    parser.add_argument("--prior-preference", type=float, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def samples_from_args(args: argparse.Namespace) -> list[SampleConfig]:
    defaults = defaults_from_args(args)
    if args.input_dir:
        if args.pixel_size_um is None:
            raise ValueError("--pixel-size-um is required with --input-dir")
        rows = [
            {"image_path": str(path), "sample_id": path.stem}
            for path in discover_svs(args.input_dir.resolve(), args.recursive)
        ]
        base_dir = Path.cwd()
    else:
        manifest = args.manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        sheet: str | int = args.manifest_sheet
        if isinstance(sheet, str) and sheet.isdigit():
            sheet = int(sheet)
        rows = load_manifest_rows(manifest, sheet)
        base_dir = manifest.parent
    return [config_from_row(row, defaults, base_dir) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        samples = samples_from_args(args)
        return run_batch(
            samples, args.output_root.resolve(), dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
