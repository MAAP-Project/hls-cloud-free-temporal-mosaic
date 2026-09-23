"""Bounded synthetic and small real-data compute profiling for the exporter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import dask
import dask.array as dask_array
import dask.array.core as dask_array_core
import dask.base as dask_base
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from dask.callbacks import Callback
from pyproj import CRS
from rustac import DuckdbClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from main import NODATA, NativeGrid  # noqa: E402


class TaskCounter(Callback):
    """Count Dask tasks executed in one isolated profiling process."""

    def __init__(self) -> None:
        self.tasks = 0
        self.task_prefixes: Counter[str] = Counter()
        super().__init__()

    def _pretask(self, key: Any, dask_graph: Any, state: Any) -> None:
        del dask_graph, state
        self.tasks += 1
        key_name = key[0] if isinstance(key, tuple) else key
        self.task_prefixes[str(key_name).split("-")[0]] += 1


class SearchCounter:
    """Delegate DuckDB access while counting archive queries."""

    def __init__(self, client: Any, *, archive_https: bool = False) -> None:
        self.client = client
        self.archive_https = archive_https
        self.calls = 0

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        href = args[0] if args else kwargs.get("href")
        if (
            self.archive_https
            and isinstance(href, str)
            and href.startswith("s3://nasa-maap-data-store/")
        ):
            href = (
                "https://nasa-maap-data-store.s3.us-west-2.amazonaws.com/"
                + href.removeprefix("s3://nasa-maap-data-store/")
            )
            if args:
                args = (href, *args[1:])
            else:
                kwargs["href"] = href
        return self.client.search(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class DiagnosticHandler(logging.Handler):
    """Collect lazycogs debug messages without recording URLs or credentials."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "read_chunk_async" in message or "duckdb_client.search" in message:
            self.messages.append(message)


def _metadata_grid(size: int) -> NativeGrid:
    """Build a deterministic native-grid description for synthetic data."""
    transform = Affine(30, 0, 500000, 0, -30, 5100000)
    return NativeGrid(
        crs=CRS.from_epsg(32615),
        transform=transform,
        shape=(size, size),
        bbox=(500000, 5100000 - size * 30, 500000 + size * 30, 5100000),
        resolution=30,
    )


def _synthetic_inputs(
    size: int, time_count: int, bands: list[str]
) -> tuple[xr.DataArray, xr.DataArray, NativeGrid, list[bool]]:
    """Create deterministic chunked spectral and shared-Fmask inputs."""
    grid = _metadata_grid(size)
    y = grid.transform.f + (np.arange(size) + 0.5) * grid.transform.e
    x = grid.transform.c + (np.arange(size) + 0.5) * grid.transform.a
    times = np.datetime64("2025-05-01") + np.arange(time_count).astype("timedelta64[D]")
    values = np.empty((time_count, len(bands), size, size), dtype=np.int16)
    row_coords = np.arange(size, dtype=np.int32)[:, None]
    cols = np.arange(size, dtype=np.int32)[None, :]
    for time_index in range(time_count):
        for band_index in range(len(bands)):
            values[time_index, band_index] = (
                (time_index * 37 + band_index * 101 + row_coords * 3 + cols) % 1800
            ) - 900
    values[(values % 23) == 0] = NODATA
    spectral = xr.DataArray(
        dask_array.from_array(
            values,
            chunks=(min(4, time_count), 1, min(256, size), min(256, size)),
        ),
        dims=("time", "band", "y", "x"),
        coords={"time": times, "band": bands, "y": y, "x": x},
    )

    fmask_calls: list[bool] = []
    chunk = min(256, size)
    fmask_rows: list[Any] = []
    for row_start in range(0, size, chunk):
        row_blocks = []
        row_size = min(chunk, size - row_start)
        for col_start in range(0, size, chunk):
            col_size = min(chunk, size - col_start)

            def load_fmask(
                row_start: int = row_start,
                col_start: int = col_start,
                row_size: int = row_size,
                col_size: int = col_size,
            ) -> np.ndarray:
                fmask_calls.append(True)
                result = np.zeros((time_count, row_size, col_size), dtype=np.uint8)
                for t in range(time_count):
                    if (t + row_start // chunk + col_start // chunk) % 5 == 0:
                        result[t, :, ::3] = 4
                    if (t + row_start // chunk) % 7 == 0:
                        result[t, 1::3, :] |= 8
                return result

            row_blocks.append(
                dask_array.from_delayed(
                    dask.delayed(load_fmask)(),
                    shape=(time_count, row_size, col_size),
                    dtype=np.uint8,
                )
            )
        fmask_rows.append(dask_array.concatenate(row_blocks, axis=2))
    fmask = xr.DataArray(
        dask_array.concatenate(fmask_rows, axis=1),
        dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
    )
    return spectral, fmask, grid, fmask_calls


def _baseline_module() -> ModuleType:
    """Load the main-branch exporter without changing the working tree."""
    source = subprocess.check_output(
        ["git", "show", "main:main.py"], cwd=ROOT, text=True
    )
    path = ROOT / "profiling" / "_baseline_main.py"
    path.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location("baseline_main", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load baseline main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        path.unlink(missing_ok=True)


def _timed_export(
    module: ModuleType,
    composite: xr.DataArray,
    grid: NativeGrid,
    bands: list[str],
    workers: int,
    source_item_ids: list[str],
) -> dict[str, Any]:
    """Run one exporter while separating Dask work from COG writes."""
    compute_calls: list[dict[str, Any]] = []
    array_compute_calls: list[dict[str, Any]] = []
    cog_writes: list[float] = []
    real_compute = dask.compute
    real_base_compute = dask_base.compute
    real_array_compute = dask_array_core.compute
    real_to_raster = type(composite.rio).to_raster

    def timed_compute(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = real_compute(*args, **kwargs)
        compute_calls.append(
            {
                "args": len(args),
                "seconds": time.perf_counter() - started,
                "kwargs": kwargs,
            }
        )
        return result

    def timed_array_compute(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = real_array_compute(*args, **kwargs)
        array_compute_calls.append(
            {
                "args": len(args),
                "seconds": time.perf_counter() - started,
                "kwargs": kwargs,
            }
        )
        return result

    def timed_to_raster(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = real_to_raster(self, *args, **kwargs)
        cog_writes.append(time.perf_counter() - started)
        return result

    dask.compute = timed_compute
    dask_base.compute = timed_array_compute
    dask_array_core.compute = timed_array_compute
    type(composite.rio).to_raster = timed_to_raster
    try:
        with tempfile.TemporaryDirectory(prefix="hls-profile-") as directory:
            started = time.perf_counter()
            module.export_outputs(
                composite,
                tile_id="T15TYJ",
                grid=grid,
                bands=bands,
                start_datetime=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end_datetime=datetime(2025, 5, 31, 23, 59, 59, tzinfo=timezone.utc),
                output_dir=Path(directory),
                source_item_ids=source_item_ids,
                dask_workers=workers,
            )
            elapsed = time.perf_counter() - started
            cog_layouts = []
            for band in bands:
                with rasterio.open(Path(directory) / f"{band}.tif") as dataset:
                    cog_layouts.append(dataset.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"))
    finally:
        dask.compute = real_compute
        dask_base.compute = real_base_compute
        dask_array_core.compute = real_array_compute
        type(composite.rio).to_raster = real_to_raster

    return {
        "elapsed_seconds": elapsed,
        "dask_compute_calls": compute_calls,
        "dask_array_compute_calls": array_compute_calls,
        "cog_write_seconds": cog_writes,
        "cog_layouts": cog_layouts,
    }


def profile_synthetic(
    mode: str, size: int, time_count: int, bands: list[str], workers: int
) -> dict[str, Any]:
    """Profile baseline or fixed export on one deterministic synthetic case."""
    module = main if mode == "fixed" else _baseline_module()
    spectral, fmask, grid, fmask_calls = _synthetic_inputs(size, time_count, bands)
    graph_started = time.perf_counter()
    composite = module.create_composite(spectral, fmask)
    graph_seconds = time.perf_counter() - graph_started
    task_counter = TaskCounter()
    with task_counter:
        export = _timed_export(module, composite, grid, bands, workers, [])
    return {
        "kind": "synthetic",
        "mode": mode,
        "size": size,
        "time_count": time_count,
        "bands": bands,
        "workers_requested": workers,
        "graph_seconds": graph_seconds,
        "fmask_materializations": len(fmask_calls),
        "dask_tasks": task_counter.tasks,
        "task_prefixes": dict(task_counter.task_prefixes),
        "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "export": export,
    }


def _subset_grid(grid: NativeGrid, size: int) -> NativeGrid:
    """Select a centered native-grid subset without resampling."""
    row_offset = (grid.shape[0] - size) // 2
    col_offset = (grid.shape[1] - size) // 2
    transform = grid.transform * Affine.translation(col_offset, row_offset)
    corners = [
        transform * point for point in ((0, 0), (size, 0), (0, size), (size, size))
    ]
    return NativeGrid(
        crs=grid.crs,
        transform=transform,
        shape=(size, size),
        bbox=(
            min(x for x, _ in corners),
            min(y for _, y in corners),
            max(x for x, _ in corners),
            max(y for _, y in corners),
        ),
        resolution=grid.resolution,
    )


def profile_real(
    tile: str, start: str, end: str, size: int, bands: list[str], workers: int
) -> dict[str, Any]:
    """Profile one bounded native-grid HLS subset over native HTTPS paths."""
    logger = logging.getLogger("lazycogs")
    handler = DiagnosticHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        started = time.perf_counter()
        start_dt = main.parse_datetime_utc(start)
        end_dt = main.parse_datetime_utc(end)
        client = SearchCounter(DuckdbClient(), archive_https=True)
        setup_seconds = time.perf_counter() - started

        discovery_started = time.perf_counter()
        items_by_collection = {
            collection: main.discover_hls_items(
                collection,
                tile_id=tile,
                start_datetime=start_dt,
                end_datetime=end_dt,
                duckdb_client=cast(DuckdbClient, client),
            )
            for collection in main.COLLECTION_BAND_ALIASES
        }
        discovery_seconds = time.perf_counter() - discovery_started
        discovery_searches = client.calls
        all_items = [item for items in items_by_collection.values() for item in items]
        if not all_items:
            raise RuntimeError("No real HLS items matched the bounded profile query")

        store_started = time.perf_counter()
        store_kwargs = main.build_store_config(False)
        grid = main.native_grid_for_items(
            all_items, bands=bands, store_kwargs=store_kwargs
        )
        subset = _subset_grid(grid, size)
        store_seconds = time.perf_counter() - store_started

        open_started = time.perf_counter()
        spectral_arrays = []
        fmask_arrays = []
        for collection, items in items_by_collection.items():
            if not items:
                continue
            spectral, fmask = main.open_hls_collection(
                collection,
                items=items,
                grid=subset,
                start_datetime=start_dt,
                end_datetime=end_dt,
                bands=bands,
                store_kwargs=store_kwargs,
                duckdb_client=cast(DuckdbClient, client),
            )
            spectral_arrays.append(spectral)
            fmask_arrays.append(fmask)
        spectral_stack = xr.concat(spectral_arrays, dim="time").sortby("time")
        fmask_stack = xr.concat(fmask_arrays, dim="time").sortby("time")
        spectral_stack, fmask_stack = xr.align(
            spectral_stack, fmask_stack, join="exact"
        )
        open_seconds = time.perf_counter() - open_started
        open_searches = client.calls - discovery_searches

        graph_started = time.perf_counter()
        composite = main.create_composite(spectral_stack, fmask_stack)
        graph_seconds = time.perf_counter() - graph_started
        task_counter = TaskCounter()
        export_started = time.perf_counter()
        with task_counter:
            export = _timed_export(
                main,
                composite,
                subset,
                bands,
                workers,
                [item["id"] for item in all_items],
            )
        export["elapsed_seconds"] = time.perf_counter() - export_started
        read_bands = Counter(
            message.partition("bands=")[2].partition(" datetime")[0]
            for message in handler.messages
            if "read_chunk_async" in message
        )
        search_bands = Counter(
            message.partition("bands=")[2].partition(" datetime")[0]
            for message in handler.messages
            if "duckdb_client.search " in message
            and "returned unexpected" not in message
        )
        return {
            "kind": "real",
            "tile": tile,
            "start": start,
            "end": end,
            "size": size,
            "bands": bands,
            "workers_requested": workers,
            "auth_mode": "authenticated_https_assets_public_https_archive",
            "auth_config_present": {
                key: bool(os.getenv(key))
                for key in ("MAAP_PGT", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
            },
            "setup_seconds": setup_seconds,
            "discovery_seconds": discovery_seconds,
            "store_and_header_seconds": store_seconds,
            "open_and_stack_seconds": open_seconds,
            "graph_seconds": graph_seconds,
            "item_counts": {
                collection: len(items)
                for collection, items in items_by_collection.items()
            },
            "searches": {
                "discovery": discovery_searches,
                "lazycogs_startup": open_searches,
                "tracked_total": client.calls,
            },
            "lazycogs_diagnostics": {
                "read_chunk_async": sum(
                    "read_chunk_async" in message for message in handler.messages
                ),
                "read_chunk_bands": dict(read_bands),
                "duckdb_search": sum(search_bands.values()),
                "duckdb_search_bands": dict(search_bands),
            },
            "dimensions": {
                key: int(value) for key, value in spectral_stack.sizes.items()
            },
            "dask_tasks": task_counter.tasks,
            "task_prefixes": dict(task_counter.task_prefixes),
            "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "export": export,
        }
    finally:
        logger.removeHandler(handler)


def main_cli() -> None:
    """Run one bounded profiling case and print one JSON result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "fixed", "real"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--time-count", type=int, default=16)
    parser.add_argument("--bands", default="red,green,blue")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tile", default="T15TYJ")
    parser.add_argument("--start", default="2025-05-01T00:00:00Z")
    parser.add_argument("--end", default="2025-05-16T00:00:00Z")
    args = parser.parse_args()
    bands = [band for band in args.bands.split(",") if band]
    if args.mode == "real":
        result = profile_real(
            args.tile, args.start, args.end, args.size, bands, args.workers
        )
    else:
        result = profile_synthetic(
            args.mode, args.size, args.time_count, bands, args.workers
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main_cli()
