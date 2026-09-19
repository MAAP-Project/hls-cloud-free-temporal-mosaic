import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import NODATA, create_composite, export_outputs, parse_args  # noqa: E402


def test_parse_args_accepts_named_inputs_without_running_work():
    args = parse_args(
        [
            "--start_datetime",
            "2024-01-01T00:00:00Z",
            "--end_datetime",
            "2024-01-31T23:59:59Z",
            "--bbox",
            "500000",
            "5000000",
            "500060",
            "5000060",
            "--crs",
            "EPSG:32615",
            "--output_dir",
            "/tmp/output",
        ]
    )

    assert args.bbox == [500000.0, 5000000.0, 500060.0, 5000060.0]
    assert args.crs == "EPSG:32615"
    assert args.direct_bucket_access is False


def run_wrapper_with_fake_uv(tmp_path, inputs):
    capture = tmp_path / "uv-args"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\nprintf '%s\\0' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ | {
        "CAPTURE": str(capture),
        "DIRECT_BUCKET_ACCESS": "false",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    subprocess.run(
        ["bash", str(ROOT / "run.sh"), *inputs],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    return capture.read_bytes().rstrip(b"\0").split(b"\0")


def test_wrapper_adapts_positional_inputs_and_skips_runtime_sync(tmp_path):
    args = run_wrapper_with_fake_uv(
        tmp_path,
        [
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            "500000 5000000 500060 5000060",
            "EPSG:32615",
        ],
    )

    assert args == [
        b"run",
        b"--no-sync",
        b"--no-dev",
        str(ROOT / "main.py").encode(),
        b"--start_datetime",
        b"2024-01-01T00:00:00Z",
        b"--end_datetime",
        b"2024-01-31T23:59:59Z",
        b"--bbox",
        b"500000",
        b"5000000",
        b"500060",
        b"5000060",
        b"--crs",
        b"EPSG:32615",
        b"--output_dir=output",
    ]


def test_wrapper_passes_named_inputs_through(tmp_path):
    args = run_wrapper_with_fake_uv(
        tmp_path,
        [
            "--start_datetime",
            "2024-01-01T00:00:00Z",
            "--end_datetime",
            "2024-01-31T23:59:59Z",
            "--bbox",
            "500000",
            "5000000",
            "500060",
            "5000060",
            "--crs",
            "EPSG:32615",
        ],
    )

    assert args[4:] == [
        b"--start_datetime",
        b"2024-01-01T00:00:00Z",
        b"--end_datetime",
        b"2024-01-31T23:59:59Z",
        b"--bbox",
        b"500000",
        b"5000000",
        b"500060",
        b"5000060",
        b"--crs",
        b"EPSG:32615",
        b"--output_dir=output",
    ]


def synthetic_stacks():
    transform = Affine(30, 0, 500000, 0, -30, 5100060)
    spectral = xr.DataArray(
        np.array(
            [
                [[[1, 2], [NODATA, 4]]],
                [[[3, NODATA], [5, 6]]],
            ],
            dtype=np.int16,
        ),
        dims=("time", "band", "y", "x"),
        coords={
            "time": ["2024-01-01", "2024-01-02"],
            "band": ["red"],
            "x": [500015, 500045],
            "y": [5100045, 5100015],
        },
        attrs={"spatial:transform": tuple(transform)},
    )
    fmask = xr.DataArray(
        np.array(
            [
                [[0, 8], [0, 0]],
                [[4, 0], [0, 0]],
            ],
            dtype=np.uint8,
        ),
        dims=("time", "y", "x"),
        coords={"time": spectral.time, "x": spectral.x, "y": spectral.y},
    )
    return spectral, fmask, transform


def test_create_composite_masks_clouds_and_nodata():
    spectral, fmask, _ = synthetic_stacks()

    composite = create_composite(spectral, fmask)

    np.testing.assert_array_equal(
        composite.sel(band="red").values,
        np.array([[1, NODATA], [5, 5]]),
    )


def test_export_writes_georeferenced_cog_and_self_contained_stac(tmp_path):
    spectral, fmask, transform = synthetic_stacks()
    composite = create_composite(spectral, fmask)
    bbox = (500000, 5100000, 500060, 5100060)

    export_outputs(
        composite,
        bands=["red"],
        bbox=bbox,
        start_datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2024, 1, 2, tzinfo=timezone.utc),
        output_dir=tmp_path,
        crs=CRS.from_epsg(32615),
    )

    with rasterio.open(tmp_path / "red.tif") as dataset:
        assert dataset.crs == CRS.from_epsg(32615)
        assert dataset.transform == transform
        np.testing.assert_array_equal(dataset.read(1), [[1, NODATA], [5, 5]])

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    item_href = next(link["href"] for link in catalog["links"] if link["rel"] == "item")
    item = json.loads((tmp_path / item_href).read_text())
    assert item["assets"]["red"]["href"] == "../red.tif"
    assert item["properties"]["proj:epsg"] == 32615
    assert item["properties"]["proj:shape"] == [2, 2]
    assert item["properties"]["proj:transform"] == list(transform)
