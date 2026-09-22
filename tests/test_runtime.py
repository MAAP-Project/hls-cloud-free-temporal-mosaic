import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
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
            "500000 5000000 500060 5000060",
            "--crs",
            "EPSG:32615",
            "--output_dir",
            "/tmp/output",
        ]
    )

    assert args.bbox == [500000.0, 5000000.0, 500060.0, 5000060.0]
    assert args.crs == "EPSG:32615"
    assert args.direct_bucket_access is False


@pytest.mark.parametrize("bbox", ["1 2 3", "1 2 3 4 5", "1 2 nope 4", "1 2 nan 4"])
def test_parse_args_rejects_invalid_bbox(bbox):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--start_datetime",
                "2024-01-01T00:00:00Z",
                "--end_datetime",
                "2024-01-31T23:59:59Z",
                "--bbox",
                bbox,
                "--crs",
                "EPSG:32615",
                "--output_dir",
                "/tmp/output",
            ]
        )


def test_application_package_invokes_main_with_direct_access_default():
    application_package = (ROOT / "hls-cloud-free-temporal-mosaic.cwl").read_text()

    assert "run.sh" not in application_package
    assert "/app/hls-cloud-free-temporal-mosaic/main.py" in application_package
    assert (
        "direct_bucket_access:\n        type: boolean\n        default: true"
        in application_package
    )


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
    collection_links = [link for link in catalog["links"] if link["rel"] == "child"]
    assert len(collection_links) == 1

    collection_href = collection_links[0]["href"]
    collection_path = tmp_path / collection_href
    collection = json.loads(collection_path.read_text())
    assert collection["id"] == "hls-cloud-free-temporal-mosaic"
    assert collection["title"] == "HLS Cloud-Free Temporal Mosaic"
    assert collection["license"] == "other"
    assert collection["extent"]["spatial"]["bbox"] == [[-180.0, -90.0, 180.0, 90.0]]
    assert collection["extent"]["temporal"]["interval"] == [[None, None]]
    assert collection["item_assets"] == {
        "red": {
            "description": "median red band value from cloud-free pixels in the temporal mosaic",
            "roles": ["data"],
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        }
    }
    assert {
        (link["rel"], link["href"])
        for link in collection["links"]
        if link["rel"] in {"documentation", "source"}
    } == {
        (
            "documentation",
            "https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic",
        ),
        ("source", "https://doi.org/10.5067/HLS/HLSL30.002"),
        ("source", "https://doi.org/10.5067/HLS/HLSS30.002"),
    }

    item_links = [link for link in collection["links"] if link["rel"] == "item"]
    assert len(item_links) == 1
    item = json.loads((collection_path.parent / item_links[0]["href"]).read_text())
    assert item["collection"] == collection["id"]
    assert item["assets"]["red"]["href"] == "../../red.tif"
    assert item["properties"]["proj:epsg"] == 32615
    assert item["properties"]["proj:shape"] == [2, 2]
    assert item["properties"]["proj:transform"] == list(transform)
