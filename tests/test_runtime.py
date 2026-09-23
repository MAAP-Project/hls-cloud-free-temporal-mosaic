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

import main  # noqa: E402
from main import (  # noqa: E402
    NODATA,
    NativeGrid,
    MaapEarthdataCredentialProvider,
    build_s3_store,
    create_composite,
    discover_hls_items,
    hls_geoparquet_hrefs,
    item_tile_id,
    export_outputs,
    item_id,
    native_grid_for_items,
    normalize_interval,
    open_hls_collection,
    parse_args,
)

UTC = timezone.utc


def test_parse_args_accepts_tile_contract_without_running_work():
    args = parse_args(
        [
            "--start_datetime",
            "2024-01-01T00:00:00Z",
            "--end_datetime",
            "2024-02-01T00:00:00Z",
            "--tile_id",
            "T15tyj",
            "--output_dir",
            "/tmp/output",
        ]
    )

    assert args.tile_id == "T15TYJ"
    assert not hasattr(args, "bbox")
    assert not hasattr(args, "crs")
    assert args.direct_bucket_access is False


@pytest.mark.parametrize("tile_id", ["15TYJ", "T1TYJ", "T15TY", "T15- YJ"])
def test_parse_args_rejects_invalid_tile_id(tile_id):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--start_datetime",
                "2024-01-01T00:00:00Z",
                "--end_datetime",
                "2024-02-01T00:00:00Z",
                "--tile_id",
                tile_id,
                "--output_dir",
                "/tmp/output",
            ]
        )


def test_normalize_interval_supports_exclusive_and_inclusive_day_end():
    start = datetime(2024, 12, 1, tzinfo=UTC)
    assert normalize_interval(start, datetime(2025, 1, 1, tzinfo=UTC)) == (
        start,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
    )
    assert normalize_interval(start, datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC))[
        1
    ] == datetime(2025, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "start,end",
    [
        ("2024-01-01T01:00:00+00:00", "2024-02-01T00:00:00+00:00"),
        ("2024-01-01T00:00:00+00:00", "2024-01-31T12:00:00+00:00"),
    ],
)
def test_normalize_interval_rejects_partial_days(start, end):
    with pytest.raises(ValueError):
        normalize_interval(datetime.fromisoformat(start), datetime.fromisoformat(end))


def test_item_id_is_stable_and_includes_composite_method():
    args = (
        "T15TYJ",
        datetime(2025, 5, 1, tzinfo=UTC),
        datetime(2025, 5, 31, 23, 59, 59, tzinfo=UTC),
    )
    assert item_id(*args) == "hls-composite-T15TYJ-20250501-20250531-lower-median-v1"
    assert item_id(*args) == item_id(*args)


def test_maap_credential_provider_refreshes_and_maps_obstore_schema():
    expiration = "2026-01-02T03:04:05+00:00"

    class FakeAws:
        def __init__(self):
            self.calls = []
            self.responses = [
                {
                    "accessKeyId": "access-1",
                    "secretAccessKey": "secret-1",
                    "sessionToken": "token-1",
                    "expiration": expiration,
                },
                {
                    "accessKeyId": "access-2",
                    "secretAccessKey": "secret-2",
                    "sessionToken": "token-2",
                    "expiration": "2026-01-02T04:04:05Z",
                },
            ]

        def earthdata_s3_credentials(self, endpoint_uri):
            self.calls.append(endpoint_uri)
            return self.responses.pop(0)

    class FakeMaap:
        def __init__(self):
            self.aws = FakeAws()

    client = FakeMaap()
    provider = MaapEarthdataCredentialProvider(client)

    assert provider() == {
        "access_key_id": "access-1",
        "secret_access_key": "secret-1",
        "token": "token-1",
        "expires_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    }
    assert provider()["access_key_id"] == "access-2"


def test_build_s3_store_uses_maap_provider_in_lp_region(monkeypatch):
    class StubProvider:
        def __call__(self):
            return {}

    provider = StubProvider()
    monkeypatch.setattr(main, "MaapEarthdataCredentialProvider", lambda: provider)

    store = build_s3_store()

    assert store.config["bucket"] == "lp-prod-protected"
    assert store.config["region"] == "us-west-2"
    assert store.credential_provider is provider


def _metadata_grid(epsg=32615, origin=(500000, 5100060), shape=(2, 2)):
    transform = Affine(30, 0, origin[0], 0, -30, origin[1])
    corners = [
        transform * point
        for point in ((0, 0), (shape[1], 0), (0, shape[0]), shape[::-1])
    ]
    return NativeGrid(
        CRS.from_epsg(epsg),
        transform,
        shape,
        (
            min(x for x, _ in corners),
            min(y for _, y in corners),
            max(x for x, _ in corners),
            max(y for _, y in corners),
        ),
        30,
    )


def test_native_grid_for_items_reads_one_representative_cog_header(monkeypatch):
    items = [_item("one"), _item("two", collection="HLSL30_2.0")]
    cog_grid = _metadata_grid(origin=(600000, 5200060), shape=(3660, 3660))
    calls = []

    def inspect(item, asset_name, store_kwargs):
        calls.append((item["id"], asset_name, store_kwargs))
        return cog_grid

    monkeypatch.setattr(main, "_inspect_cog_grid", inspect)
    store_kwargs = {"store": object()}

    assert (
        native_grid_for_items(items, bands=["red"], store_kwargs=store_kwargs)
        == cog_grid
    )
    assert [(item_id, asset_name) for item_id, asset_name, _ in calls] == [
        ("one", "B04")
    ]
    assert calls[0][2] is store_kwargs


def _item(
    item_id,
    collection="HLSS30_2.0",
    tile="T15TYJ",
    grid=None,
    dt="2025-05-01T12:00:00Z",
):
    grid = grid or _metadata_grid()
    return {
        "id": item_id,
        "collection": collection,
        "properties": {
            "datetime": dt,
            "proj:epsg": grid.crs.to_epsg(),
            "proj:shape": list(grid.shape),
            "proj:transform": list(grid.transform),
        },
        "assets": {
            "B04": {"href": "https://example/B04.tif"},
            "Fmask": {"href": "https://example/Fmask.tif"},
        },
        "tile": tile,
    }


class FakeSearch:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def search(self, href, **kwargs):
        self.calls.append((href, kwargs))
        return self.items


def test_item_tile_property_takes_precedence_over_id_fallback():
    item = {
        "id": "HLS.S30.T15TYK.2025123T120000.v2.0",
        "properties": {"tile_id": "t15tyj"},
    }
    assert item_tile_id(item) == "T15TYJ"


def test_hls_geoparquet_hrefs_enumerates_exact_monthly_partitions():
    hrefs = hls_geoparquet_hrefs(
        "HLSS30_2.0",
        datetime(2024, 12, 1, tzinfo=UTC),
        datetime(2025, 3, 1, tzinfo=UTC),
    )

    assert hrefs == [
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2024/month=12/HLSS30_2.0-2024-12.parquet",
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=1/HLSS30_2.0-2025-1.parquet",
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=2/HLSS30_2.0-2025-2.parquet",
    ]


def test_discovery_uses_exact_partitions_and_excludes_neighboring_tiles():
    matching = _item("HLS.S30.T15TYJ.2025123T120000.v2.0")
    neighbor = _item("HLS.S30.T15TYK.2025123T120000.v2.0", tile="T15TYK")
    client = FakeSearch([neighbor, matching, matching])

    items = discover_hls_items(
        "HLSS30_2.0",
        tile_id="t15tyj",
        start_datetime=datetime(2025, 5, 1, tzinfo=UTC),
        end_datetime=datetime(2025, 6, 1, tzinfo=UTC),
        duckdb_client=client,
    )

    assert [item["id"] for item in items] == [matching["id"]]
    assert "/year=2025/month=5/" in client.calls[0][0]
    assert "*.parquet" not in client.calls[0][0]
    assert client.calls[0][1]["filter"]["op"] == "like"
    assert "T15TYJ." in client.calls[0][1]["filter"]["args"][1]
    assert client.calls[0][1]["datetime"].endswith("23:59:59.999999Z")


def test_no_data_discovery_is_empty():
    client = FakeSearch([])
    assert (
        discover_hls_items(
            "HLSL30_2.0",
            tile_id="T15TYJ",
            start_datetime=datetime(2025, 5, 1, tzinfo=UTC),
            end_datetime=datetime(2025, 6, 1, tzinfo=UTC),
            duckdb_client=client,
        )
        == []
    )


def test_open_hls_collection_reads_each_matching_monthly_partition(monkeypatch):
    calls = []

    def open_partition(**kwargs):
        calls.append(kwargs)
        month = np.datetime64(
            "2025-05-01" if "month=5/" in kwargs["href"] else "2025-06-01"
        )
        shape = (1, 1, 1, 1)
        data = np.zeros(
            shape, dtype=np.uint8 if kwargs["bands"] == ["Fmask"] else np.int16
        )
        return xr.DataArray(
            data,
            dims=("time", "band", "y", "x"),
            coords={"time": [month], "band": kwargs["bands"], "y": [0], "x": [0]},
        )

    monkeypatch.setattr(main.lazycogs, "open", open_partition)
    items = [
        _item("may", dt="2025-05-01T12:00:00Z"),
        _item("june", dt="2025-06-01T12:00:00Z"),
    ]
    spectral, fmask = open_hls_collection(
        "HLSS30_2.0",
        items=items,
        grid=_metadata_grid(),
        start_datetime=datetime(2025, 5, 1, tzinfo=UTC),
        end_datetime=datetime(2025, 7, 1, tzinfo=UTC),
        bands=["red"],
        store_kwargs={"store": object()},
        duckdb_client=object(),
    )

    assert [call["href"] for call in calls] == [
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=5/HLSS30_2.0-2025-5.parquet",
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=5/HLSS30_2.0-2025-5.parquet",
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=6/HLSS30_2.0-2025-6.parquet",
        "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/HLSS30_2.0/year=2025/month=6/HLSS30_2.0-2025-6.parquet",
    ]
    assert list(spectral.time.dt.month.values) == [5, 6]
    assert list(fmask.time.dt.month.values) == [5, 6]


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


def test_create_composite_masks_clouds_and_preserves_lower_median():
    spectral, fmask, _ = synthetic_stacks()
    composite = create_composite(spectral.chunk({"time": 1}), fmask.chunk({"time": 1}))
    assert hasattr(composite.data, "compute")
    np.testing.assert_array_equal(
        composite.compute().sel(band="red").values,
        np.array([[1, NODATA], [5, 4]], dtype=np.int16),
    )


def test_export_writes_native_grid_and_stac_identity(tmp_path):
    spectral, fmask, transform = synthetic_stacks()
    composite = create_composite(spectral, fmask)
    grid = NativeGrid(
        CRS.from_epsg(32615), transform, (2, 2), (500000, 5100000, 500060, 5100060), 30
    )

    export_outputs(
        composite,
        bands=["red"],
        tile_id="T15TYJ",
        grid=grid,
        start_datetime=datetime(2024, 1, 1, tzinfo=UTC),
        end_datetime=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
        output_dir=tmp_path,
        source_item_ids=["HLS.S30.T15TYJ.2024001T000000.v2.0"],
    )

    with rasterio.open(tmp_path / "red.tif") as dataset:
        assert dataset.crs == CRS.from_epsg(32615)
        assert dataset.transform == transform
        assert dataset.shape == (2, 2)
        np.testing.assert_array_equal(dataset.read(1), [[1, NODATA], [5, 4]])

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    collection_path = tmp_path / next(
        link["href"] for link in catalog["links"] if link["rel"] == "child"
    )
    collection = json.loads(collection_path.read_text())
    item_path = collection_path.parent / next(
        link["href"] for link in collection["links"] if link["rel"] == "item"
    )
    item = json.loads(item_path.read_text())
    assert item["id"] == "hls-composite-T15TYJ-20240101-20240131-lower-median-v1"
    assert "https://stac-extensions.github.io/mgrs/v1.0.0/schema.json" in item[
        "stac_extensions"
    ]
    assert item["properties"]["mgrs:utm_zone"] == 15
    assert item["properties"]["mgrs:latitude_band"] == "T"
    assert item["properties"]["mgrs:grid_square"] == "YJ"
    assert item["properties"]["hls-composite:method"] == "lower-median-v1"
    assert item["properties"]["hls-composite:mask"] == (
        "Fmask bitmask 14 equals zero and spectral nodata is excluded"
    )
    assert item["properties"]["hls-composite:source_item_ids"] == [
        "HLS.S30.T15TYJ.2024001T000000.v2.0"
    ]
    assert item["properties"]["start_datetime"] == "2024-01-01T00:00:00Z"
    assert item["properties"]["end_datetime"] == "2024-01-31T23:59:59Z"
    assert item["properties"]["proj:shape"] == [2, 2]
    assert item["properties"]["proj:transform"] == list(transform)


def test_application_package_uses_tile_input_and_direct_access_default():
    application_package = (ROOT / "hls-cloud-free-temporal-mosaic.cwl").read_text()
    assert "--tile_id" in application_package
    assert "--bbox" not in application_package
    assert "--crs" not in application_package
    assert (
        "direct_bucket_access:\n        type: boolean\n        default: true"
        in application_package
    )
