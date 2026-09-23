"""Create a cloud-free composite image from a native HLS tile."""

import argparse
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import dask
import lazycogs
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from obstore.store import HTTPStore, S3Store
from pyproj import CRS
from pystac import (
    Asset,
    Catalog,
    CatalogType,
    Collection,
    Extent,
    ItemAssetDefinition,
    Link,
    MediaType,
    Provider,
    ProviderRole,
    SpatialExtent,
    TemporalExtent,
)
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from rio_stac import create_stac_item
from rustac import DuckdbClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from obstore.store import S3Credential

BBox = tuple[float, float, float, float]

DEFAULT_BANDS = ["red", "green", "blue", "nir_narrow", "swir_1", "swir_2"]
DTYPE = "int16"
FMASK_DTYPE = "uint8"
NODATA = -9999
INT16_SENTINEL = -32768
FMASK_NODATA = 255
HLS_BITMASK = 14
COMPOSITE_ID = "median-v1"
URL_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov"
LP_DAAC_CREDENTIALS_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
EARTHDATA_TOKEN_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
HLS_STAC_GEOPARQUET_HREF = "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/{collection}/year={year}/month={month}/{collection}-{year}-{month}.parquet"
CHUNKS = {"time": -1, "band": 1, "x": -1, "y": -1}
HLS_TILE_RE = re.compile(r"^T\d{2}[A-Z]{3}$")
HLS_ITEM_RE = re.compile(r"^HLS\.[LS]30\.(T\d{2}[A-Z]{3})\.")

COLLECTION_BAND_ALIASES = {
    "HLSL30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
        "cirrus": "B09",
        "thermal_infrared_1": "B10",
        "thermal": "B11",
    },
    "HLSS30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "red_edge_1": "B05",
        "red_edge_2": "B06",
        "red_edge_3": "B07",
        "nir_broad": "B08",
        "nir_narrow": "B8A",
        "water_vapor": "B09",
        "cirrus": "B10",
        "swir_1": "B11",
        "swir_2": "B12",
    },
}


@dataclass(frozen=True)
class NativeGrid:
    """The source CRS, transform, shape, and exact native footprint."""

    crs: CRS
    transform: Affine
    shape: tuple[int, int]
    bbox: BBox
    resolution: float


def parse_datetime_utc(dt_string: str) -> datetime:
    """Parse an ISO datetime string and normalize it to UTC."""
    dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_tile_id(tile_id: str) -> str:
    """Validate and normalize an HLS MGRS tile identifier."""
    tile_id = tile_id.upper()
    if not HLS_TILE_RE.fullmatch(tile_id):
        raise ValueError("tile_id must look like T15TYJ")
    return tile_id


def normalize_interval(
    start_datetime: datetime, end_datetime: datetime
) -> tuple[datetime, datetime, datetime]:
    """Normalize a whole-calendar-day interval to inclusive and exclusive bounds.

    The start must be UTC midnight. The end is either the next exclusive UTC
    midnight or 23:59:59 on the last included day. The returned values are the
    UTC start, exclusive query end, and precise last included second.
    """
    start = start_datetime.astimezone(timezone.utc)
    end = end_datetime.astimezone(timezone.utc)
    if start.time() != datetime.min.time():
        raise ValueError("start_datetime must be at UTC midnight")
    if end <= start:
        raise ValueError("end_datetime must be after start_datetime")
    if end.time() == datetime.min.time():
        end_exclusive = end
    elif end.time() == datetime.strptime("23:59:59", "%H:%M:%S").time():
        end_exclusive = end.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    else:
        raise ValueError(
            "end_datetime must be an exclusive UTC midnight or 23:59:59 on the last day"
        )
    if end_exclusive <= start:
        raise ValueError("end_datetime must include at least one calendar day")
    return start, end_exclusive, end_exclusive - timedelta(seconds=1)


def format_datetime(dt: datetime) -> str:
    """Format a UTC datetime as RFC 3339 without a redundant offset."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class MaapEarthdataCredentialProvider:
    """Refresh LP DAAC S3 credentials through the authenticated MAAP API."""

    def __init__(self, maap_client: Any | None = None) -> None:
        if maap_client is None:
            from maap.maap import MAAP  # type: ignore[import-untyped]

            maap_client = MAAP()
        self._maap_client = maap_client

    def __call__(self) -> "S3Credential":
        """Fetch and translate credentials into obstore's S3 schema."""
        credentials = self._maap_client.aws.earthdata_s3_credentials(
            LP_DAAC_CREDENTIALS_URL
        )
        return {
            "access_key_id": credentials["accessKeyId"],
            "secret_access_key": credentials["secretAccessKey"],
            "token": credentials["sessionToken"],
            "expires_at": parse_datetime_utc(credentials["expiration"]),
        }


def get_earthdata_token() -> str:
    """Fetch an Earthdata bearer token from EARTHDATA_USERNAME/PASSWORD."""
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "HTTP access requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables."
        )
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    request = Request(
        EARTHDATA_TOKEN_URL,
        method="POST",
        headers={"Authorization": f"Basic {credentials}"},
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310
        payload = json.loads(response.read().decode())
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Earthdata token response did not contain access_token.")
    return token


def build_duckdb_client() -> DuckdbClient:
    """Create a DuckDB client configured for the HLS parquet archive."""
    client = DuckdbClient(use_hive_partitioning=True)
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
             TYPE S3,
             PROVIDER CREDENTIAL_CHAIN
        );
        """
    )
    return client


def build_http_store() -> HTTPStore:
    """Create the authenticated HTTP store for LP DAAC HLS assets."""
    return HTTPStore(
        URL_PREFIX,
        client_options={
            "default_headers": {"Authorization": f"Bearer {get_earthdata_token()}"}
        },
    )


def build_s3_store() -> S3Store:
    """Create the authenticated direct-S3 store for LP DAAC HLS assets."""
    return S3Store(
        bucket="lp-prod-protected",
        region="us-west-2",
        credential_provider=MaapEarthdataCredentialProvider(),
    )


def path_from_lpdaac_href(href: str) -> str:
    """Translate an LP DAAC asset HREF into a bucket-relative object path."""
    return urlparse(href).path.lstrip("/").removeprefix("lp-prod-protected/")


def build_store_config(direct_bucket_access: bool) -> dict[str, Any]:
    """Build lazycogs store kwargs for the requested access mode."""
    if direct_bucket_access:
        logger.info("using direct S3 bucket access for HLS assets")
        return {"store": build_s3_store(), "path_from_href": path_from_lpdaac_href}
    logger.info("using authenticated HTTPS access for HLS assets")
    return {"store": build_http_store()}


def get_collection_band_names(collection: str, bands: list[str]) -> list[str]:
    """Resolve shared band aliases to collection-specific HLS asset names."""
    aliases = COLLECTION_BAND_ALIASES[collection]
    missing = [band for band in bands if band not in aliases]
    if missing:
        raise ValueError(
            f"Collection {collection} does not support requested bands {missing}."
        )
    return [aliases[band] for band in bands]


def datetime_range(start_datetime: datetime, end_datetime: datetime) -> str:
    """Format an inclusive query datetime range for rustac/lazycogs."""
    return f"{format_datetime(start_datetime)}/{format_datetime(end_datetime)}"


def hls_geoparquet_hrefs(
    collection: str, start_datetime: datetime, end_datetime: datetime
) -> list[str]:
    """Build exact monthly HREFs intersecting ``[start_datetime, end_datetime)``."""
    month = start_datetime.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    end = end_datetime.astimezone(timezone.utc)
    hrefs = []
    while month < end:
        hrefs.append(
            HLS_STAC_GEOPARQUET_HREF.format(
                collection=collection, year=month.year, month=month.month
            )
        )
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return hrefs


def hls_item_tile_id(item_id: str) -> str | None:
    """Extract the documented MGRS tile component from an HLS item ID."""
    match = HLS_ITEM_RE.match(item_id)
    return match.group(1) if match else None


def item_tile_id(item: dict[str, Any]) -> str | None:
    """Read a tile property when available, otherwise use the HLS ID component."""
    properties = item.get("properties", {})
    for key in ("tile_id", "mgrs:tile", "hls:tile_id"):
        value = properties.get(key)
        if value:
            return str(value).upper()
    return hls_item_tile_id(item.get("id", ""))


def discover_hls_items(
    collection: str,
    *,
    tile_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    duckdb_client: DuckdbClient,
) -> list[dict[str, Any]]:
    """Find exact-tile HLS items in ``[start_datetime, end_datetime)``."""
    tile_id = normalize_tile_id(tile_id)
    logger.info(
        f"querying stac-geoparquet archive for {collection} {tile_id} {start_datetime} {end_datetime}"
    )
    items = []
    for href in hls_geoparquet_hrefs(collection, start_datetime, end_datetime):
        items.extend(
            duckdb_client.search(
                href,
                datetime=datetime_range(
                    start_datetime, end_datetime - timedelta(microseconds=1)
                ),
                filter={
                    "op": "like",
                    "args": [{"property": "id"}, f"%{tile_id}.%"],
                },
            )
        )
    exact = {item["id"]: item for item in items if item_tile_id(item) == tile_id}
    return sorted(
        exact.values(),
        key=lambda item: (item.get("properties", {}).get("datetime", ""), item["id"]),
    )


def _asset_href_path(href: str, store_kwargs: dict[str, Any]) -> str:
    path_fn = store_kwargs.get("path_from_href")
    if path_fn is not None:
        return path_fn(href)
    return urlparse(href).path.lstrip("/")


def _inspect_cog_grid(
    item: dict[str, Any], asset_name: str, store_kwargs: dict[str, Any]
) -> NativeGrid:
    """Read one representative COG header for the native grid."""
    from async_geotiff import GeoTIFF

    href = item.get("assets", {}).get(asset_name, {}).get("href")
    if not href:
        raise ValueError(f"HLS item {item.get('id')} has no {asset_name} asset")

    async def inspect() -> NativeGrid:
        geotiff = await GeoTIFF.open(
            _asset_href_path(href, store_kwargs), store=store_kwargs["store"]
        )
        transform = Affine(*tuple(geotiff.transform)[:6])
        shape = (geotiff.height, geotiff.width)
        corners = [
            transform * point
            for point in ((0, 0), (shape[1], 0), (0, shape[0]), shape[::-1])
        ]
        if (
            transform.b
            or transform.d
            or not np.isclose(abs(transform.a), abs(transform.e))
        ):
            raise ValueError(
                f"HLS item {item.get('id')} asset {asset_name} is not an unrotated square native grid"
            )
        return NativeGrid(
            CRS.from_user_input(geotiff.crs),
            transform,
            shape,
            (
                min(x for x, _ in corners),
                min(y for _, y in corners),
                max(x for x, _ in corners),
                max(y for _, y in corners),
            ),
            abs(transform.a),
        )

    return lazycogs.run_on_loop(inspect())


def native_grid_for_items(
    items: list[dict[str, Any]],
    *,
    bands: list[str],
    store_kwargs: dict[str, Any],
) -> NativeGrid:
    """Read one representative COG header for the shared native grid."""
    if not items:
        raise ValueError("No native grid was available for COG headers.")
    item = items[0]
    spectral_asset = COLLECTION_BAND_ALIASES[item["collection"]][bands[0]]
    return _inspect_cog_grid(item, spectral_asset, store_kwargs)


def open_hls_collection(
    collection: str,
    *,
    items: list[dict[str, Any]],
    grid: NativeGrid,
    start_datetime: datetime,
    end_datetime: datetime,
    bands: list[str],
    store_kwargs: dict[str, Any],
    duckdb_client: DuckdbClient,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Open exact discovered source IDs at their native HLS grid."""
    collection_band_names = get_collection_band_names(collection, bands)
    items_by_month: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in items:
        dt = parse_datetime_utc(item["properties"]["datetime"])
        items_by_month.setdefault((dt.year, dt.month), []).append(item)

    spectral_arrays = []
    fmask_arrays = []
    month = start_datetime.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    for href in hls_geoparquet_hrefs(collection, start_datetime, end_datetime):
        month_items = items_by_month.get((month.year, month.month))
        if not month_items:
            month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
            continue
        common = {
            "href": href,
            "datetime": datetime_range(
                start_datetime, end_datetime - timedelta(microseconds=1)
            ),
            "ids": [item["id"] for item in month_items],
            "bbox": grid.bbox,
            "crs": grid.crs,
            "resolution": grid.resolution,
            "time_period": "P1D",
            "sortby": ["datetime", "id"],
            "chunks": CHUNKS,
            "duckdb_client": duckdb_client,
            **store_kwargs,
        }
        spectral_arrays.append(
            lazycogs.open(
                bands=collection_band_names,
                dtype=DTYPE,
                nodata=NODATA,
                **common,
            ).assign_coords(band=bands)
        )
        fmask_arrays.append(
            lazycogs.open(
                bands=["Fmask"],
                dtype=FMASK_DTYPE,
                nodata=FMASK_NODATA,
                **common,
            ).squeeze("band", drop=True)
        )
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)

    return (
        xr.concat(spectral_arrays, dim="time").sortby("time"),
        xr.concat(fmask_arrays, dim="time").sortby("time"),
    )


def open_hls_stacks(
    *,
    tile_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    bands: list[str],
    direct_bucket_access: bool,
) -> tuple[xr.DataArray, xr.DataArray, NativeGrid, list[str]]:
    """Discover both HLS collections and open their exact native-grid stacks."""
    duckdb_client = build_duckdb_client()
    items_by_collection = {
        collection: discover_hls_items(
            collection,
            tile_id=tile_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            duckdb_client=duckdb_client,
        )
        for collection in COLLECTION_BAND_ALIASES
    }
    all_items = [item for items in items_by_collection.values() for item in items]
    if not all_items:
        raise ValueError(
            f"No HLS items matched tile {tile_id} across HLSL30_2.0 or HLSS30_2.0."
        )
    store_kwargs = build_store_config(direct_bucket_access)
    grid = native_grid_for_items(all_items, bands=bands, store_kwargs=store_kwargs)

    spectral_arrays = []
    fmask_arrays = []
    for collection, items in items_by_collection.items():
        if not items:
            logger.info("no matching items found for %s", collection)
            continue
        spectral, fmask = open_hls_collection(
            collection,
            items=items,
            grid=grid,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            bands=bands,
            store_kwargs=store_kwargs,
            duckdb_client=duckdb_client,
        )
        spectral_arrays.append(spectral)
        fmask_arrays.append(fmask)

    spectral_stack = xr.concat(spectral_arrays, dim="time").sortby("time")
    fmask_stack = xr.concat(fmask_arrays, dim="time").sortby("time")
    spectral_stack, fmask_stack = xr.align(spectral_stack, fmask_stack, join="exact")
    return spectral_stack, fmask_stack, grid, [item["id"] for item in all_items]


def create_composite(
    spectral_stack: xr.DataArray, fmask_stack: xr.DataArray
) -> xr.DataArray:
    """Apply HLS masking and calculate the integer lower-median composite."""
    valid_mask = ((fmask_stack & HLS_BITMASK) == 0) & (spectral_stack != NODATA)
    valid_count = valid_mask.sum(dim="time")
    masked = xr.where(valid_mask, spectral_stack, INT16_SENTINEL).chunk({"time": -1})

    def lower_median(values: Any, count: Any) -> Any:
        sorted_values = np.sort(values, axis=-1)
        index = values.shape[-1] - count + (count - 1) // 2
        return np.take_along_axis(sorted_values, index[..., None], axis=-1)[..., 0]

    composite = xr.apply_ufunc(
        lower_median,
        masked,
        valid_count,
        input_core_dims=[["time"], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[spectral_stack.dtype],
    )
    composite = composite.where(valid_count > 0, NODATA)
    return composite.transpose(
        *[dimension for dimension in spectral_stack.dims if dimension != "time"]
    )


def item_id(tile_id: str, start_datetime: datetime, end_datetime: datetime) -> str:
    """Return the deterministic identity for one tile, interval, and method."""
    return f"hls-{normalize_tile_id(tile_id)}-{start_datetime:%Y%m%d}-{end_datetime:%Y%m%d}-{COMPOSITE_ID}"


def export_outputs(
    composite: xr.DataArray,
    *,
    tile_id: str,
    grid: NativeGrid,
    bands: list[str],
    start_datetime: datetime,
    end_datetime: datetime,
    output_dir: Path,
    source_item_ids: list[str],
) -> None:
    """Write native-grid COGs and a deterministic self-contained STAC item."""
    assets: dict[str, Asset] = {}
    writes = []
    for band in bands:
        href = f"{band}.tif"
        logger.info("exporting %s", href)
        da = composite.sel(band=band, drop=True)
        da_to_export = (
            da.rio.write_crs(grid.crs, inplace=False)
            .rio.write_transform(grid.transform, inplace=False)
            .rio.write_nodata(NODATA, encoded=True, inplace=False)
        )
        writes.append(
            da_to_export.rio.to_raster(
                output_dir / href,
                driver="COG",
                dtype=DTYPE,
                compress="DEFLATE",
                compute=False,
            )
        )
        assets[band] = Asset(
            href=href,
            description=f"lower-median {band} band value from cloud-free pixels in the temporal mosaic",
            media_type=MediaType.COG,
            roles=["data"],
        )
    dask.compute(*writes, scheduler="threads", num_workers=1)

    catalog = Catalog(
        id="DPS", description="DPS", catalog_type=CatalogType.SELF_CONTAINED
    )
    collection = Collection(
        id="hls-cloud-free-temporal-mosaic",
        title="HLS Cloud-Free Temporal Mosaic",
        description=(
            "Cloud-free temporal mosaics of HLS surface reflectance. "
            "The algorithm masks cloud and cloud-shadow pixels using HLS Fmask "
            "quality flags, then computes a per-pixel integer lower-median composite."
        ),
        extent=Extent(
            spatial=SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=TemporalExtent(cast(list[list[datetime | None]], [[None, None]])),
        ),
        license="other",
        keywords=["HLS", "cloud-free", "temporal mosaic", "Fmask"],
        providers=[
            Provider(
                name="MAAP Project",
                roles=[ProviderRole.PROCESSOR],
                url="https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic",
            )
        ],
    )
    collection.item_assets = {
        band: ItemAssetDefinition.create(
            title=assets[band].title,
            description=assets[band].description,
            media_type=assets[band].media_type,
            roles=assets[band].roles,
        )
        for band in bands
    }
    collection.add_link(
        Link(
            rel="documentation",
            target="https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic",
            title="Algorithm source repository",
        )
    )
    collection.add_link(
        Link(
            rel="source",
            target="https://doi.org/10.5067/HLS/HLSL30.002",
            title="HLSL30 2.0",
        )
    )
    collection.add_link(
        Link(
            rel="source",
            target="https://doi.org/10.5067/HLS/HLSS30.002",
            title="HLSS30 2.0",
        )
    )
    catalog.add_child(collection)

    source_file = str(output_dir / assets[bands[0]].href)
    item = create_stac_item(
        source=source_file,
        input_datetime=None,
        id=item_id(tile_id, start_datetime, end_datetime),
        with_proj=True,
        properties={
            "datetime": None,
            "start_datetime": format_datetime(start_datetime),
            "end_datetime": format_datetime(end_datetime),
            "hls:tile_id": normalize_tile_id(tile_id),
            "hls:bands": bands,
            "hls:composite": COMPOSITE_ID,
            "hls:reducer": "lower-median",
            "hls:mask": "Fmask bitmask 14 equals zero and spectral nodata is excluded",
            "hls:time_grouping": "P1D",
            "hls:daily_sampling": "first-valid per collection after datetime,id ordering",
            "hls:source_item_ids": source_item_ids,
        },
    )
    item.assets = {}
    for band, asset in assets.items():
        item.add_asset(band, asset)
        RasterExtension.ext(asset, add_if_missing=True).bands = [
            RasterBand.create(data_type=DataType(DTYPE))
        ]
    item.set_self_href(f"{output_dir}/item.json")
    collection.add_item(item)
    item.make_asset_hrefs_relative()
    catalog.normalize_and_save(
        root_href=str(output_dir), catalog_type=CatalogType.SELF_CONTAINED
    )


def run(
    start_datetime: datetime,
    end_datetime: datetime,
    tile_id: str,
    output_dir: Path,
    bands: list[str] = DEFAULT_BANDS,
    direct_bucket_access: bool = False,
) -> None:
    """Generate one native HLS tile composite and write its outputs."""
    start, query_end, included_end = normalize_interval(start_datetime, end_datetime)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_stack, fmask_stack, grid, source_item_ids = open_hls_stacks(
        tile_id=tile_id,
        start_datetime=start,
        end_datetime=query_end,
        bands=bands,
        direct_bucket_access=direct_bucket_access,
    )
    composite = create_composite(spectral_stack, fmask_stack)
    export_outputs(
        composite,
        tile_id=tile_id,
        grid=grid,
        bands=bands,
        start_datetime=start,
        end_datetime=included_end,
        output_dir=output_dir,
        source_item_ids=source_item_ids,
    )


def parse_tile_id(value: str) -> str:
    """Parse one HLS MGRS tile ID for argparse."""
    try:
        return normalize_tile_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line inputs without starting network or filesystem work."""
    parser = argparse.ArgumentParser(
        description="Query the HLS STAC geoparquet archive for one native tile and write the result"
    )
    parser.add_argument(
        "--start_datetime", help="UTC midnight start in ISO format", required=True
    )
    parser.add_argument(
        "--end_datetime",
        help="exclusive UTC midnight, or 23:59:59 UTC on the last included day",
        required=True,
    )
    parser.add_argument(
        "--tile_id",
        help="HLS MGRS tile ID, for example T15TYJ",
        required=True,
        type=parse_tile_id,
    )
    parser.add_argument(
        "--output_dir", help="Directory in which to save output", required=True
    )
    parser.add_argument(
        "--direct_bucket_access",
        help="Use direct LP DAAC S3 bucket access instead of HTTPS URLs.",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the cloud-free mosaic command for parsed command-line inputs."""
    args = parse_args(argv)
    start_datetime = parse_datetime_utc(args.start_datetime)
    end_datetime = parse_datetime_utc(args.end_datetime)
    logger.info(
        "running with start_datetime=%s end_datetime=%s tile_id=%s output_dir=%s direct_bucket_access=%s",
        start_datetime,
        end_datetime,
        args.tile_id,
        args.output_dir,
        args.direct_bucket_access,
    )
    run(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        tile_id=args.tile_id,
        output_dir=Path(args.output_dir),
        direct_bucket_access=args.direct_bucket_access,
    )
    logger.info("Successfully completed processing")


if __name__ == "__main__":
    main()
