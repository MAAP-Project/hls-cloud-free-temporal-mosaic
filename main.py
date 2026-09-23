"""Create a cloud-free composite image from a temporal mosaic of HLS granules."""

import argparse
import base64
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import dask
import lazycogs
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
    SpatialExtent,
    TemporalExtent,
)
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
DEFAULT_RESOLUTION = 30
DTYPE = "int16"
FMASK_DTYPE = "uint8"
NODATA = -9999
FMASK_NODATA = 255
HLS_BITMASK = 14
URL_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov"
LP_DAAC_CREDENTIALS_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
EARTHDATA_TOKEN_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
HLS_STAC_GEOPARQUET_HREF = "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/{collection}/**/*.parquet"
CHUNKS = {"time": -1, "band": 1, "x": 2048, "y": 2048}

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


def parse_datetime_utc(dt_string: str) -> datetime:
    """Parse an ISO datetime string and ensure it is timezone-aware UTC."""
    dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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


def validate_crs_units_in_meters(crs: CRS) -> None:
    """Validate that the CRS uses meters as its linear unit."""
    axis_info = crs.axis_info
    if not axis_info:
        raise ValueError(
            f"Cannot determine units for CRS '{crs}'. Please provide a CRS with meter units."
        )

    for axis in axis_info:
        unit_name = axis.unit_name.lower()
        if unit_name not in ["metre", "meter", "m"]:
            raise ValueError(
                f"CRS '{crs}' uses '{axis.unit_name}' units, but only CRS with meter units are supported. "
                "Please provide a CRS that uses meters (e.g., UTM zones, Web Mercator)."
            )


def get_earthdata_token() -> str:
    """Fetch an Earthdata bearer token from EARTHDATA_USERNAME/PASSWORD."""
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "HTTP access requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables."
        )

    credentials = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
        "ascii"
    )
    request = Request(
        EARTHDATA_TOKEN_URL,
        method="POST",
        headers={"Authorization": f"Basic {credentials}"},
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Earthdata token response did not contain access_token.")
    return token


def build_duckdb_client() -> DuckdbClient:
    """Create a DuckDB client configured for the hive-partitioned HLS parquet archive."""
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
    token = get_earthdata_token()
    return HTTPStore(
        URL_PREFIX,
        client_options={
            "default_headers": {
                "Authorization": f"Bearer {token}",
            },
        },
    )


def build_s3_store() -> S3Store:
    """Create the authenticated direct-S3 store for LP DAAC HLS assets."""
    credential_provider = MaapEarthdataCredentialProvider()
    return S3Store(
        bucket="lp-prod-protected",
        region="us-west-2",
        credential_provider=credential_provider,
    )


def path_from_lpdaac_href(href: str) -> str:
    """Translate an LP DAAC asset HREF into a bucket-relative object path."""
    return urlparse(href).path.lstrip("/").removeprefix("lp-prod-protected/")


def build_store_config(direct_bucket_access: bool) -> dict[str, Any]:
    """Build lazycogs store kwargs for the requested access mode."""
    if direct_bucket_access:
        logger.info("using direct S3 bucket access for HLS assets")
        return {
            "store": build_s3_store(),
            "path_from_href": path_from_lpdaac_href,
        }

    logger.info("using authenticated HTTPS access for HLS assets")
    return {"store": build_http_store()}


def get_collection_band_names(collection: str, bands: list[str]) -> list[str]:
    """Resolve requested shared band aliases to collection-specific HLS asset names."""
    aliases = COLLECTION_BAND_ALIASES[collection]
    missing = [band for band in bands if band not in aliases]
    if missing:
        raise ValueError(
            f"Collection {collection} does not support requested bands {missing}."
        )
    return [aliases[band] for band in bands]


def datetime_range(start_datetime: datetime, end_datetime: datetime) -> str:
    """Format the query datetime range for rustac/lazycogs."""
    return f"{start_datetime.isoformat()}/{end_datetime.isoformat()}"


def open_hls_collection(
    collection: str,
    *,
    duckdb_client: DuckdbClient,
    bbox: BBox,
    crs: CRS,
    start_datetime: datetime,
    end_datetime: datetime,
    bands: list[str],
    resolution: int | float,
    store_kwargs: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Open one HLS collection's spectral bands and Fmask as lazy arrays."""
    href = HLS_STAC_GEOPARQUET_HREF.format(collection=collection)
    collection_band_names = get_collection_band_names(collection, bands)
    search_args = {
        "href": href,
        "datetime": datetime_range(start_datetime, end_datetime),
        "duckdb_client": duckdb_client,
    }

    try:
        spectral = lazycogs.open(
            crs=crs,
            bbox=bbox,
            resolution=resolution,
            time_period="P1D",
            bands=collection_band_names,
            chunks=CHUNKS,
            dtype=DTYPE,
            nodata=NODATA,
            **store_kwargs,
            **search_args,
        )
        fmask = lazycogs.open(
            crs=crs,
            bbox=bbox,
            resolution=resolution,
            time_period="P1D",
            bands=["Fmask"],
            chunks=CHUNKS,
            dtype=FMASK_DTYPE,
            nodata=FMASK_NODATA,
            **store_kwargs,
            **search_args,
        ).squeeze("band", drop=True)
    except ValueError as exc:
        if "No STAC items matched the query" in str(exc):
            logger.info("no matching items found for %s", collection)
            return None
        raise

    spectral = spectral.assign_coords(band=bands)
    return spectral, fmask


def open_hls_stacks(
    *,
    bbox: BBox,
    crs: CRS,
    start_datetime: datetime,
    end_datetime: datetime,
    bands: list[str],
    resolution: int | float,
    direct_bucket_access: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Open combined HLS spectral and Fmask stacks across both collections."""
    duckdb_client = build_duckdb_client()
    store_kwargs = build_store_config(direct_bucket_access)

    spectral_arrays: list[xr.DataArray] = []
    fmask_arrays: list[xr.DataArray] = []

    for collection in COLLECTION_BAND_ALIASES:
        opened = open_hls_collection(
            collection,
            duckdb_client=duckdb_client,
            bbox=bbox,
            crs=crs,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            bands=bands,
            resolution=resolution,
            store_kwargs=store_kwargs,
        )
        if opened is None:
            continue
        spectral, fmask = opened
        spectral_arrays.append(spectral)
        fmask_arrays.append(fmask)

    if not spectral_arrays or not fmask_arrays:
        raise ValueError(
            "No HLS items matched the query across HLSL30_2.0 or HLSS30_2.0."
        )

    spectral_stack = xr.concat(spectral_arrays, dim="time").sortby("time")
    fmask_stack = xr.concat(fmask_arrays, dim="time").sortby("time")
    logger.info(
        "concatenated arrays: spectral dims=%s shape=%s sizes=%s; fmask dims=%s shape=%s sizes=%s",
        spectral_stack.dims,
        spectral_stack.shape,
        dict(spectral_stack.sizes),
        fmask_stack.dims,
        fmask_stack.shape,
        dict(fmask_stack.sizes),
    )
    # spectral_stack, fmask_stack = xr.align(spectral_stack, fmask_stack, join="exact")
    #
    return spectral_stack, fmask_stack


def create_composite(
    spectral_stack: xr.DataArray, fmask_stack: xr.DataArray
) -> xr.DataArray:
    """Apply the HLS Fmask QA mask and lazily calculate the temporal median composite."""
    valid_mask = (fmask_stack & HLS_BITMASK) == 0
    cloud_free = spectral_stack.where(valid_mask).where(spectral_stack != NODATA)
    return cloud_free.median(dim="time", skipna=True).fillna(NODATA)


def export_outputs(
    composite: xr.DataArray,
    *,
    bands: list[str],
    bbox: BBox,
    start_datetime: datetime,
    end_datetime: datetime,
    output_dir: Path,
    crs: CRS,
) -> None:
    """Write per-band COGs and the output STAC item."""
    assets: dict[str, Asset] = {}
    transform = Affine(*composite.attrs["spatial:transform"])

    writes = []
    for band in bands:
        href = f"{band}.tif"
        logger.info("exporting %s", href)
        da = composite.sel(band=band, drop=True)
        da_to_export = (
            da.rio.write_crs(crs, inplace=False)
            .rio.write_transform(transform, inplace=False)
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
            description=f"median {band} band value from cloud-free pixels in the temporal mosaic",
            media_type=MediaType.COG,
            roles=["data"],
        )

    dask.compute(*writes, scheduler="threads", num_workers=1)

    catalog = Catalog(
        id="DPS",
        description="DPS",
        catalog_type=CatalogType.SELF_CONTAINED,
    )
    collection = Collection(
        id="hls-cloud-free-temporal-mosaic",
        title="HLS Cloud-Free Temporal Mosaic",
        description=(
            "Cloud-free temporal mosaics of HLS surface reflectance. "
            "The algorithm masks cloud and cloud-shadow pixels using HLS Fmask "
            "quality flags, then computes a per-pixel median reflectance composite."
        ),
        extent=Extent(
            spatial=SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=TemporalExtent([[None, None]]),
        ),
        license="other",
        keywords=["HLS", "cloud-free", "temporal mosaic", "Fmask"],
        providers=[
            Provider(
                name="MAAP Project",
                roles=["processor"],
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
        input_datetime=end_datetime,
        id="-".join(
            [
                "_".join(str(int(x)) for x in bbox),
                start_datetime.strftime("%Y%m%d"),
                end_datetime.strftime("%Y%m%d"),
            ]
        ),
        with_proj=True,
        properties={
            "datetime": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_datetime": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )

    item.assets = assets
    item.set_self_href(f"{output_dir}/item.json")
    collection.add_item(item)
    item.make_asset_hrefs_relative()
    catalog.normalize_and_save(
        root_href=str(output_dir),
        catalog_type=CatalogType.SELF_CONTAINED,
    )


def run(
    start_datetime: datetime,
    end_datetime: datetime,
    bbox: BBox,
    crs: CRS,
    output_dir: Path,
    bands: list[str] = DEFAULT_BANDS,
    resolution: int | float = DEFAULT_RESOLUTION,
    direct_bucket_access: bool = False,
) -> None:
    """Generate the cloud-free temporal mosaic and write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_stack, fmask_stack = open_hls_stacks(
        bbox=bbox,
        crs=crs,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        bands=bands,
        resolution=resolution,
        direct_bucket_access=direct_bucket_access,
    )
    composite = create_composite(spectral_stack, fmask_stack)
    export_outputs(
        composite,
        bands=bands,
        bbox=bbox,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_dir=output_dir,
        crs=crs,
    )


def parse_bbox(value: str) -> list[float]:
    """Parse four finite, space-separated bounding-box coordinates."""
    try:
        bbox = [float(coordinate) for coordinate in value.split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bbox must contain exactly four finite numeric coordinates"
        ) from exc

    if len(bbox) != 4 or not all(math.isfinite(coordinate) for coordinate in bbox):
        raise argparse.ArgumentTypeError(
            "bbox must contain exactly four finite numeric coordinates"
        )
    return bbox


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line inputs without starting network or filesystem work."""
    parser = argparse.ArgumentParser(
        description="Queries the HLS STAC geoparquet archive and writes the result to a file"
    )
    parser.add_argument(
        "--start_datetime",
        help="start datetime in ISO format (e.g., 2024-01-01T00:00:00Z)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--end_datetime",
        help="end datetime in ISO format (e.g., 2024-12-31T23:59:59Z)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--bbox",
        help="space-separated bounding box (xmin, ymin, xmax, ymax)",
        required=True,
        type=parse_bbox,
        metavar="BBOX",
    )
    parser.add_argument(
        "--crs",
        help="CRS definition of the bounding box coordinates",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir", help="Directory in which to save output", required=True
    )
    parser.add_argument(
        "--direct_bucket_access",
        help=(
            "Use direct LP DAAC S3 bucket access instead of HTTPS URLs. "
            "The OGC Application Package enables this by default."
        ),
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the cloud-free mosaic command for parsed command-line inputs."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    bbox = tuple(args.bbox)
    crs = CRS.from_string(args.crs)
    validate_crs_units_in_meters(crs)
    start_datetime = parse_datetime_utc(args.start_datetime)
    end_datetime = parse_datetime_utc(args.end_datetime)

    logger.info(
        "running with start_datetime=%s end_datetime=%s bbox=%s crs=%s output_dir=%s direct_bucket_access=%s",
        start_datetime,
        end_datetime,
        bbox,
        crs,
        output_dir,
        args.direct_bucket_access,
    )

    run(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        bbox=bbox,
        crs=crs,
        output_dir=output_dir,
        direct_bucket_access=args.direct_bucket_access,
    )
    logger.info("Successfully completed processing")


if __name__ == "__main__":
    main()
