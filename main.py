"""Create a cloud-free composite image from a temporal mosaic of HLS granules"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import odc.stac
import rioxarray  # noqa
from maap.maap import MAAP
from odc.geo.geobox import GeoBox
from odc.stac import ParsedItem
from pyproj import CRS
from pystac import Asset, Catalog, CatalogType, Item, MediaType
from rasterio.warp import transform_bounds
from rio_stac import create_stac_item
from rustac import DuckdbClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

MEMORY_GB = 8
GDAL_CONFIG = {
    "CPL_TMPDIR": "/tmp",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
    "GDAL_CACHEMAX": "75%",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "PYTHONWARNINGS": "ignore",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "536870912",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    # "CPL_DEBUG": "ON" if debug else "OFF",
    # "CPL_CURL_VERBOSE": "YES" if debug else "NO",
}

URL_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/"
DTYPE = "int16"
FMASK_DTYPE = "uint8"
NODATA = -9999
FMASK_NODATA = 255
HLS_ODC_STAC_CONFIG = {
    "HLSL30_2.0": {
        "assets": {
            "*": {
                "nodata": NODATA,
                "data_type": DTYPE,
            },
            "Fmask": {
                "nodata": FMASK_NODATA,
                "data_type": FMASK_DTYPE,
            },
        },
        "aliases": {
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
    },
    "HLSS30_2.0": {
        "assets": {
            "*": {
                "nodata": NODATA,
                "data_type": DTYPE,
            },
            "Fmask": {
                "nodata": FMASK_NODATA,
                "data_type": FMASK_DTYPE,
            },
        },
        "aliases": {
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
    },
}

# these are the ones that we are going to use
DEFAULT_BANDS = ["red", "green", "blue"]
DEFAULT_RESOLUTION = 30

"""
hls_bitmask:
hls_mask_bitfields = [1, 2, 3]  # cloud shadow, adjacent to cloud shadow, cloud
hls_bitmask = 0
for field in hls_mask_bitfields:
    hls_bitmask |= 1 << field
"""
HLS_BITMASK = 14


def parse_datetime_utc(dt_string: str) -> datetime:
    """
    Parse a datetime string and ensure it has UTC timezone.
    If no timezone is specified, assume UTC.

    Args:
        dt_string: ISO format datetime string (e.g., '2024-01-01T00:00:00' or '2024-01-01T00:00:00Z')

    Returns:
        datetime object with UTC timezone
    """
    dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))

    # If the datetime is naive (no timezone), assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def validate_crs_units_in_meters(crs: CRS) -> None:
    """
    Validate that the CRS uses meters as its linear unit.

    Args:
        crs: The CRS to validate

    Raises:
        ValueError: If the CRS does not use meters as its linear unit
    """
    # Get the axis info to check units
    axis_info = crs.axis_info

    if not axis_info:
        raise ValueError(
            f"Cannot determine units for CRS '{crs}'. "
            "Please provide a CRS with meter units."
        )

    # Check if any axis uses non-meter units
    for axis in axis_info:
        unit_name = axis.unit_name.lower()
        # Common meter unit names: "metre", "meter", "m"
        if unit_name not in ["metre", "meter", "m"]:
            raise ValueError(
                f"CRS '{crs}' uses '{axis.unit_name}' units, but only CRS with meter units are supported. "
                f"Please provide a CRS that uses meters (e.g., UTM zones, Web Mercator)."
            )


def group_by_sensor_and_date(
    item: Item,
    parsed: ParsedItem,
    idx: int,
) -> str:
    id_split = item.id.split(".")
    sensor = id_split[1]
    day = id_split[3][:7]

    return f"{sensor}_{day}"


def get_stac_items(
    bbox: BBox, start_datetime: datetime, end_datetime: datetime, crs: CRS
) -> list[Item]:
    logger.info("querying HLS archive")
    client = DuckdbClient(use_hive_partitioning=True)
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
             TYPE S3,
             PROVIDER CREDENTIAL_CHAIN
        );
        """
    )

    items = client.search(
        href="s3://maap-ops-workspace/shared/henrydevseed/hls-stac-geoparquet-v1/year=*/month=*/*.parquet",
        datetime="/".join(dt.isoformat() for dt in [start_datetime, end_datetime]),
        bbox=transform_bounds(
            src_crs=crs,
            dst_crs="epsg:4326",
            left=bbox[0],
            bottom=bbox[1],
            right=bbox[2],
            top=bbox[3],
        ),
    )

    logger.info(f"found {len(items)} items")

    # the HLS STAC geoparquet store contains items from two collections but
    # the collection id is not set :/
    for item in items:
        item["collection"] = (
            "HLSL30_2.0" if item["id"].startswith("HLS.L30") else "HLSS30_2.0"
        )
        del item["properties"]["proj:epsg"]

        item["stac_extensions"] = [
            ext for ext in item["stac_extensions"] if "proj" not in ext
        ]

        item["type"] = "Feature"

    return [Item.from_dict(item) for item in items]


async def run(
    start_datetime: datetime,
    end_datetime: datetime,
    bbox: BBox,
    crs: CRS,
    output_dir: Path,
    bands: list[str] = DEFAULT_BANDS,
    resolution: int | float = DEFAULT_RESOLUTION,
    direct_bucket_access: bool = False,
):
    if not bands:
        raise ValueError("you must provide a list of bands")

    items = get_stac_items(
        bbox=bbox,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        crs=crs,
    )

    if direct_bucket_access:
        maap = MAAP(maap_host="api.maap-project.org")
        creds = maap.aws.earthdata_s3_credentials(
            "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        )
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={
                "aws_access_key_id": creds["accessKeyId"],
                "aws_secret_access_key": creds["secretAccessKey"],
                "aws_session_token": creds["sessionToken"],
                "region_name": "us-west-2",
            },
        )
        for item in items:
            for asset in item.assets.values():
                if asset.href.startswith(URL_PREFIX):
                    asset.href = asset.href.replace(URL_PREFIX, "s3://")

    logger.info("loading into xarray via odc.stac")
    stack = odc.stac.load(
        items,
        stac_cfg=HLS_ODC_STAC_CONFIG,
        bands=list(set(bands + ["Fmask"])),
        chunks={"x": 512, "y": 512},
        groupby=group_by_sensor_and_date,
        geobox=GeoBox.from_bbox(bbox=bbox, crs=crs, resolution=resolution, tight=True),
    ).sortby("time")

    mask = stack["Fmask"] & HLS_BITMASK

    cloud_free = stack[bands].where(mask == 0).where(stack != NODATA)

    logger.info("computing median values")
    composite = cloud_free.median(dim="time", skipna=True).fillna(NODATA).compute()

    assets = {}
    for band in bands:
        href = f"{band}.tif"
        logger.info(f"exporting {href}")
        da = composite[band]
        da.rio.set_nodata(NODATA, inplace=True)
        da_to_export = da.rio.write_nodata(NODATA, encoded=True, inplace=False)

        output_file_path = output_dir / href

        da_to_export.rio.to_raster(
            output_file_path,
            driver="COG",
            dtype=DTYPE,
            compress="DEFLATE",
        )

        assets[band] = Asset(
            href=href,
            description=f"median {band} band value from cloud-free pixels in the temporal mosaic",
            media_type=MediaType.COG,
            roles=["data"],
        )

    catalog = Catalog(
        id="DPS",
        description="DPS",
        catalog_type=CatalogType.SELF_CONTAINED,
    )

    source_file = f"{output_dir}/{assets[bands[0]].href}"
    item = create_stac_item(
        source=source_file,
        id=f"{'_'.join(str(int(x)) for x in bbox)}-{start_datetime.strftime('%Y%m%d')}-{end_datetime.strftime('%Y%m%d')}",
        with_proj=True,
        properties={
            "datetime": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_datetime": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )

    # replace auto-generated assets with our own
    item.assets = assets

    # set datetime properties
    item.set_self_href(f"{output_dir}/item.json")

    catalog.add_item(item)
    item.make_asset_hrefs_relative()

    catalog.normalize_and_save(
        root_href=str(output_dir),
        catalog_type=CatalogType.SELF_CONTAINED,
    )


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Queries the HLS STAC geoparquet archive and writes the result to a file"
    )
    parse.add_argument(
        "--start_datetime",
        help="start datetime in ISO format (e.g., 2024-01-01T00:00:00Z)",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--end_datetime",
        help="end datetime in ISO format (e.g., 2024-12-31T23:59:59Z)",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--bbox",
        help="bounding box (xmin, ymin, xmax, ymax)",
        required=True,
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
    )
    parse.add_argument(
        "--crs",
        help="CRS definition of the bounding box coordinates",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--output_dir", help="Directory in which to save output", required=True
    )
    parse.add_argument(
        "--direct_bucket_access",
        help="Use direct S3 bucket access instead of HTTP URLs",
        action="store_true",
        default=False,
    )
    args = parse.parse_args()

    output_dir = Path(args.output_dir)
    bbox = tuple(args.bbox)
    crs = CRS.from_string(args.crs)
    validate_crs_units_in_meters(crs)
    start_datetime = parse_datetime_utc(args.start_datetime)
    end_datetime = parse_datetime_utc(args.end_datetime)

    logging.info(
        f"setting GDAL config environment variables:\n{json.dumps(GDAL_CONFIG, indent=2)}"
    )
    os.environ.update(GDAL_CONFIG)

    logging.info(
        f"running with start_datetime: {start_datetime}, end_datetime: {end_datetime}, bbox: {bbox}, crs: {crs}, output_dir: {output_dir}"
    )

    # Retry loop for handling intermittent failures
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            asyncio.run(
                run(
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    bbox=bbox,
                    crs=crs,
                    output_dir=output_dir,
                    direct_bucket_access=args.direct_bucket_access,
                )
            )
            logging.info("Successfully completed processing")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # exponential backoff
                logging.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed with error: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} attempts failed. Last error: {e}")
                raise
