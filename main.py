"""Create a cloud-free composite image from a temporal mosaic of HLS granules"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import odc.stac
import rioxarray  # noqa
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
NODATA = -9999
HLS_ODC_STAC_CONFIG = {
    "HLSL30_2.0": {
        "assets": {
            "*": {
                "nodata": NODATA,
                "data_type": "int16",
            },
            "Fmask": {
                "nodata": 0,
                "data_type": "uint8",
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
                "dtype": "int16",
            },
            "Fmask": {
                "nodata": 0,
                "dtype": "uint8",
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


def group_by_sensor_and_date(
    item: Item,
    parsed: ParsedItem,
    idx: int,
) -> str:
    id_split = item.id.split(".")
    sensor = id_split[1]
    day = id_split[3][:7]

    return f"{sensor}_{day}"


async def run(
    temporal: str,
    bbox: BBox,
    crs: CRS,
    output_dir: Path,
    bands: list[str] = DEFAULT_BANDS,
    resolution: int | float = DEFAULT_RESOLUTION,
):
    if not bands:
        raise ValueError("you must provide a list of bands")

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
        datetime=temporal,
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

    items = [Item.from_dict(item) for item in items]

    logger.info("loading into xarray via odc.stac")
    stack = odc.stac.load(
        items,
        stac_cfg=HLS_ODC_STAC_CONFIG,
        bands=list(set(bands + ["Fmask"])),
        chunks={"x": 512, "y": 512},
        groupby=group_by_sensor_and_date,
        geobox=GeoBox.from_bbox(bbox=bbox, crs=crs, resolution=resolution, tight=True),
    ).sortby("time")

    fmask = stack["Fmask"].astype("uint16")
    mask = fmask & HLS_BITMASK

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
            dtype="int16",
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
        id=f"{'_'.join(str(x) for x in bbox)}-{temporal}",
        with_proj=True,
    )

    # replace auto-generated assets with our own
    item.assets = assets
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
    parse.add_argument("--temporal", help="temporal range for the query", required=True)
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
    args = parse.parse_args()

    output_dir = Path(args.output_dir)
    bbox = tuple(args.bbox)
    crs = CRS.from_string(args.crs)
    logging.info(
        f"setting GDAL config environment variables:\n{json.dumps(GDAL_CONFIG, indent=2)}"
    )
    os.environ.update(GDAL_CONFIG)
    logging.info(
        f"running with temporal: {args.temporal}, bbox: {bbox}, crs: {crs}, output_dir: {output_dir}"
    )
    asyncio.run(run(temporal=args.temporal, bbox=bbox, crs=crs, output_dir=output_dir))
