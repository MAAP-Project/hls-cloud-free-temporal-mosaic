# HLS Cloud-Free Temporal Mosaic

Create cloud-free composite images from temporal mosaics of HLS granules using STAC Geoparquet

## Motivation

The CMR STAC API has imposed rate limits on the HLS collections. This algorithm queries the HLS STAC records directly from parquet files in S3 without any API rate limits, then generates cloud-free composites by mosaicking multiple observations over time.

## About

This DPS algorithm uses [`rustac`](https://github.com/stac-utils/rustac-py) to query an archive of HLS STAC records stored as STAC Geoparquet, then creates cloud-free composites using temporal median compositing. The workflow:

1. Queries HLS STAC items for a given bounding box and time range
2. Loads the spectral bands and cloud mask (Fmask) using `odc-stac`
3. Masks out cloudy pixels using the HLS Fmask layer
4. Computes the median value for each pixel across the time series
5. Exports the result as Cloud Optimized GeoTIFFs with STAC metadata

By using `rustac` + parquet files there is no API between the requester and the actual data!

> [!WARNING]
> This archive of HLS STAC records is experimental and only contains items through May 2025.

## Usage

### Direct Python Invocation

```bash
uv run main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615" \
  --output_dir "./output" \
  --direct_bucket_access  # optional: use S3 URIs instead of HTTPS (must be running in us-west-2)
```

### Shell Script Wrappers for DPS

**Legacy DPS (positional arguments)**:
```bash
./run.sh "2025-05-01T00:00:00Z" "2025-05-31T23:59:59Z" "500000 5000000 600000 5100000" "EPSG:32615"
```

**OGC Application Packages (named arguments)**:
```bash
./run-named.sh \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox "500000 5000000 600000 5100000" \
  --crs "EPSG:32615"
```

Both scripts automatically create the `output` directory and handle the `input` directory structure expected by DPS. They enable direct bucket access by default for faster data retrieval.

### Parameters

- `--start_datetime`: Start datetime in ISO format (e.g., 2025-05-01T00:00:00Z)
- `--end_datetime`: End datetime in ISO format (e.g., 2025-05-31T23:59:59Z)
- `--bbox`: Bounding box coordinates (xmin ymin xmax ymax)
- `--crs`: CRS definition for the bounding box coordinates. **Must use meter units** (e.g., UTM zones like EPSG:32615, Web Mercator EPSG:3857). Geographic CRS with degree units (like EPSG:4326) are not supported.
- `--output_dir`: Directory where output files will be saved (handled automatically by shell scripts)
- `--direct_bucket_access`: Optional flag to use S3 URIs instead of HTTPS URLs for faster data access (enabled by default in shell scripts)

## Details

The partitioned parquet dataset is available in my shared folder in the `maap-ops-workspace` bucket. The MAAP ADE and DPS both will have permissions to read from the archive.

The algorithm groups HLS observations by sensor (L30/S30) and date, masks cloudy pixels using Fmask bit flags (cloud shadow, adjacent to cloud shadow, and cloud), and computes the temporal median to fill in cloud-free values. The output includes a STAC catalog with COG assets for each requested band (default: red, green, blue).
