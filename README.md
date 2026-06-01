# HLS Cloud-Free Temporal Mosaic

Create cloud-free composite images from temporal mosaics of HLS granules using the HLS STAC geoparquet archive and `lazycogs`.

## Motivation

The CMR STAC API has imposed rate limits on the HLS collections. This algorithm queries HLS STAC records directly from parquet files in S3, then generates cloud-free composites by mosaicking multiple observations over time.

## About

This DPS algorithm uses [`rustac`](https://github.com/stac-utils/rustac-py) to query an archive of HLS STAC records stored as STAC geoparquet, then uses [`lazycogs`](https://github.com/developmentseed/lazycogs) to read the HLS COG assets and build cloud-free temporal composites. The workflow:

1. Queries HLS STAC items for a given bounding box and time range.
2. Opens the requested spectral bands and `Fmask` separately with `lazycogs`.
3. Masks out cloudy pixels using HLS `Fmask` bits 1, 2, and 3.
4. Computes the median value for each pixel across the time series.
5. Exports the result as Cloud Optimized GeoTIFFs with STAC metadata.

The export path writes the affine transform from `lazycogs` metadata explicitly before calling `rioxarray`/GDAL so the output COGs stay on the intended rectilinear grid.

> [!WARNING]
> This archive of HLS STAC records is experimental and is a few days behind the latest records in CMR.
> See the [hls-stac-geoparquet-archive repo](https://github.com/MAAP-Project/hls-stac-geoparquet-archive) for details.

## Authentication and access modes

The parquet query path still reads from the MAAP-hosted geoparquet archive through DuckDB's AWS credential chain.

There are two HLS asset access modes, but they are not equally important:

- Production DPS / OGC Application Package runs use direct S3 mode (`direct_bucket_access=True`). `run.sh` enables this by default and reads LP DAAC assets from `s3://lp-prod-protected/...` through an authenticated `S3Store`.
- Local smoke tests use HTTPS mode (`direct_bucket_access=False`). This reads LP DAAC asset URLs over HTTPS through an authenticated `HTTPStore` and avoids the `us-west-2` direct-S3 constraint.

Do not add `direct_bucket_access` as a MAAP DPS application input. The deployed package should always use the `run.sh` default. The only supported reason to set `DIRECT_BUCKET_ACCESS=false` is local testing, such as `smoketest.sh`.

### Production DPS direct-S3 requirements

Direct S3 mode uses `obstore.auth.earthdata.NasaEarthdataCredentialProvider` to fetch short-lived LP DAAC S3 credentials. This mode is intended for DPS compute running in `us-west-2`.

### Local HTTPS smoke-test requirements

Set these environment variables before running the local HTTPS smoke test:

```bash
export EARTHDATA_USERNAME="your-earthdata-username"
export EARTHDATA_PASSWORD="your-earthdata-password"
```

## Usage

### DPS shell wrapper

`run.sh` is the MAAP DPS entry point. It keeps the four positional inputs from `algorithm-config.yml` for backwards compatibility with current MAAP DPS and enables direct S3 bucket access by default:

```bash
./run.sh "2025-05-01T00:00:00Z" "2025-05-31T23:59:59Z" "500000 5000000 600000 5100000" "EPSG:32615"
```

`run.sh` also accepts named arguments for direct development use and the in-development OGC Application Package invocation style:

```bash
./run.sh \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615"
```

For local-only testing, disable direct bucket access with `DIRECT_BUCKET_ACCESS=false`:

```bash
DIRECT_BUCKET_ACCESS=false ./run.sh "2025-05-01T00:00:00Z" "2025-05-31T23:59:59Z" "500000 5000000 600000 5100000" "EPSG:32615"
```

### Direct Python invocation

Direct Python invocation is useful for development. Unlike `run.sh`, `main.py` defaults to HTTPS mode unless `--direct_bucket_access` is passed.

HTTPS-backed local path:

```bash
uv run main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615" \
  --output_dir "/tmp/output"
```

Direct S3 path matching DPS behavior:

```bash
uv run main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615" \
  --output_dir "/tmp/output" \
  --direct_bucket_access
```

### Smoke test

The repo includes a bounded smoke test that exercises the DPS wrapper through the HTTPS path:

```bash
./smoketest.sh
```

That script is equivalent to running `run.sh` with `DIRECT_BUCKET_ACCESS=false`. This is the intended place to set direct bucket access to false.

## Parameters

- `--start_datetime`: Start datetime in ISO format.
- `--end_datetime`: End datetime in ISO format.
- `--bbox`: Bounding box coordinates (`xmin ymin xmax ymax`).
- `--crs`: CRS definition for the bounding box coordinates. Must use meter units.
- `--output_dir`: Directory where output files will be saved.
- `--direct_bucket_access`: Developer-level `main.py` flag for direct S3 reads. `run.sh` passes this automatically for DPS; local smoke tests omit it via `DIRECT_BUCKET_ACCESS=false`.

## Output contract

A successful run writes:

- `output/<band>.tif` for each requested band
- `output/item.json`

The default output bands are `red`, `green`, `blue`, `nir_narrow`, `swir_1`, and `swir_2`.
