# HLS Cloud-Free Temporal Mosaic

Create cloud-free composite images from temporal mosaics of HLS granules using the HLS STAC geoparquet archive and `lazycogs`.

## About

The algorithm queries HLS STAC records directly from parquet files in S3, reads the HLS COG assets, masks cloud and cloud-shadow pixels, and computes a median composite across time. It writes Cloud Optimized GeoTIFFs with STAC metadata for the requested bounding box and date range.

The HLS STAC geoparquet archive is experimental and can lag CMR by a few days. See the [archive repository](https://github.com/MAAP-Project/hls-stac-geoparquet-archive) for details.

## MAAP deployment

The CWL is the sole MAAP registration interface:

```text
dps/hls-cloud-free-temporal-mosaic.cwl
```

Release automation registers that OGC Application Package and points it at the matching standalone, versioned image:

```text
ghcr.io/maap-project/hls-cloud-free-temporal-mosaic:v0.2.0
```

MAAP does not build the image or install this repository from a legacy descriptor. The `run.sh` wrapper remains useful for local runs and for the container command invoked by the CWL.

## Build, run, and test

Install the locked development environment for local work:

```bash
uv sync --frozen
```

The wrapper keeps the historical four positional inputs for local and existing DPS callers:

```bash
./run.sh \
  "2025-05-01T00:00:00Z" \
  "2025-05-31T23:59:59Z" \
  "500000 5000000 600000 5100000" \
  "EPSG:32615"
```

It also accepts the named input form used by the OGC package:

```bash
./run.sh \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615"
```

Production DPS runs use direct LP DAAC S3 access by default. The local smoke-test path deliberately uses HTTPS instead:

```bash
./smoketest.sh
# equivalent to: DIRECT_BUCKET_ACCESS=false ./run.sh ...
```

For direct Python development, `main.py` defaults to HTTPS. Add `--direct_bucket_access` to exercise the deployed direct-S3 path:

```bash
uv run --frozen main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox 500000 5000000 600000 5100000 \
  --crs "EPSG:32615" \
  --output_dir /tmp/hls-output
```

The standalone image is built and published by release automation. A local `uv sync` is only for development; it is not a MAAP registration or image-release step.

## Credentials and access modes

The parquet query uses DuckDB's AWS credential chain to access the MAAP-hosted archive.

- **Deployed DPS / OGC jobs:** use `direct_bucket_access=True` and read `s3://lp-prod-protected/...` through an authenticated `S3Store`. This path is intended for DPS workers in `us-west-2` and uses `NasaEarthdataCredentialProvider` for short-lived LP DAAC credentials.
- **Local HTTPS smoke tests:** use `direct_bucket_access=False` and read LP DAAC URLs through an authenticated `HTTPStore`. Set credentials before running the smoke test:

  ```bash
  export EARTHDATA_USERNAME="your-earthdata-username"
  export EARTHDATA_PASSWORD="your-earthdata-password"
  ./smoketest.sh
  ```

Do not put Earthdata credentials, MAAP tokens, or other secrets in the CWL or job inputs. The only supported reason to set `DIRECT_BUCKET_ACCESS=false` is local testing where direct S3 is unavailable.

## Submit an OGC job

A deployed process selects its release, so job submission uses `submit_job(process_id, inputs, queue)` rather than the deprecated legacy API or an algorithm version argument:

```python
from datetime import UTC, datetime, timedelta

from maap.maap import MAAP

maap = MAAP()
start = datetime(2025, 5, 1, tzinfo=UTC)
end = datetime(2025, 6, 1, tzinfo=UTC) - timedelta(seconds=1)

response = maap.submit_job(
    process_id="hls_cloud_free_temporal_mosaic",
    inputs={
        "start_datetime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_datetime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": "500000 5000000 600000 5100000",
        "crs": "EPSG:32615",
    },
    queue="maap-dps-worker-16gb",
    tag="demo",
)
response.raise_for_status()
print(response.headers["Location"])
```

## Inputs

- `start_datetime`: HLS query start in ISO format.
- `end_datetime`: HLS query end in ISO format.
- `bbox`: `xmin ymin xmax ymax` in the supplied CRS.
- `crs`: CRS for the bounding box. It must use meter units.

## Output contract

The implementation creates one COG per requested band and then calls `Catalog.normalize_and_save()` with a self-contained catalog rooted at the output directory. With the current `pystac` layout strategy, a successful run contains:

```text
output/
├── catalog.json
├── <item-id>/
│   └── <item-id>.json
├── red.tif
├── green.tif
├── blue.tif
├── nir_narrow.tif
├── swir_1.tif
└── swir_2.tif
```

The item ID is derived from the projected bounding box and date range. The catalog links to the STAC item, and the item links to the band assets. The output is therefore a catalog plus a STAC item; it is not only a promised root-level `item.json`.

## Release and recovery

A release requires all of the following:

- `RELEASE_PLEASE_TOKEN` configured as a GitHub secret with permission for Release Please to create releases and release PRs.
- `MAAP_TOKEN` configured as a repository or protected `production` environment secret for MAAP deployment.
- The protected `production` environment enabled, with its required approval reviewers available.
- The versioned GHCR package readable by MAAP workers. Make the package public or configure an equivalent pull path before submitting jobs.

A release publishes the versioned GHCR image and registers the release CWL. The CWL and image tag must stay aligned; do not retag an image that MAAP already uses.

If a release fails, first check whether image publication or MAAP registration failed. Fix the release metadata, token, environment approval, or GHCR visibility as appropriate, then rerun the failed release workflow. If the release PR contains the wrong version or image tag, correct it before merging and create a new release rather than overwriting an existing image. After recovery, validate the registered process and submit one small OGC smoke job before starting a larger batch.
