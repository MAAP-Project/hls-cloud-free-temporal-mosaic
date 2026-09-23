# HLS Cloud-Free Temporal Mosaic

Create cloud-free composite images from temporal mosaics of HLS granules using the HLS STAC geoparquet archive and `lazycogs`.

## About

The algorithm queries HLS STAC records directly from parquet files in S3, reads the HLS COG assets, masks cloud and cloud-shadow pixels, and computes a median composite across time. It writes Cloud Optimized GeoTIFFs with STAC metadata for the requested bounding box and date range.

The HLS STAC geoparquet archive is experimental and can lag CMR by a few days. See the [archive repository](https://github.com/MAAP-Project/hls-stac-geoparquet-archive) for details.

## MAAP deployment

The CWL is the sole MAAP registration interface:

```text
hls-cloud-free-temporal-mosaic.cwl
```

Release automation registers that OGC Application Package and points it at the matching standalone, versioned image:

```text
ghcr.io/maap-project/hls-cloud-free-temporal-mosaic:v0.3.3
```

MAAP does not build the image or install this repository from a legacy descriptor. The CWL invokes `main.py` in the image directly.

## Build, run, and test

Install the locked development environment for local work:

```bash
uv sync --frozen
```

For direct Python development, `main.py` defaults to HTTPS. Add `--direct_bucket_access` to exercise the deployed direct-S3 path:

```bash
uv run --frozen main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-05-31T23:59:59Z" \
  --bbox "500000 5000000 600000 5100000" \
  --crs "EPSG:32615" \
  --output_dir /tmp/hls-output
```

The standalone image is built and published by release automation. A local `uv sync` is only for development; it is not a MAAP registration or image-release step.

## Credentials and access modes

The parquet query uses DuckDB's AWS credential chain to access the MAAP-hosted archive.

- **Deployed DPS / OGC jobs:** `direct_bucket_access` defaults to `true`, reading `s3://lp-prod-protected/...` through an authenticated `S3Store` in `us-west-2`. The store refreshes short-lived LP DAAC credentials through `MAAP().aws.earthdata_s3_credentials(...)`, so DPS must provide the MAAP authentication context, including `MAAP_PGT` where required.
- **Local CWL runs:** override `direct_bucket_access` to `false` to read LP DAAC URLs through an authenticated `HTTPStore`. The local HTTPS path does not use the MAAP credential proxy and requires Earthdata username/password credentials.

For example, with Docker available, create a local job file:

```yaml
# local-job.yml
start_datetime: "2025-05-01T00:00:00Z"
end_datetime: "2025-05-31T23:59:59Z"
bbox: "500000 5000000 550000 5050000"
crs: "EPSG:32615"
direct_bucket_access: false
```

Then pass Earthdata credentials through the CWL runner:

```bash
export EARTHDATA_USERNAME="your-earthdata-username"
export EARTHDATA_PASSWORD="your-earthdata-password"
uvx --from cwltool cwltool \
  --preserve-environment EARTHDATA_USERNAME \
  --preserve-environment EARTHDATA_PASSWORD \
  hls-cloud-free-temporal-mosaic.cwl local-job.yml
```

The direct-S3 DPS path is not a local replacement for Earthdata credentials: it assumes a MAAP-authenticated DPS runtime and the LP DAAC bucket's `us-west-2` region. Do not put Earthdata credentials, MAAP tokens, or other secrets in the CWL or job inputs.

## Submit an OGC job

A deployed process selects its release, so job submission uses `submit_job(process_id, inputs, queue)` rather than the deprecated legacy API or an algorithm version argument:

```python
from datetime import UTC, datetime, timedelta

from maap.maap import MAAP

maap = MAAP()

# locate the process ID
response = maap.list_algorithms()
response.raise_for_status()

process_id = next(
   (
       process["processID"]
       for process in response.json()["processes"]
       if process["title"] == "HLS Cloud-Free Temporal Mosaic"
       and process["version"] == "0.3.3"
   ),
   None,
)
if process_id is None:
   raise ValueError("algorithm not found")

start = datetime(2026, 6, 1, tzinfo=UTC)
end = datetime(2026, 7, 1, tzinfo=UTC) - timedelta(seconds=1)

full_bbox = (-105000, 2264000, 566000, 2937000)

bboxes = []
resolution = 30
grid_len_pixels = 8192
grid_len_meters = resolution * grid_len_pixels
xmin_orig, ymin_orig = full_bbox[:2]

xmin_start = xmin_orig - xmin_orig % grid_len_meters
ymin_start = ymin_orig - ymin_orig % grid_len_meters

xmin = xmin_start

while xmin < full_bbox[2]:
    xmax = xmin + grid_len_meters
    ymin = ymin_start

    while ymin < full_bbox[3]:
        ymax = ymin + grid_len_meters
        bboxes.append((xmin, ymin, xmax, ymax))
        ymin = ymax

    xmin = xmax

jobs = (
    {
        "inputs": {
            "start_datetime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_datetime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bbox": " ".join(str(coord) for coord in bbox),
            "crs": "EPSG:5070",
        },
        "tag": "group-A" if not i % 2 else "group-B"
    }
    for i, bbox in enumerate(bboxes)
)

# run the first job
response = maap.submit_job(
    process_id=process_id,
    queue="maap-dps-worker-32gb",
    **next(jobs)
)
response.raise_for_status()

# run the rest
for job in jobs:
    response = maap.submit_job(
        process_id=process_id,
        queue="maap-dps-worker-16gb",
        **job
    )
    response.raise_for_status()
```

## Inputs

- `start_datetime`: HLS query start in ISO format.
- `end_datetime`: HLS query end in ISO format.
- `bbox`: `xmin ymin xmax ymax` in the supplied CRS.
- `crs`: CRS for the bounding box. It must use meter units.

## Output contract

The implementation creates one COG per requested band and writes a self-contained STAC catalog rooted at the output directory. A successful run contains:

```text
output/
├── catalog.json
├── hls-cloud-free-temporal-mosaic/
│   ├── collection.json
│   └── <item-id>/
│       └── <item-id>.json
├── red.tif
├── green.tif
├── blue.tif
├── nir_narrow.tif
├── swir_1.tif
└── swir_2.tif
```

The hierarchy is `Catalog -> Collection -> Item`. The collection has stable algorithm metadata, global/open extent, `item_assets` definitions for the output COG bands, links to this repository and the HLSL30 2.0 and HLSS30 2.0 source collections, and the STAC `other` license value because this output contract does not assert a redistribution license. The item ID is derived from the projected bounding box and date range, and the item links to the band assets.

## Release and recovery

A release requires all of the following:

- `RELEASE_PLEASE_TOKEN` configured as a GitHub secret with permission for Release Please to create releases and release PRs.
- `MAAP_TOKEN` configured as a repository or protected `production` environment secret for MAAP deployment.
- The protected `production` environment enabled, with its required approval reviewers available.
- The versioned GHCR package readable by MAAP workers. Make the package public or configure an equivalent pull path before submitting jobs.

A release publishes the versioned GHCR image and registers the release CWL. The CWL and image tag must stay aligned; do not retag an image that MAAP already uses.

If a release fails, first check whether image publication or MAAP registration failed. Fix the release metadata, token, environment approval, or GHCR visibility as appropriate, then rerun the failed release workflow. If the release PR contains the wrong version or image tag, correct it before merging and create a new release rather than overwriting an existing image. After recovery, validate the registered process and submit one small OGC smoke job before starting a larger batch.
