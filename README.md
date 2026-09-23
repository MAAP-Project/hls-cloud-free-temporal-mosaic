# HLS Cloud-Free Temporal Mosaic

Create one cloud-free lower-median composite for one native HLS tile and one calendar-day interval. The workflow queries both `HLSL30_2.0` and `HLSS30_2.0` in the HLS STAC GeoParquet archive, masks HLS Fmask cloud and shadow bits, and writes native-grid COGs plus a self-contained STAC item.

## Contract

Inputs are:

- `start_datetime`: UTC midnight, inclusive.
- `end_datetime`: either the exclusive UTC midnight after the interval, or `23:59:59` UTC on its last included day.
- `tile_id`: an HLS MGRS tile such as `T15TYJ`.

For example, both `2025-05-01` to `2025-06-01` and `2025-05-01` to `2025-05-31T23:59:59Z` include May 1 through May 31. The search uses the normalized UTC interval; STAC metadata retains the precise normalized start and last included second. Partial-day intervals are rejected so the date-only product ID cannot collide for distinct sub-day jobs.

The deterministic item ID is:

```text
hls-composite-T15TYJ-20250501-20250531-lower-median-v1
```

The item uses the STAC MGRS extension for tile identity and stores `hls-composite:method`, `hls-composite:mask`, and source-item lineage. `lower-median-v1` means the current integer lower median with P1D grouping, first-valid daily source selection, and HLS Fmask masking. Any future reducer, grouping, output, or band-subset variant must get a distinct method identity; this implementation does not provide a plugin registry.

The output uses the full native HLS footprint, normally 3660 x 3660 pixels at 30 m, including the product's overlap. CRS, transform, shape, and resolution come from one representative source COG header, never from STAC projection metadata, an inferred MGRS hemisphere, or a reconstructed geographic STAC bbox. The remaining assets are assumed to share that native grid and are read by lazycogs. Downstream reprojection and mosaicking across tiles remains the caller's responsibility.

## MAAP deployment

The CWL is the sole MAAP registration interface:

```text
hls-cloud-free-temporal-mosaic.cwl
```

Release automation registers that OGC Application Package and points it at the matching standalone image. This input-contract refactor does not retag an image or change release metadata.

## Build, run, and test

```bash
uv sync --frozen
uv run --frozen main.py \
  --start_datetime "2025-05-01T00:00:00Z" \
  --end_datetime "2025-06-01T00:00:00Z" \
  --tile_id T15TYJ \
  --output_dir /tmp/hls-output
```

Use `--direct_bucket_access` for the deployed MAAP path. Without it, local runs use authenticated HTTPS and require `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

```yaml
# local-job.yml
start_datetime: "2025-05-01T00:00:00Z"
end_datetime: "2025-06-01T00:00:00Z"
tile_id: "T15TYJ"
direct_bucket_access: false
```

```bash
export EARTHDATA_USERNAME="your-earthdata-username"
export EARTHDATA_PASSWORD="your-earthdata-password"
uvx --from cwltool cwltool \
  --preserve-environment EARTHDATA_USERNAME \
  --preserve-environment EARTHDATA_PASSWORD \
  hls-cloud-free-temporal-mosaic.cwl local-job.yml
```

## Archive discovery and job enumeration

For an AOI, query the archive in both HLS collections across the overall interval, use a tile property when the archive provides one or otherwise extract the documented tile component from each HLS item ID, deduplicate and sort the IDs, then submit one job per tile and calendar month. This produces tiles with observations, not every theoretical MGRS tile. A tile/month job can still have no data and should be handled as a normal empty result.

The archive is partitioned by year and month. The runtime enumerates exact monthly parquet HREFs from the requested interval instead of using a recursive wildcard. It supports STAC `bbox`, datetime, and CQL2 queries. A polygon AOI should use exact intersection when the query service supports it; otherwise use its geographic bbox as a candidate query and over-select tiles, then filter candidates with the polygon locally. Do not build a projected 8192-pixel grid or submit arbitrary bboxes.

Run this example from the repository root in an authenticated MAAP ADE session with the project dependencies installed. Set `HLS_PROCESS_VERSION` in your environment to a deployed release that accepts `tile_id`. Version `0.3.3` uses the old bbox/CRS interface; wait for the native-tile release before submitting these jobs.

**Note:** This example submits a real job for each discovered tile/month on `maap-dps-worker-16gb`. Check the AOI, dates, and queue before running it. Keep the interval aligned to calendar-month boundaries.

```python
import logging
import os
from datetime import UTC, datetime, timedelta

from maap.maap import MAAP

from main import build_duckdb_client, hls_geoparquet_hrefs, item_tile_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

maap = MAAP()
process_version = os.environ["HLS_PROCESS_VERSION"]
response = maap.list_algorithms()
response.raise_for_status()
process_ids = [
    process["processID"]
    for process in response.json()["processes"]
    if process["title"] == "HLS Cloud-Free Temporal Mosaic"
    and process["version"] == process_version
]
if len(process_ids) != 1:
    raise ValueError(f"Expected one deployed process for version {process_version}, got {process_ids}")
process_id = process_ids[0]

client = build_duckdb_client()
aoi_bbox = (-92.2, 40.0, -91.0, 41.0)
overall_start = datetime(2024, 12, 1, tzinfo=UTC)
overall_end = datetime(2025, 3, 1, tzinfo=UTC)
tile_ids = set()

for collection in ("HLSL30_2.0", "HLSS30_2.0"):
    for href in hls_geoparquet_hrefs(collection, overall_start, overall_end):
        items = client.search(
            href,
            bbox=aoi_bbox,
            datetime=f"{overall_start.isoformat()}/{(overall_end - timedelta(microseconds=1)).isoformat()}",
        )
        tile_ids.update(
            tile_id
            for item in items
            if (tile_id := item_tile_id(item))
        )

job_ids = []
month_start = overall_start
while month_start < overall_end:
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    for tile_id in sorted(tile_ids):
        response = maap.submit_job(
            process_id=process_id,
            inputs={
                "tile_id": tile_id,
                "start_datetime": month_start.strftime("%Y-%m-%dT00:00:00Z"),
                "end_datetime": next_month.strftime("%Y-%m-%dT00:00:00Z"),
                "direct_bucket_access": True,
            },
            queue="maap-dps-worker-16gb",
            tag="native-hls-tiles",
        )
        response.raise_for_status()
        job_id = response.json()["jobID"]
        job_ids.append(job_id)
        logger.info("Submitted %s for %s %s", job_id, tile_id, month_start.date())
    month_start = next_month
```

Submission does not mean completion. Save `job_ids` to check the jobs later; rerunning the submission loop can create duplicate jobs. To check their current status without resubmitting:

```python
for job_id in job_ids:
    response = maap.get_job_status(job_id)
    response.raise_for_status()
    logger.info("Job %s: %s", job_id, response.json()["status"])
```

If the AOI is a polygon, use `intersects` where supported or polygon-filter the bbox candidates before submission. See `hls-cloud-free-temporal-mosaics.ipynb` for the notebook workflow; set its process version to the same deployed native-tile release.

Multiple same-collection acquisitions on one day are grouped by `P1D`; lazycogs' default first-valid mosaic is selected in deterministic `datetime,id` order. Spectral and Fmask reads use the same discovered item IDs and grouping. Missing one collection is allowed; no observations across both collections fails clearly.

## Output contract

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

The hierarchy is `Catalog -> Collection -> Item`. The COGs retain the exact native source transform, CRS, shape, and footprint. The output STAC item records precise temporal bounds, resulting native projection metadata, tile identity, composite definition, and discovered source item IDs.
