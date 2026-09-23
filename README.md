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

```python
from datetime import UTC, datetime, timedelta

from rustac import DuckdbClient

from main import hls_geoparquet_hrefs, hls_item_tile_id

client = DuckdbClient()
aoi_bbox = (-92.2, 40.0, -91.0, 41.0)
overall_start = datetime(2024, 12, 1, tzinfo=UTC)
overall_end = datetime(2025, 3, 1, tzinfo=UTC)
tile_ids = set()

for collection in ("HLSL30_2.0", "HLSS30_2.0"):
    for href in hls_geoparquet_hrefs(collection, overall_start, overall_end):
        items = client.search(
            href,
            bbox=aoi_bbox,
            datetime=f"{overall_start.isoformat()}/{overall_end.isoformat()}",
        )
        tile_ids.update(
            tile_id
            for item in items
            if (tile_id := hls_item_tile_id(item.get("id", "")))
        )

month_start = overall_start
while month_start < overall_end:
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    for tile_id in sorted(tile_ids):
        submit_job(
            tile_id=tile_id,
            start_datetime=month_start.strftime("%Y-%m-%dT00:00:00Z"),
            end_datetime=next_month.strftime("%Y-%m-%dT00:00:00Z"),
        )
    month_start = next_month
```

If the AOI is a polygon, use `intersects` where supported or polygon-filter the bbox candidates before submission. See `hls-cloud-free-temporal-mosaics.ipynb` for the same workflow with MAAP job submission.

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
