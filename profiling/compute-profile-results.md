# Compute profiling results

Date of runs: 2026-09-23. The profiling was local, isolated per synthetic case, and did not touch the notebook or submit a remote job.

## Method

The reproducible harness is `profiling/compute_profile.py`. It loads the historical exporter from `git show main:main.py` without changing the worktree, and compares it with the current exporter.

Synthetic commands:

```bash
uv run --frozen python profiling/compute_profile.py baseline \
  --size 256 --time-count 16 --bands red,green,blue --workers 2
uv run --frozen python profiling/compute_profile.py fixed \
  --size 256 --time-count 16 --bands red,green,blue --workers 2
uv run --frozen python profiling/compute_profile.py baseline \
  --size 1024 --time-count 24 \
  --bands red,green,blue,nir_narrow,swir_1,swir_2 --workers 2
uv run --frozen python profiling/compute_profile.py fixed \
  --size 1024 --time-count 24 \
  --bands red,green,blue,nir_narrow,swir_1,swir_2 --workers 2
```

Synthetic inputs are deterministic Dask-backed `int16` spectral arrays, a shared delayed Fmask array, and daily temporal coordinates. Each mode ran in a separate subprocess. `ru_maxrss` is process peak RSS on Linux; it includes imports, source arrays, Dask graphs, and output arrays, and is not a precise container working-set measurement.

Environment: Python 3.13.14, Dask 2026.8.0, xarray 2026.4.0, rioxarray 0.20.0, rasterio 1.4.3, lazycogs 0.7.0, rustac 0.9.17, obstore 0.9.5. The process saw 16 CPUs and approximately 7443 MiB available memory; no cgroup memory files were present, so the application used its host-memory fallback. Every exporter comparison used `scheduler="threads", num_workers=2` for the fixed boundary.

## Synthetic results

| case | mode | max RSS | elapsed | Dask tasks | shared Fmask materializations |
| --- | --- | ---: | ---: | ---: | ---: |
| 256² × 16 × 3 bands | baseline | 224.4 MiB | 0.108 s | 90 | 3 |
| 256² × 16 × 3 bands | fixed | 221.7 MiB | 0.072 s | 68 | 1 |
| 1024² × 24 × 6 bands | baseline | 896.0 MiB | 1.427 s | 4038 | 96 |
| 1024² × 24 × 6 bands | fixed | 894.2 MiB | 1.439 s | 2838 | 16 |

At 1024², the four-by-four spatial Fmask chunks were materialized once per band in the baseline and once overall in the fixed exporter. The synthetic CPU timings do not show a meaningful speedup: the fixed path removes work but combines it into one larger Dask computation. This synthetic setup does not model object-store latency.

The stage measurements at 1024² were:

| mode | Dask median/computation | COG writes | historical per-band Dask calls |
| --- | ---: | ---: | ---: |
| baseline | 1.116 s total | 1.393 s total | 6, each with only `traverse=False` |
| fixed | 1.227 s | 0.189 s total | none |

The fixed exporter recorded one Dask call with `scheduler="threads", num_workers=2`. The baseline's final configured `dask.compute` call operated on `None` write results and did no pixel work; the actual per-band computations occurred inside rioxarray/Dask with no configured worker count.

## Limited real-data profile

Input was the documented tile and month example narrowed to a short interval:

- tile: `T15TYJ`
- interval: `2025-05-01T00:00:00Z` through `2025-05-16T00:00:00Z`
- bands: red, green, blue
- centered native-grid subset: 384 × 384 pixels at 30 m
- workers: 2
- matched items: 4 `HLSL30_2.0`, 6 `HLSS30_2.0`
- resulting stack: 7 time steps × 3 bands × 384 × 384

The production direct-S3 archive path was attempted first but was blocked by the local DuckDB credential-chain configuration. No credentials were requested or printed. The successful bounded profile used the existing authenticated HTTPS LP DAAC store path, with the public HTTPS archive URL only as a transport fallback for the GeoParquet query. Environment configuration was recorded only as present/not present; `MAAP_PGT`, `EARTHDATA_USERNAME`, and `EARTHDATA_PASSWORD` were present.

Three identical runs were used to expose cache/network variability:

| stage | run range |
| --- | ---: |
| discovery | 22.8–29.5 s |
| header/store setup | 2.3–4.1 s |
| lazy open and stack construction | 61.3–66.6 s |
| graph construction | 0.02 s |
| fixed Dask compute, including remote pixel reads and median | 54.3–66.0 s |
| COG writes | 0.062–0.072 s |
| process peak RSS | 945–997 MiB |

All outputs were reported with the COG layout marker. Lazycogs debug diagnostics during the final run showed 28 pixel chunk reads and 28 matching per-chunk DuckDB searches:

- 7 reads/searches for each spectral band (`B02`, `B03`, `B04`)
- 7 reads/searches for `Fmask`

This is the expected band × time expansion for the current `band: 1` chunking, not proof that those reads are redundant. It does show that the small real run was dominated by archive queries and remote COG reads, not median arithmetic or COG creation. Lazy opening itself also spent about a minute on metadata and time-step inspection before pixel reads began.

## Conclusions and next measurement

- The fixed exporter controls the actual median computation with the requested worker count and removes repeated Fmask materialization.
- Simultaneous final arrays did not increase measured peak RSS in the synthetic comparison. Six full native 3660 × 3660 `int16` outputs are approximately 153 MiB before intermediate/read buffers; the current worker estimator does not explicitly reserve this fixed output footprint.
- The limited HTTPS real run cannot explain a full deployed tile/month directly. It does demonstrate that network/archive access can dominate the compute boundary and that COG writing is negligible at this subset size.
- The next useful measurement is the same 384–1024 pixel profile through the deployed-like direct-S3 path with valid native credentials, followed by one 1024-pixel subset with the same number of observations. Capture per-band/per-time read latency and memory before considering any lazycogs chunking or query changes.
