#!/usr/bin/env -S bash --login
set -euo pipefail

# Local smoke test path only: use HTTPS-backed reads instead of the DPS direct-S3 path.
DIRECT_BUCKET_ACCESS=false ./run.sh \
  "2025-05-01T00:00:00Z" \
  "2025-05-31T23:59:59Z" \
  "500000 5000000 550000 5050000" \
  "EPSG:32615"
