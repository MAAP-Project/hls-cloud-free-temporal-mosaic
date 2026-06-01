#!/usr/bin/env -S bash --login
set -euo pipefail
# This script is the MAAP DPS entry point. Keep the four positional arguments
# for backwards compatibility with current MAAP DPS. Named arguments are also
# accepted for the in-development OGC Application Package path.
#
# Production runs always use direct LP DAAC S3 bucket access. The
# DIRECT_BUCKET_ACCESS=false override exists only for local smoke tests where
# direct S3 access is not available or not running from us-west-2.

basedir=$(dirname "$(readlink -f "$0")")
output_dir=output
mkdir -p "${output_dir}"

main_args=()
if [[ $# -eq 4 && "$1" != --* ]]; then
    # Current MAAP DPS invokes run.sh with positional arguments in the order
    # declared in algorithm-config.yml.
    main_args=(
        --start_datetime "$1"
        --end_datetime "$2"
        --bbox $3
        --crs "$4"
    )
else
    # Named arguments are for direct development use and the emerging OGC
    # Application Package invocation style.
    main_args=("$@")
fi

# Default to the production DPS behavior. Only local callers, such as
# smoketest.sh, should disable this with DIRECT_BUCKET_ACCESS=false.
if [[ "${DIRECT_BUCKET_ACCESS:-true}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    main_args+=(--direct_bucket_access)
fi

# unset PROJ env vars
unset PROJ_LIB
unset PROJ_DATA

UV_PROJECT="${basedir}" uv run --no-dev "${basedir}/main.py" \
    "${main_args[@]}" \
    --output_dir="${output_dir}"
