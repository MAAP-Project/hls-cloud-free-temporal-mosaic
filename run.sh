#!/usr/bin/env -S bash --login
set -euo pipefail
# This script is the one that is called by the DPS.
# Use this script to prepare input paths for any files
# that are downloaded by the DPS and outputs that are
# required to be persisted

# Get current location of build script
basedir=$(dirname "$(readlink -f "$0")")

# Create output directory to store outputs.
# The name is output as required by the DPS.
# Note how we dont provide an absolute path
# but instead a relative one as the DPS creates
# a temp working directory for our code.

mkdir -p output


# DPS downloads all files provided as inputs to
# this directory called input.
# In our example the image will be downloaded here.
INPUT_DIR=input
OUTPUT_DIR=output

# Parse positional arguments
if [[ $# -ne 4 ]]; then
    echo "Error: Expected 4 arguments, got $#"
    echo "Usage: $0 <start_datetime> <end_datetime> <bbox_string> <crs>"
    exit 1
fi

start_datetime="$1"
end_datetime="$2"
bbox="$3"
crs="$4"

# Call the script using the absolute paths
# Use the updated environment when calling 'uv run'
# This lets us run the same way in a Terminal as in DPS
# Any output written to the stdout and stderr streams will be automatically captured and placed in the output dir

# unset PROJ env vars
unset PROJ_LIB
unset PROJ_DATA

UV_PROJECT=${basedir} uv run --no-dev ${basedir}/main.py \
    --start_datetime "${start_datetime}" \
    --end_datetime "${end_datetime}" \
    --bbox ${bbox} \
    --crs "${crs}" \
    --output_dir="${OUTPUT_DIR}" \
    --direct_bucket_access
