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

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start_datetime)
            start_datetime="$2"
            shift 2
            ;;
        --end_datetime)
            end_datetime="$2"
            shift 2
            ;;
        --bbox)
            bbox_xmin="$2"
            bbox_ymin="$3"
            bbox_xmax="$4"
            bbox_ymax="$5"
            shift 5
            ;;
        --crs)
            crs="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --start_datetime <datetime> --end_datetime <datetime> --bbox <xmin> <ymin> <xmax> <ymax> --crs <crs>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "${start_datetime:-}" ]] || [[ -z "${end_datetime:-}" ]] || \
   [[ -z "${bbox_xmin:-}" ]] || [[ -z "${bbox_ymin:-}" ]] || \
   [[ -z "${bbox_xmax:-}" ]] || [[ -z "${bbox_ymax:-}" ]] || \
   [[ -z "${crs:-}" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --start_datetime <datetime> --end_datetime <datetime> --bbox <xmin> <ymin> <xmax> <ymax> --crs <crs>"
    exit 1
fi

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
    --bbox ${bbox_xmin} ${bbox_ymin} ${bbox_xmax} ${bbox_ymax} \
    --crs "${crs}" \
    --output_dir="${OUTPUT_DIR}" 
