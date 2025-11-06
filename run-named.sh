#!/usr/bin/env -S bash --login
set -euo pipefail
# This wrapper script accepts named arguments and calls run.sh with positional arguments

# Get current location of script
basedir=$(dirname "$(readlink -f "$0")")

# Initialize variables
start_datetime=""
end_datetime=""
bbox=""
crs=""

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
            bbox="$2"
            shift 2
            ;;
        --crs)
            crs="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Usage: $0 --start_datetime <value> --end_datetime <value> --bbox <value> --crs <value>"
            exit 1
            ;;
    esac
done

# Validate all required arguments are provided
if [[ -z "$start_datetime" ]]; then
    echo "Error: --start_datetime is required"
    exit 1
fi

if [[ -z "$end_datetime" ]]; then
    echo "Error: --end_datetime is required"
    exit 1
fi

if [[ -z "$bbox" ]]; then
    echo "Error: --bbox is required"
    exit 1
fi

if [[ -z "$crs" ]]; then
    echo "Error: --crs is required"
    exit 1
fi

# Call run.sh with positional arguments
"${basedir}/run.sh" "$start_datetime" "$end_datetime" "$bbox" "$crs"
