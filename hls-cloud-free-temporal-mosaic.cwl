cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
$schemas:
  - >-
    https://raw.githubusercontent.com/schemaorg/schemaorg/refs/heads/main/data/releases/9.0/schemaorg-current-http.rdf
s:author:
  - class: s:Organization
    s:name: MAAP Project
s:codeRepository: https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic
# x-release-please-start-version
s:softwareVersion: 0.3.0
s:version: 0.3.0
# x-release-please-end
s:keywords:
  - HLS
  - cloud-free
  - temporal mosaic
  - MAAP
$graph:
  - class: Workflow
    id: hls_cloud_free_temporal_mosaic
    label: HLS Cloud-Free Temporal Mosaic
    doc: Generate a cloud-free temporal mosaic from HLS data.
    inputs:
      start_datetime:
        label: Start datetime
        doc: Start of the HLS STAC query interval in ISO format.
        type: string
      end_datetime:
        label: End datetime
        doc: End of the HLS STAC query interval in ISO format.
        type: string
      bbox:
        label: Bounding box
        doc: Space-separated bounding box coordinates in CRS coordinates.
        type: string
      crs:
        label: CRS
        doc: Coordinate reference system for the bounding box.
        type: string
      direct_bucket_access:
        label: Direct bucket access
        doc: Read protected LP DAAC assets through S3 rather than HTTPS.
        type: boolean
        default: true
    outputs:
      output:
        type: Directory
        outputSource: process/output
    steps:
      process:
        run: '#main'
        in:
          start_datetime: start_datetime
          end_datetime: end_datetime
          bbox: bbox
          crs: crs
          direct_bucket_access: direct_bucket_access
        out:
          - output
  - class: CommandLineTool
    id: main
    requirements:
      DockerRequirement:
        # x-release-please-start-version
        dockerPull: ghcr.io/maap-project/hls-cloud-free-temporal-mosaic:v0.3.0
        # x-release-please-end
      NetworkAccess:
        networkAccess: true
      ResourceRequirement:
        ramMin: 16384
        coresMin: 4
        outdirMin: 8192
    baseCommand:
      - /app/hls-cloud-free-temporal-mosaic/.venv/bin/python
      - /app/hls-cloud-free-temporal-mosaic/main.py
      - --output_dir
      - output
    successCodes:
      - 0
    inputs:
      start_datetime:
        type: string
        inputBinding:
          position: 1
          prefix: '--start_datetime'
      end_datetime:
        type: string
        inputBinding:
          position: 2
          prefix: '--end_datetime'
      bbox:
        type: string
        inputBinding:
          position: 3
          prefix: '--bbox'
      crs:
        type: string
        inputBinding:
          position: 4
          prefix: '--crs'
      direct_bucket_access:
        type: boolean
        default: true
        inputBinding:
          prefix: '--direct_bucket_access'
    outputs:
      output:
        type: Directory
        outputBinding:
          glob: output
