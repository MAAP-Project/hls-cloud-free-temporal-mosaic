# Changelog

## [0.3.3](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/compare/v0.3.2...v0.3.3) (2026-09-23)


### Bug Fixes

* use maap-py to get LPDAAC S3 creds ([#14](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/14)) ([7ea091d](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/7ea091d3bee20f61ae93293af6b35740831514fe))

## [0.3.2](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/compare/v0.3.1...v0.3.2) (2026-09-22)


### Bug Fixes

* declare all dependencies ([#12](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/12)) ([5b6b75c](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/5b6b75c85e9d3d4711e2a7a2db139a48d1277780))

## [0.3.1](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/compare/v0.3.0...v0.3.1) (2026-09-22)


### Bug Fixes

* include a STAC collection ([#11](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/11)) ([7faf371](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/7faf37166971915441983aa528b42284627a8924))
* parse OGC bbox as one argument ([#9](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/9)) ([75b5092](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/75b509295e7a0d032f7080600dde0cf78bfd7ef5))

## [0.3.0](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/compare/v0.2.0...v0.3.0) (2026-09-21)


### Features

* add named arguments to new run-named.sh script for OGC Application Packages ([#2](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/2)) ([7a1302c](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/7a1302c91d4031cbbf6f4dc95fcd1c20b234053e))
* add notebook showing how to submit jobs and query results ([4caaaf6](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/4caaaf61078bca183327f864375194310875cff8))
* add option to use direct bucket access ([a60eb7c](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/a60eb7cac1cb131aafef0bba6962a065b8fe8d82))
* add retry logic to catch intermittent errors ([983a8be](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/983a8be3878c87819de35a269d471fc46d871249))
* only allow CRS definitions with meters as units ([cb675c1](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/cb675c18e2b1529c522428e9ccad03ad9dc3eb79))
* replace odc-stac with lazycogs ([#6](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/6)) ([8dfd4d4](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/8dfd4d423b2d606befab520f0c893ab59a8f0c99))
* update to use v2 of the HLS STAC Geoparquet ([0595fff](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/0595fffa28dfc12190d30f13c5f97a72e360946b))
* upgrade to v2 of hls-stac-geoparquet-archive ([4132de5](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/4132de52012ceff45acd2ad579a95bfb7402aa05))


### Bug Fixes

* cache the duckdb spatial extension during build ([d92a2db](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/d92a2dbd466b7223712f5d4a7a9924237f42413c))
* convert bbox to string ([e85680c](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/e85680c21e8f9bedb71f5ae316bb90d559e26e16))
* convert to OGC Application package ([#7](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/issues/7)) ([ed09232](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/ed09232bea25ec16a3784fb05b5d26a20fe5b534))
* fix Fmask nodata and set up run.sh with named args ([3d34d10](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/3d34d107e38b5f0cc6c1e6a39381b151c18421e5))
* improve item ids ([18bfd44](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/18bfd4438804c7c678e2c9bda4b4650ba45399f1))
* set cloud_defaults=True in configure_rio ([53060ee](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/53060ee1cbe8458218f9fc7d26ede03a0e14fb4f))
* set proj:transform and proj:shape if missing ([3b60f49](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/3b60f49a63a4447601cb5fbdcc28a4592de8c0a6))


### Documentation

* update readme ([4e0f291](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/4e0f291c7a3236e83abc4afdf4893e2e21d54d6b))
* update readme ([212c954](https://github.com/MAAP-Project/hls-cloud-free-temporal-mosaic/commit/212c954da41797e4e765400770277a7d2279f5f2))
