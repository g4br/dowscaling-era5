# Bundled derived data (`dados/`)

Derived artifacts the pipeline needs as **inputs**, bundled so the repo runs the
model stages (F5→F8) without regenerating F0–F2. **None of it is raw ERA5-Land or
raw station data** — those stay out (see the root `README.md`). Everything here is
built by the scripts in the parent directory from the raw sources; this file
records **how each layer was generated (GDAL / Python)** so a bundled `.tif` /
`.parquet` is never a black box.

Canonical grid for every raster: **EPSG:31982 (UTM 22S), 560 × 807 at 1 km**,
`nodata = -9999.0` (`grid_1km/grid_1km.json`). The 90 m → 1 km rule is a **fixed
11 × 11 block aggregate** with water masked (`skipna`); the GDAL equivalents below
reproduce the same reduction.

Paths are read via `_root.py` (`DIR_GRID`, `DIR_ESTATICAS`, `DIR_MATRIZ`);
override with the env vars of the same name. Re-running the generators writes
back into these same folders.

---

## `grid_1km/` — canonical grid — **F0** (`f0_grade_canonica.py`)

`cop1km.tif` (= `z_alvo`, mean elevation) and `z_std_1km.tif` (sub-grid std) are
the Copernicus GLO-90 DEM (`dem_90m_utm.tif`) aggregated 11 × 11 with `rioxarray`
`.coarsen(...).mean()/.std()` (water masked, so a coastal cell averages only its
land sub-pixels). `centros_wgs84.npz` are the cell-center coordinates (pyproj
UTM→WGS84); `mascara_sc_1km.tif` is the IBGE SC state boundary rasterized on the
grid; `z_era5_orog_1km.tif` is the ERA5-Land invariant geopotential regridded to
1 km; `celulas_estacoes*.npz` (F3setup) and the `*.json` are grid metadata.

Equivalent GDAL for the elevation aggregate:
```bash
# mean 90 m -> 1 km on the canonical grid (z_alvo = cop1km)
gdalwarp -t_srs EPSG:31982 -tr 1000 1000 -r average \
         -te 96590.908 6675590.842 902590.900 7234590.837 \
         dem_90m_utm.tif cop1km.tif
# sub-grid std (z_std): gdaldem/gdal_calc over the 90 m tile per 1 km block,
# or the rioxarray .coarsen(...).std() the script uses.
```

## `estaticas_1km/*_1km.tif` — topographic predictors — **F2b** (`f2b_agrega_1km.py`)

19 single-band layers: `slope_{mean,max}`, `tri_{mean,std,max}`, `z_std`,
`northness_mean`, `eastness_mean`, `svf_{mean,p10}`, `mrvbf_p90`,
`hand_{mean,p10}`, and DEV rings `dev_r{300,1000,3000}_{mean,p10}`. All are the
corresponding **90 m feature raster** aggregated 11 × 11 (`mean/max/std/p10/p90`,
water-masked). Northness/eastness are `cos/sin(aspect)` decomposed **before**
aggregating and zeroed on flats (slope < 0.5°). DEV rings are focal
(elevation − ring-mean)/ring-std via NaN-aware `scipy.signal.fftconvolve`.

The 90 m feature rasters feeding this step come from the GLO-90 DEM via **GDAL**
and **WhiteboxTools**:
```bash
gdaldem slope  dem_90m_utm.tif slope_90m.tif  -compute_edges
gdaldem aspect dem_90m_utm.tif aspect_90m.tif -compute_edges
gdaldem TRI    dem_90m_utm.tif tri_90m.tif    -compute_edges
# svf_90m.tif, mrvbf_90m.tif via WhiteboxTools (sky-view factor, MRVBF)
```

## `estaticas_1km/hand_{mean,p10}_1km.tif` — HAND — **F2a3** (`f2a3_hand_wbt.py`)

Height Above Nearest Drainage, WhiteboxTools chain on the GLO-90 DEM:
`BreachDepressionsLeastCost` → `D8Pointer` → `D8FlowAccumulation` →
`ExtractStreams` (thresh 2000 cells) → `ElevationAboveStream` (on the
**breach-conditioned** DEM). Aggregated to 1 km in F2b (`hand mean` / `p10`).
`hand_*` carry ~15 % `nodata` inland by construction (see
`../matriz/cobertura_features.json`); LightGBM routes the NaNs, so they are kept.

## `estaticas_1km/dist_oceano_1km_alinhado.tif` — distance to coast — `Dados/build_dist_oceano_*.py`

Per-cell great-distance from the 1 km cell center to the GSHHS high-res Atlantic
coastline (L1 land/ocean boundary), computed in EPSG:31982 with a `scipy`
`cKDTree` over coastline vertices densified to 100 m, then written on the
canonical grid (`_alinhado` = the buffered variant so coastal/island cells get a
real distance instead of `nodata`).

## `estaticas_1km/fracoes_1km_alinhado.tif` — thermal land-cover fractions — **F2c** (`f2c_fracoes_1km.p
y`)

9 bands: thermal groups **T1..T8** + `fracao_observada`. Each MapBiomas 30 m class
is mapped to a thermal group (`Dados/mapbiomas/de_para_mapbiomas_termico.csv`);
each group becomes a 0/1 mask at 30 m and is resampled to the cop1km grid by
**area average** (`rasterio.warp.reproject`, `Resampling.average`) in a single
pass, giving each group's areal fraction per 1 km cell. Denominator is
NaN/`Não Observado`-aware, so `T1..T8` sum to 1 where observed.

Equivalent GDAL per group mask:
```bash
gdal_calc.py -A brazil_coverage_2023.tif --calc="(A==3)|(A==4)|(A==5)" \
             --outfile=grupo_T3_30m.tif --type=Byte
gdalwarp -t_srs EPSG:31982 -tr 1000 1000 -r average \
         -te 96590.908 6675590.842 902590.900 7234590.837 \
         grupo_T3_30m.tif grupo_T3_1km.tif   # fraction of the cell in group T3
```

## `matriz/matriz_treino.parquet` — training matrix — **F5** (`f5_matriz_treino.py`)

The single table F6/F7/F7q/F8b consume: **57 features** (16 ERA5 + 31 static +
4 temporal + 6 t2m deltas) with target `y_residuo = T_obs − T_ERA5(bilinear)`.
Built by joining the static layers above (sampled once per cell at the center)
with the monthly ERA5-Land 1 km stack (F3) at the training-station cells, over
2020–2025. It embeds **derived station residuals** (not raw station series) — it
is included here by explicit choice so training runs offline. Sidecars:
`cobertura_features.json` (row/feature/NaN inventory) and
`diagnostico_representatividade.csv` (cell `z_alvo` vs true station altitude).

---

**Not bundled** (regenerable, not consumed by the shipped F0–F8 path): the
per-year `fracoes_1km_alinhado_<ano>.tif` (F2c `--ano`, experiment-only) and any
`*.bak` backups. The monthly ERA5 1 km stack and the trained models are written
under `DOWNSCALING_ROOT/saidas/` and are **not** part of `dados/`.
