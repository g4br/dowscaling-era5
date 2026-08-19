#!/usr/bin/env python3

# ============================================================
# F4 — Motor de extracao (operador UNICO = celula-contenedora, nearest)
# Modulo reutilizado por F5 (matriz de treino) e F7 (inferencia).
# Regra cardinal: estacao (treino) e celula (inferencia) leem o MESMO
# valor por .sel(method='nearest') na grade canonica de 1 km. NENHUM
# caminho bilinear na lon/lat exata; NENHUMA derivacao no ponto
# (tudo ja materializado em F2/F3). T_ERA5_bilinear = t2m_1km na celula.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
from pyproj import Transformer
from _root import ROOT, DIR_ESTATICAS


# ============================================================
# CONFIGURACOES
# ============================================================

PASTA_EST  = DIR_ESTATICAS
PASTA_ERA5 = Path(ROOT + '/saidas/era5land_interp1km_celulas')

CRS_GRADE = 'EPSG:31982'   # UTM 22S, grade canonica cop1km


# ============================================================
# REPROJECAO (uma vez, WGS84 -> UTM)
# ============================================================

def estacoes_para_utm(lons_wgs, lats_wgs):
    trans = Transformer.from_crs('EPSG:4326', CRS_GRADE, always_xy=True)
    xs, ys = trans.transform(np.asarray(lons_wgs, float), np.asarray(lats_wgs, float))
    return np.asarray(xs), np.asarray(ys)


# ============================================================
# OPERADOR UNICO DE EXTRACAO
# ============================================================

def amostrar_raster(da_raster, xs_utm, ys_utm):
    # celula-contenedora vetorizada: 1 valor por estacao. UNICO operador.
    return da_raster.sel(
        x=xr.DataArray(xs_utm, dims='estacao'),
        y=xr.DataArray(ys_utm, dims='estacao'),
        method='nearest').values


# ============================================================
# CAMADAS ESTATICAS (F2)
# ============================================================

def carregar_estaticas():
    # globa *_1km.tif single-band; fracoes e dist_oceano entram a parte.
    camadas = {}
    for tif in sorted(PASTA_EST.glob('*_1km.tif')):
        if tif.stem.startswith('fracoes') or tif.stem.startswith('dist_oceano'):
            continue
        camadas[tif.stem] = rioxarray.open_rasterio(tif, masked=True).squeeze()
    camadas['dist_oceano'] = rioxarray.open_rasterio(
        PASTA_EST / 'dist_oceano_1km_alinhado.tif', masked=True).squeeze()
    return camadas


def carregar_fracoes():
    # bandas 1-8 = T1..T8; banda 9 (fracao_observada) DESCARTADA.
    da = rioxarray.open_rasterio(PASTA_EST / 'fracoes_1km_alinhado.tif', masked=True)
    return da.isel(band=slice(0, 8))


def amostrar_estaticas_estacoes(camadas, da_fracoes, xs_utm, ys_utm):
    out = {nome: amostrar_raster(da, xs_utm, ys_utm) for nome, da in camadas.items()}
    frac = da_fracoes.sel(x=xr.DataArray(xs_utm, dims='estacao'),
                          y=xr.DataArray(ys_utm, dims='estacao'),
                          method='nearest').values   # (8, estacao)
    for b in range(frac.shape[0]):
        out[f'fracao_t{b + 1}'] = frac[b]
    return out


# ============================================================
# STACK ERA5 1 km (F3) — CRS nao auto-detectavel; fixar ao abrir
# ============================================================

def abrir_era5_stack(ano, mes):
    ds = xr.open_dataset(PASTA_ERA5 / f'era5land_interp1km_celulas_{ano}_{mes:02d}.nc')
    if 'spatial_ref' in ds.data_vars:        # vem como data_var no NetCDF
        ds = ds.set_coords('spatial_ref')
    return ds.rio.write_crs(CRS_GRADE)        # pendencia F3: rio.crs=None


def amostrar_era5_ds(ds, xs_utm, ys_utm):
    # amostra todas as data_vars no instante/fatia JA selecionada.
    # IMPORTANTE: .load() antes do nearest vetorizado — sobre o array lazy
    # do netCDF a selecao pontual method='nearest' e patologicamente lenta;
    # carregada em memoria fica instantanea. Passar uma fatia pequena
    # (1 instante ~25 MB), nunca o stack inteiro (744 h ~21 GB).
    ds = ds.load()
    return {v: amostrar_raster(ds[v], xs_utm, ys_utm) for v in ds.data_vars}


def amostrar_era5_estacoes(ano, mes, instante, xs_utm, ys_utm):
    ds = abrir_era5_stack(ano, mes).sel(time=instante)
    return amostrar_era5_ds(ds, xs_utm, ys_utm)


def amostrar_era5_celulas(ds, iys, ixs):
    # Caminho RAPIDO p/ F5/F7: indexacao inteira direta nas celulas (iy,ix)
    # da grade canonica. Na grade cop1km (identica entre estaticas/ERA5),
    # nearest no centro da celula == arr[iy, ix] — provado no gate F4.
    # Carrega 1 var por vez (.compute() = copia temporaria, NAO muta o ds nem
    # acumula as 16 vars na memoria) e indexa em numpy. Usa .compute()/.values
    # ao inves do isel vetorizado, que cai na armadilha de performance do
    # array lazy do netCDF. 3D -> (time, ncel); 2D -> (ncel,).
    iys = np.asarray(iys); ixs = np.asarray(ixs)
    return {v: ds[v].compute().values[..., iys, ixs] for v in ds.data_vars}
