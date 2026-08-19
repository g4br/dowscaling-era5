#!/usr/bin/env python3

# ============================================================
# F2-B — Agregacao 90 m -> 1 km na grade canonica cop1km (regra de ouro)
# Hardenings: masked=True em todas as derivadas + skipna=True (agua mascarada),
# north/east mascarados por agua e por plana(<0.5), DEV aneis via fftconvolve,
# salvar() fixa CRS/nodata e limpa attrs.
# D1 = B.3 (aneis): dev_r300/r1000/r3000 x (mean,p10).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import rioxarray
from scipy.signal import fftconvolve
from _root import ROOT, DIR_ESTATICAS


# ============================================================
# CONFIGURACOES
# ============================================================

PASTA_DEM   = Path(ROOT + '/Dados/MDTs')
PASTA_FEIC  = PASTA_DEM / 'feicoes'
PASTA_SAIDA = DIR_ESTATICAS

FATOR_AGREGACAO  = 11
RES_SUB          = 1000.0 / FATOR_AGREGACAO   # ~90.909 m
MASC_PLANA_GRAUS = 0.5
NODATA           = -9999.0
BANDAS_DEV = {'r300': (0, 300), 'r1000': (300, 1000), 'r3000': (1000, 3000)}


# ============================================================
# FUNCOES
# ============================================================

def agrega(da_90m, estatistica):
    bloco = da_90m.coarsen(x=FATOR_AGREGACAO, y=FATOR_AGREGACAO, boundary='trim')
    if estatistica == 'mean':
        return bloco.mean(skipna=True)
    if estatistica == 'std':
        return bloco.std(skipna=True)
    if estatistica == 'max':
        return bloco.max(skipna=True)
    if estatistica.startswith('p'):
        q = float(estatistica[1:])
        return bloco.reduce(lambda a, axis: np.nanpercentile(a, q, axis=axis))
    raise ValueError('Estatistica nao suportada: ' + estatistica)


def salvar(da_1km, nome):
    da = da_1km.copy()
    da.attrs = {}
    da = da.rio.write_crs('EPSG:31982')
    da = da.rio.write_nodata(NODATA, encoded=True)
    da.rio.to_raster(PASTA_SAIDA / (nome + '.tif'))
    v = da.values[np.isfinite(da.values)]
    print('  %-22s min/max = %.3f / %.3f' % (nome, float(v.min()), float(v.max())))


def kernel_anel(raio_int_m, raio_ext_m, resolucao_m):
    raio_ext_px = int(round(raio_ext_m / resolucao_m))
    yy, xx = np.mgrid[-raio_ext_px:raio_ext_px + 1, -raio_ext_px:raio_ext_px + 1]
    dist_m = np.hypot(xx, yy) * resolucao_m
    anel   = ((dist_m > raio_int_m) & (dist_m <= raio_ext_m)).astype(float)
    return anel / anel.sum()


def foco_anel(campo, kernel, valido):
    # media focal NaN-aware via FFT: ignora agua/NoData dentro do anel
    soma_pesos = fftconvolve(valido.astype(float), kernel, mode='same')
    soma_pesos = np.where(soma_pesos < 1e-9, np.nan, soma_pesos)
    soma_val   = fftconvolve(np.where(valido, campo, 0.0), kernel, mode='same')
    return soma_val / soma_pesos


def calcular_dev(z, raio_int_m, raio_ext_m, resolucao_m, valido):
    kernel  = kernel_anel(raio_int_m, raio_ext_m, resolucao_m)
    media   = foco_anel(z,      kernel, valido)
    media_q = foco_anel(z ** 2, kernel, valido)
    desvio  = np.sqrt(np.clip(media_q - media ** 2, 0.0, None))
    dev     = (z - media) / np.where(desvio < 1e-6, np.nan, desvio)
    return np.where(valido, dev, np.nan)


# ============================================================
# CAMINHOS E CHECAGENS
# ============================================================

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

da_dem = rioxarray.open_rasterio(PASTA_DEM / 'dem_90m_utm.tif', masked=True).squeeze()
valido = np.isfinite(da_dem.values)
print('DEM 90 m:', da_dem.shape, '| validos (terra):', int(valido.sum()))


# ============================================================
# SLOPE / TRI / Z_STD
# ============================================================

print('Agregando slope / TRI / z_std...')
da_slope = rioxarray.open_rasterio(PASTA_FEIC / 'slope_90m.tif', masked=True).squeeze()
da_tri   = rioxarray.open_rasterio(PASTA_FEIC / 'tri_90m.tif',   masked=True).squeeze()

salvar(agrega(da_slope, 'mean'), 'slope_mean_1km')
salvar(agrega(da_slope, 'max'),  'slope_max_1km')
salvar(agrega(da_tri,   'mean'), 'tri_mean_1km')
salvar(agrega(da_tri,   'std'),  'tri_std_1km')
salvar(agrega(da_tri,   'max'),  'tri_max_1km')
salvar(agrega(da_dem,   'std'),  'z_std_1km')


# ============================================================
# NORTHNESS / EASTNESS (decompor antes de agregar; mascarar agua + plana)
# ============================================================

print('Northness / eastness...')
da_aspect  = rioxarray.open_rasterio(PASTA_FEIC / 'aspect_90m.tif').squeeze()
aspect_rad = np.deg2rad(da_aspect.values)
north = np.cos(aspect_rad)
east  = np.sin(aspect_rad)

plana = da_slope.values < MASC_PLANA_GRAUS    # graus (NaN no oceano -> comparacao False)
north[plana] = 0.0
east[plana]  = 0.0
north[~valido] = np.nan                       # mascara agua
east[~valido]  = np.nan

salvar(agrega(da_dem.copy(data=north), 'mean'), 'northness_mean_1km')
salvar(agrega(da_dem.copy(data=east),  'mean'), 'eastness_mean_1km')


# ============================================================
# SVF / MRVBF / HAND
# ============================================================

print('SVF / MRVBF / HAND...')
da_svf   = rioxarray.open_rasterio(PASTA_FEIC / 'svf_90m.tif',   masked=True).squeeze()
da_mrvbf = rioxarray.open_rasterio(PASTA_FEIC / 'mrvbf_90m.tif', masked=True).squeeze()
da_hand  = rioxarray.open_rasterio(PASTA_FEIC / 'hand_90m.tif',  masked=True).squeeze()

salvar(agrega(da_svf,   'mean'), 'svf_mean_1km')
salvar(agrega(da_svf,   'p10'),  'svf_p10_1km')
salvar(agrega(da_mrvbf, 'p90'),  'mrvbf_p90_1km')
salvar(agrega(da_hand,  'mean'), 'hand_mean_1km')
salvar(agrega(da_hand,  'p10'),  'hand_p10_1km')


# ============================================================
# DEV EM ANEIS (D1 = B.3) — fftconvolve, agua mascarada
# ============================================================

print('DEV em aneis (D1 = B.3)...')
z_arr = np.where(valido, da_dem.values, np.nan)
for nome, (rin, rext) in BANDAS_DEV.items():
    print('  banda', nome, '(', rin, '-', rext, 'm)...')
    dev = calcular_dev(z_arr, rin, rext, RES_SUB, valido)
    da_dev = da_dem.copy(data=dev)
    salvar(agrega(da_dev, 'mean'), 'dev_' + nome + '_mean_1km')
    salvar(agrega(da_dev, 'p10'),  'dev_' + nome + '_p10_1km')


# ============================================================
# SAIDA
# ============================================================

print('Estaticas topograficas 1 km salvas em:', PASTA_SAIDA)
