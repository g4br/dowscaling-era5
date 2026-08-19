#!/usr/bin/env python3

# ============================================================
# F0 — GRADE CANONICA (cop1km = z_alvo), z_std, centros, metadados
# Regra de ouro: dem_90m_utm (agua mascarada) -> coarsen 11x11.
# skipna=True: celula costeira = media SO dos sub-pixels de terra.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import json
import glob
import re
from pathlib import Path

import numpy as np
import rioxarray
from pyproj import Transformer
from _root import ROOT, DIR_GRID


# ============================================================
# CONFIGURACOES
# ============================================================

CAMINHO_DEM_UTM = Path(ROOT + '/Dados/MDTs/dem_90m_utm.tif')
PASTA_GRID      = DIR_GRID
FATOR_AGREGACAO = 11
NODATA          = -9999.0

PASTA_T2M  = ROOT + '/Dados/ERA5_land/t2m'
PASTA_D2M  = ROOT + '/Dados/ERA5_land/d2m'
PASTA_RAD  = ROOT + '/Dados/ERA5_land/rad'
PASTA_STN  = ROOT + '/stn_data/192'


# ============================================================
# FUNCOES
# ============================================================

def coarsen_estatistica(da_90m, fator, estatistica):
    bloco = da_90m.coarsen(x=fator, y=fator, boundary='trim')
    if estatistica == 'mean':
        return bloco.mean(skipna=True)
    if estatistica == 'std':
        return bloco.std(skipna=True)
    raise ValueError('Estatistica nao suportada: ' + estatistica)


def meses_presentes(pasta, regex):
    achados = set()
    for caminho in glob.glob(pasta + '/*.nc'):
        m = re.search(regex, Path(caminho).name)
        if m:
            achados.add(m.group(1) + m.group(2))
    return achados


def meses_estacoes(pasta):
    achados = set()
    for caminho in glob.glob(pasta + '/*.csv'):
        m = re.search(r'_([0-9]{4})([0-9]{2})\.csv', Path(caminho).name)
        if m:
            achados.add(m.group(1) + m.group(2))
    return achados


# ============================================================
# CAMINHOS E CHECAGENS
# ============================================================

if not CAMINHO_DEM_UTM.exists():
    raise FileNotFoundError('DEM UTM ausente (rodar passo A): ' + str(CAMINHO_DEM_UTM))

PASTA_GRID.mkdir(parents=True, exist_ok=True)


# ============================================================
# LEITURA
# ============================================================

print('Lendo DEM 90 m UTM (masked=True -> oceano vira NaN)...')
da_dem_90m = rioxarray.open_rasterio(CAMINHO_DEM_UTM, masked=True).squeeze()


# ============================================================
# PROCESSAMENTO
# ============================================================

print('Agregando 90 m -> 1 km (z_alvo, z_std) com skipna=True...')
da_z_alvo = coarsen_estatistica(da_dem_90m, FATOR_AGREGACAO, 'mean')
da_z_std  = coarsen_estatistica(da_dem_90m, FATOR_AGREGACAO, 'std')

xs_1km = da_z_alvo['x'].values
ys_1km = da_z_alvo['y'].values
X_grid, Y_grid = np.meshgrid(xs_1km, ys_1km)

print('Calculando centros em WGS84 (para regrid do ERA5 em F3)...')
trans_utm_wgs = Transformer.from_crs('EPSG:31982', 'EPSG:4326', always_xy=True)
lon_centros, lat_centros = trans_utm_wgs.transform(X_grid, Y_grid)


# ============================================================
# SAIDA — rasters
# ============================================================

print('Salvando cop1km (= z_alvo), z_std e metadados...')

# limpar attrs herdados do DEM (STATISTICS_* obsoletos) p/ nao gravar metadado conflitante
da_z_alvo.attrs = {}
da_z_std.attrs  = {}

da_z_alvo = da_z_alvo.rio.write_crs('EPSG:31982')
da_z_std  = da_z_std.rio.write_crs('EPSG:31982')
da_z_alvo = da_z_alvo.rio.write_nodata(NODATA, encoded=True)
da_z_std  = da_z_std.rio.write_nodata(NODATA, encoded=True)

da_z_alvo.rio.to_raster(PASTA_GRID / 'cop1km.tif')      # GRADE DE REFERENCIA = z_alvo
da_z_std.rio.to_raster(PASTA_GRID / 'z_std_1km.tif')


# ============================================================
# SAIDA — centros e metadados de grade
# ============================================================

np.savez(PASTA_GRID / 'centros_wgs84.npz',
         X_grid=X_grid, Y_grid=Y_grid,
         lon_centros=lon_centros, lat_centros=lat_centros,
         xs_1km=xs_1km, ys_1km=ys_1km)

meta_grade = {
    'crs':             'EPSG:31982',
    'shape':           [int(da_z_alvo.sizes['y']), int(da_z_alvo.sizes['x'])],
    'res_m':           1000.0,
    'x_min':           float(xs_1km.min()),
    'x_max':           float(xs_1km.max()),
    'y_min':           float(ys_1km.min()),
    'y_max':           float(ys_1km.max()),
    'fator_agregacao': FATOR_AGREGACAO,
    'nodata':          NODATA,
}
with open(PASTA_GRID / 'grid_1km.json', 'w', encoding='utf-8') as f:
    json.dump(meta_grade, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# SAIDA — cobertura temporal e gaps (gate-out F0)
# ============================================================

print('Registrando cobertura temporal ERA5 x estacoes...')

t2m = meses_presentes(PASTA_T2M, r'ERA5land_([0-9]{4})_([0-9]{2})\.nc')
d2m = meses_presentes(PASTA_D2M, r'ERA5land_umid_vento_sp_([0-9]{4})_([0-9]{2})\.nc')
rad = meses_presentes(PASTA_RAD, r'ERA5land_radiacao_([0-9]{4})_([0-9]{2})\.nc')
stn = meses_estacoes(PASTA_STN)

esperados = [f'{a}{m:02d}' for a in range(2020, 2026) for m in range(1, 13)]
gaps_t2m  = sorted([e for e in esperados if e not in t2m])
gaps_d2m  = sorted([e for e in esperados if e not in d2m])
gaps_rad  = sorted([e for e in esperados if e not in rad])

# periodo comum usavel: precisa de t2m (base do residuo) E estacao
comum = sorted([m for m in esperados if m in t2m and m in d2m and m in rad and m in stn])

cobertura = {
    'era5_t2m': {'min': min(t2m), 'max': max(t2m), 'n': len(t2m), 'gaps': gaps_t2m},
    'era5_d2m': {'min': min(d2m), 'max': max(d2m), 'n': len(d2m), 'gaps': gaps_d2m},
    'era5_rad': {'min': min(rad), 'max': max(rad), 'n': len(rad), 'gaps': gaps_rad},
    'estacoes': {'min': min(stn), 'max': max(stn), 'n_meses': len(stn)},
    'periodo_comum_usavel': {'n_meses': len(comum),
                             'min': comum[0], 'max': comum[-1],
                             'limitante': 'gap de t2m (base do residuo)'},
}
with open(PASTA_GRID / 'periodo_cobertura.json', 'w', encoding='utf-8') as f:
    json.dump(cobertura, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# RELATORIO
# ============================================================

print('Grade de referencia (cop1km) fixada em:', PASTA_GRID)
print('  shape (lin,col):', meta_grade['shape'])
print('  z_alvo  min/max:', round(float(da_z_alvo.min()), 2), '/', round(float(da_z_alvo.max()), 2))
print('  z_std   min/max:', round(float(da_z_std.min()), 2), '/', round(float(da_z_std.max()), 2))
print('  periodo comum usavel:', cobertura['periodo_comum_usavel']['min'],
      '..', cobertura['periodo_comum_usavel']['max'],
      '(', cobertura['periodo_comum_usavel']['n_meses'], 'meses )')
print('  gaps t2m:', gaps_t2m)
