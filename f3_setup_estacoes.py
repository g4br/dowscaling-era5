#!/usr/bin/env python3

# ============================================================
# F3-setup — classificar estacoes e mapear celula-contenedora (Opcao B)
# Treinavel = dentro da grade cop1km E celula de terra (z_alvo finito).
# .interp do ERA5 sera no CENTRO da celula-contenedora (regra cardinal),
# nunca na lon/lat exata da estacao.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
from pyproj import Transformer
from _root import ROOT, DIR_GRID


# ============================================================
# CONFIGURACOES
# ============================================================

# EST_PARQUET/SUF sobrepoem entrada e sufixo das saidas: rodar o MESMO mapeamento
# latlon->celula nas estacoes da aba Exclusao sem sobrescrever os arquivos do treino.
#   EST_PARQUET=.../estacoes_qc_exclusao.parquet SUF=_exclusao python f3_setup_estacoes.py
PARQUET   = os.environ.get('EST_PARQUET') or ROOT + '/stn_data/estacoes_qc.parquet'
SUF       = os.environ.get('SUF', '')
PASTA_GRID = DIR_GRID
PASTA_STN = Path(ROOT + '/stn_data')
RES = 1000.0


# ============================================================
# LEITURA
# ============================================================

est = pd.read_parquet(PARQUET)[['codigo_estacao', 'longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
g   = np.load(PASTA_GRID / 'centros_wgs84.npz')
xs, ys = g['xs_1km'], g['ys_1km']
lat_centros, lon_centros = g['lat_centros'], g['lon_centros']
cop = rioxarray.open_rasterio(PASTA_GRID / 'cop1km.tif', masked=True).squeeze().values


# ============================================================
# CLASSIFICACAO
# ============================================================

tr = Transformer.from_crs('EPSG:4326', 'EPSG:31982', always_xy=True)
ex, ey = tr.transform(est['longitude'].values, est['latitude'].values)

dentro = (ex >= xs.min() - RES / 2) & (ex <= xs.max() + RES / 2) & \
         (ey >= ys.min() - RES / 2) & (ey <= ys.max() + RES / 2)

ix = np.clip(np.abs(xs[None, :] - ex[:, None]).argmin(axis=1), 0, len(xs) - 1)
iy = np.clip(np.abs(ys[None, :] - ey[:, None]).argmin(axis=1), 0, len(ys) - 1)
terra = np.array([np.isfinite(cop[iy[i], ix[i]]) for i in range(len(est))])

treinavel = dentro & terra
motivo = np.where(~dentro, 'fora_extent', np.where(~terra, 'celula_oceano', 'ok'))

est['x_utm'] = ex; est['y_utm'] = ey
est['iy'] = iy; est['ix'] = ix
est['cell_lon'] = lon_centros[iy, ix]
est['cell_lat'] = lat_centros[iy, ix]
est['treinavel'] = treinavel
est['motivo'] = motivo


# ============================================================
# SAIDA
# ============================================================

est_ok = est[est['treinavel']].copy()
est_no = est[~est['treinavel']].copy()

est_ok.to_csv(PASTA_STN / f'estacoes_treinaveis{SUF}.csv', sep=';', index=False, encoding='utf-8')
est_no[['codigo_estacao', 'longitude', 'latitude', 'motivo']].to_csv(
    PASTA_STN / f'estacoes_excluidas_extracao{SUF}.csv', sep=';', index=False, encoding='utf-8')

# celulas unicas a popular nas grades F3 (centro p/ .interp)
cels = est_ok[['iy', 'ix', 'cell_lon', 'cell_lat']].drop_duplicates().reset_index(drop=True)
np.savez(PASTA_GRID / f'celulas_estacoes{SUF}.npz',
         iy=cels['iy'].values, ix=cels['ix'].values,
         cell_lon=cels['cell_lon'].values, cell_lat=cels['cell_lat'].values)

print('estacoes treinaveis:', len(est_ok), '| excluidas:', len(est_no))
print('  fora_extent:', int((motivo == 'fora_extent').sum()),
      '| celula_oceano:', int((motivo == 'celula_oceano').sum()))
print('celulas unicas a popular:', len(cels))
print(f'salvos: estacoes_treinaveis{SUF}.csv, estacoes_excluidas_extracao{SUF}.csv, '
      f'celulas_estacoes{SUF}.npz')
