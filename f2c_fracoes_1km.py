#!/usr/bin/env python3

# ============================================================
# F2-C (fracoes) — fracoes termicas 30 m (MapBiomas) -> grade cop1km de 1 km,
# EM UM UNICO RESAMPLE (sem o duplo-passo do antigo f2c_fracoes_alinha).
#
# Fonte: brazil_coverage_<ano>.tif (cobre todo o dominio canonico; o recorte
# sc_coverage so cobria 192/206 celulas -> 14 celulas de borda davam NaN).
# Destino: grade EXATA do cop1km.tif (mesma transform/CRS/shape 560x807), logo
# todo pixel de saida cai num ponto da grade cop1km; saida mascarada as celulas
# VALIDAS do cop1km (terra).
#
# Denominador NaN-aware / "nao classificado"-aware (== logica original):
#   cada grupo T1..T8 vira mascara 0/1 a 30 m -> reamostra por MEDIA p/ 1 km
#   (= fracao do grupo na area TOTAL da celula); o denominador da normalizacao
#   e a SOMA das fracoes dos grupos (= fracao_observada). Codigos fora do de_para
#   (27 "Nao Observado", 0/nodata, etc.) nao entram em grupo nenhum -> ficam fora
#   do denominador. Ex.: [3,4,5,nan,99] -> conta so {3,4,5}.
#   T1..T8 somam 1 onde ha observacao; celula 100% nao observada vira nodata.
#
# --ano <YYYY> gera a saida por ano (fracoes_1km_alinhado_<ano>.tif) a partir
# de brazil_coverage_<ano>.tif; sem --ano mantem o comportamento original
# (2023, saida sem sufixo) usado por f4_motor_extracao.py e f8_infere_grade.py.
#
# So leitura da fonte (janela do dominio) + escrita do raster de fracoes.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, Resampling, transform_bounds
from _root import ROOT, DIR_GRID, DIR_ESTATICAS


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ        = Path(ROOT)
DE_PARA     = RAIZ / 'Dados' / 'mapbiomas' / 'de_para_mapbiomas_termico.csv'
COP1KM      = DIR_GRID / 'cop1km.tif'

NODATA   = -9999.0
MARGEM_DEG = 0.03            # folga ao recortar a janela da fonte (cobre a borda)

parser = argparse.ArgumentParser()
parser.add_argument('--ano', type=int, default=2023,
                    help='ano do MapBiomas (brazil_coverage_<ano>.tif); default 2023')
args = parser.parse_args()

SRC_30M = RAIZ / 'Dados' / 'mapbiomas' / f'brazil_coverage_{args.ano}.tif'
SAIDA   = (DIR_ESTATICAS / 'fracoes_1km_alinhado.tif' if args.ano == 2023
          else DIR_ESTATICAS / f'fracoes_1km_alinhado_{args.ano}.tif')


# ============================================================
# DE-PARA (MapBiomas code -> grupo termico T1..T8)
# ============================================================

def carregar_de_para(csv):
    dp = pd.read_csv(csv, sep=';')
    dp = dp[dp['grupo'].str.startswith('T', na=False)]      # descarta 'NA' (cod 27)
    de_para = {g: sub['code'].astype(int).tolist() for g, sub in dp.groupby('grupo')}
    return de_para, sorted(de_para.keys())


# ============================================================
# EXECUCAO
# ============================================================

t0 = time.time()
print('Lendo de-para e grade destino (cop1km)...')
de_para, grupos = carregar_de_para(DE_PARA)
cods_mapeados = {c for cs in de_para.values() for c in cs}
print('  grupos:', ', '.join(grupos))

with rasterio.open(COP1KM) as ref:
    dst_transform = ref.transform
    dst_crs       = ref.crs
    dst_h, dst_w  = ref.height, ref.width
    ref_bounds    = ref.bounds
    cop           = ref.read(1)
    cop_nodata    = ref.nodata
valid_cop = np.isfinite(cop) & (cop != cop_nodata)
print(f'  grade destino: {dst_w} x {dst_h}  | celulas validas (terra): {int(valid_cop.sum())}')

# ---- janela da fonte 30 m que cobre o dominio do cop1km (+ margem) ----
print('Recortando janela do dominio em', SRC_30M.name, '...')
with rasterio.open(SRC_30M) as src:
    oeste, sul, leste, norte = transform_bounds(dst_crs, src.crs, *ref_bounds, densify_pts=21)
    win = from_bounds(oeste - MARGEM_DEG, sul - MARGEM_DEG,
                      leste + MARGEM_DEG, norte + MARGEM_DEG, transform=src.transform)
    win = win.intersection(Window(0, 0, src.width, src.height))
    win = Window(int(np.floor(win.col_off)), int(np.floor(win.row_off)),
                 int(np.ceil(win.width)) + 1, int(np.ceil(win.height)) + 1
                 ).intersection(Window(0, 0, src.width, src.height))
    classes       = src.read(1, window=win)
    src_transform = src.window_transform(win)
    src_crs       = src.crs
print(f'  janela: {classes.shape[1]} x {classes.shape[0]} px 30 m')

# auditoria: codigos presentes excluidos da contabilidade
presentes = set(int(c) for c in np.unique(classes))
print('  codigos excluidos (fora do de-para):', sorted(presentes - cods_mapeados))

# ---- fracao de cada grupo: mascara 0/1 (30 m) -> media -> 1 km (cop1km) ----
print('Reamostrando mascaras por grupo (average) direto na grade cop1km...')
frac_full = {}
for g in grupos:
    mascara = np.isin(classes, de_para[g]).astype('float32')
    dst = np.zeros((dst_h, dst_w), dtype='float32')
    reproject(source=mascara, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.average)
    frac_full[g] = dst

# ---- fracao observada (denominador) e normalizacao ----
print('Fracao observada (= denominador, exclui nao-classificados) e normalizando...')
observada = np.zeros((dst_h, dst_w), dtype='float32')
for g in grupos:
    observada += frac_full[g]

tem_obs = observada > 0
frac_norm = {}
with np.errstate(divide='ignore', invalid='ignore'):
    for g in grupos:
        fr = np.where(tem_obs & valid_cop, frac_full[g] / observada, NODATA)
        frac_norm[g] = fr.astype('float32')
observada_out = np.where(valid_cop, observada, NODATA).astype('float32')

# ---- checagem fisica: T1..T8 em [0,1] e soma ~1 onde ha observacao ----
soma = np.zeros((dst_h, dst_w), dtype='float64')
for g in grupos:
    fr = frac_norm[g]
    fin = fr[fr != NODATA]
    if fin.size and (fin.min() < -1e-3 or fin.max() > 1 + 1e-3):
        raise ValueError(f'{g} fora de [0,1]: {fin.min():.4f}/{fin.max():.4f}')
    soma += np.where(fr != NODATA, fr, 0.0)
m = tem_obs & valid_cop
err_soma = float(np.max(np.abs(soma[m] - 1.0))) if m.any() else 0.0
print(f'  T1..T8 em [0,1] OK | max|soma-1| (celulas observadas) = {err_soma:.2e}')

# ============================================================
# SAIDA — 9 bandas (T1..T8 + fracao_observada), alinhada ao cop1km
# ============================================================

perfil = {'driver': 'GTiff', 'height': dst_h, 'width': dst_w, 'count': len(grupos) + 1,
          'dtype': 'float32', 'crs': dst_crs, 'transform': dst_transform,
          'nodata': NODATA, 'compress': 'deflate', 'predictor': 3, 'tiled': True}
print('Salvando', SAIDA, '...')
with rasterio.open(SAIDA, 'w', **perfil) as dst:
    for i, g in enumerate(grupos, start=1):
        dst.write(frac_norm[g], i)
        dst.set_band_description(i, g)
    dst.write(observada_out, len(grupos) + 1)
    dst.set_band_description(len(grupos) + 1, 'fracao_observada')

n_validas = int((tem_obs & valid_cop).sum())
print(f'OK: {SAIDA}')
print(f'  celulas com fracao valida: {n_validas} / {int(valid_cop.sum())} (terra)  | {time.time()-t0:.1f} s')
