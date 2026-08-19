#!/usr/bin/env python3

# ============================================================
# F2-D — Teste de schema das estaticas contra a grade canonica cop1km (gate-out)
# ============================================================


# ============================================================
# IMPORTS / CONFIG
# ============================================================

import json
from pathlib import Path

import numpy as np
import rioxarray
from _root import ROOT, DIR_GRID, DIR_ESTATICAS

PASTA_EST  = DIR_ESTATICAS
PASTA_GRID = DIR_GRID

# faixas fisicas por nome (camadas escalares)
FAIXAS = {
    'svf_mean_1km':       (0.0, 1.0),
    'svf_p10_1km':        (0.0, 1.0),
    'northness_mean_1km': (-1.0, 1.0),
    'eastness_mean_1km':  (-1.0, 1.0),
    'slope_mean_1km':     (0.0, 90.0),
    'slope_max_1km':      (0.0, 90.0),
    'mrvbf_p90_1km':      (0.0, 10.0),
    'hand_mean_1km':      (0.0, 3000.0),
    'hand_p10_1km':       (0.0, 3000.0),
    'z_std_1km':          (0.0, 1000.0),
    'dist_oceano_1km_alinhado': (0.0, 1.0e6),
}


# ============================================================
# PROCESSAMENTO
# ============================================================

meta = json.load(open(PASTA_GRID / 'grid_1km.json'))
shape_ref = tuple(meta['shape'])
print('Grade de referencia:', shape_ref, meta['crs'])
print('Validando schema de todas as estaticas...\n')

tifs = sorted(PASTA_EST.glob('*.tif'))
n_ok = 0
for tif in tifs:
    da = rioxarray.open_rasterio(tif, masked=True).squeeze()
    nb = da.shape[0] if da.ndim == 3 else 1

    # shape espacial == grade-ref
    if tuple(da.shape[-2:]) != shape_ref:
        raise ValueError(tif.name + ': shape ' + str(da.shape[-2:]) + ' != grade ' + str(shape_ref))
    # CRS == grade-ref
    if str(da.rio.crs) != meta['crs']:
        raise ValueError(tif.name + ': CRS ' + str(da.rio.crs) + ' != ' + meta['crs'])

    nome = tif.stem
    v = da.values[np.isfinite(da.values)]
    info = ''
    if nome in FAIXAS:
        lo, hi = FAIXAS[nome]
        if v.min() < lo - 1e-3 or v.max() > hi + 1e-3:
            raise ValueError(tif.name + ': fora da faixa [' + str(lo) + ',' + str(hi) + ']: [' +
                             str(v.min()) + ',' + str(v.max()) + ']')
        info = '  [faixa OK ' + str((lo, hi)) + ']'
    print('  OK %-30s bandas=%d  min/max=%.3f/%.3f%s' % (nome, nb, float(v.min()), float(v.max()), info))
    n_ok += 1

print('\nSchema OK: %d arquivos alinhados a %s, CRS %s.' % (n_ok, shape_ref, meta['crs']))


# ============================================================
# INVENTARIO DE FEATURES ESTATICAS (D1 = B.3 aneis)
# ============================================================

features = [
    'z_alvo (cop1km, F0)', 'z_std', 'slope_mean', 'slope_max',
    'northness_mean', 'eastness_mean', 'tri_mean', 'tri_std', 'tri_max',
    'svf_mean', 'svf_p10', 'mrvbf_p90', 'hand_mean', 'hand_p10',
    'dev_r300_mean', 'dev_r300_p10', 'dev_r1000_mean', 'dev_r1000_p10',
    'dev_r3000_mean', 'dev_r3000_p10',
    'fracoes T1..T8 (8)', 'dist_oceano', 'coord_x (F5)', 'coord_y (F5)',
]
print('\nInventario estatico (D1=B.3 aneis):', len(features) - 1 + 8 - 1, 'colunas estaticas aprox.')
for f in features:
    print('  -', f)
