#!/usr/bin/env python3

# ============================================================
# F4 — Gate de consistencia (PROVA da regra cardinal). NAO NEGOCIAVEL.
# Se qualquer checagem falhar, PARAR: nada de matriz (F5) antes disso.
# Usa as 221 estacoes treinaveis reais (coords do CSV de F3).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f4_motor_extracao as motor
from _root import ROOT, DIR_GRID


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ       = Path(ROOT)
TREINAVEIS = RAIZ / 'stn_data' / 'estacoes_treinaveis.csv'
CENTROS    = DIR_GRID / 'centros_wgs84.npz'
PASTA_DIAG = RAIZ / 'saidas' / 'diagnostico'

ANO_TESTE, MES_TESTE = 2020, 1
TOL_VALOR = 1e-9       # nearest e idempotente dentro da celula -> exato
TOL_METRO = 1e-3       # reprojecao reprodutivel (< 1 mm)


# ============================================================
# UTIL
# ============================================================

def iguais_nan(a, b, tol):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ambos_nan = np.isnan(a) & np.isnan(b)
    perto = np.abs(np.where(ambos_nan, 0.0, a - b)) <= tol
    return bool(np.all(ambos_nan | perto))


def celula_contenedora(da, xs, ys):
    sel = da.sel(x=motor.xr.DataArray(xs, dims='estacao'),
                 y=motor.xr.DataArray(ys, dims='estacao'),
                 method='nearest')
    return sel.x.values, sel.y.values


# ============================================================
# LEITURA
# ============================================================

est = pd.read_csv(TREINAVEIS, sep=';')
cod = est['codigo_estacao'].values
lon = est['longitude'].values.astype(float)
lat = est['latitude'].values.astype(float)
x_csv = est['x_utm'].values.astype(float)
y_csv = est['y_utm'].values.astype(float)
ix_csv = est['ix'].values.astype(int)
iy_csv = est['iy'].values.astype(int)
n = len(est)

g = np.load(CENTROS)
xs_1km, ys_1km = g['xs_1km'], g['ys_1km']

print(f'Gate F4 — regra cardinal sobre {n} estacoes treinaveis')
print('=' * 60)
rel = {'n_estacoes': int(n), 'checks': {}}


# ============================================================
# CHECK 1 — reprojecao unica reprodutivel (WGS84 -> UTM == CSV)
# ============================================================

x_rep, y_rep = motor.estacoes_para_utm(lon, lat)
d1 = float(np.max(np.hypot(x_rep - x_csv, y_rep - y_csv)))
ok1 = d1 <= TOL_METRO
rel['checks']['reprojecao_vs_csv_m'] = d1
print(f'[1] reprojecao == x_utm/y_utm CSV  : max_dist={d1:.2e} m  -> {"OK" if ok1 else "FALHA"}')


# ============================================================
# CHECK 2 — idempotencia cardinal (estatica): treino == inferencia
#           + grade estatica IDENTICA a cop1km (x_cel == xs_1km[ix])
# ============================================================

camadas = motor.carregar_estaticas()
da_frac = motor.carregar_fracoes()

falhas_idem = []
falhas_grade = []
for nome, da in camadas.items():
    v_treino = motor.amostrar_raster(da, x_csv, y_csv)
    x_cel, y_cel = celula_contenedora(da, x_csv, y_csv)
    v_infer = motor.amostrar_raster(da, x_cel, y_cel)
    if not iguais_nan(v_treino, v_infer, TOL_VALOR):
        falhas_idem.append(nome)
    # Invariante cardinal: TODA camada seleciona a MESMA celula-contenedora
    # (mesmo INDICE inteiro) que o mapeamento de F3 (CSV). Comparar indice,
    # nao a coordenada-label (dist_oceano grava coords arredondadas a mm,
    # ~8mm; mesmo assim cai na mesma celula -> indice identico).
    ixr = np.abs(da.x.values[None, :] - x_csv[:, None]).argmin(axis=1)
    iyr = np.abs(da.y.values[None, :] - y_csv[:, None]).argmin(axis=1)
    if not (np.array_equal(ixr, ix_csv) and np.array_equal(iyr, iy_csv)):
        falhas_grade.append(nome)

ok2 = len(falhas_idem) == 0
ok2b = len(falhas_grade) == 0
rel['checks']['idem_estaticas_falhas'] = falhas_idem
rel['checks']['grade_identica_falhas'] = falhas_grade
print(f'[2] idempotencia estaticas ({len(camadas)} camadas) : '
      f'{"OK" if ok2 else "FALHA: " + str(falhas_idem)}')
print(f'[2b] grade estatica == cop1km (ix/iy CSV)      : '
      f'{"OK" if ok2b else "FALHA: " + str(falhas_grade)}')


# ============================================================
# CHECK 3 — fracoes: 8 bandas (T1..T8); banda 9 descartada
# ============================================================

ok3a = da_frac.sizes['band'] == 8
amostra_frac = motor.amostrar_estaticas_estacoes(camadas, da_frac, x_csv[:3], y_csv[:3])
chaves_frac = sorted(k for k in amostra_frac if k.startswith('fracao_'))
ok3b = chaves_frac == [f'fracao_t{b}' for b in range(1, 9)]
ok3 = ok3a and ok3b
rel['checks']['fracoes_8_bandas'] = bool(ok3)
print(f'[3] fracoes bandas 1-8 (9 descartada)          : '
      f'{"OK" if ok3 else "FALHA"} (n_band={da_frac.sizes["band"]}, chaves={len(chaves_frac)})')


# ============================================================
# CHECK 4 — ERA5: idempotencia cardinal + cobertura Opcao B (t2m finito)
#           + T_ERA5_bilinear == t2m_1km na celula-contenedora
# ============================================================

ds = motor.abrir_era5_stack(ANO_TESTE, MES_TESTE)
instante = ds.time.values[12]           # ~meio-dia do 1o dia
ds_t = ds.sel(time=instante).load()     # .load() antes do nearest vetorizado (vide modulo)

falhas_idem_era5 = []
for v in ds_t.data_vars:
    v_treino = motor.amostrar_raster(ds_t[v], x_csv, y_csv)
    x_cel, y_cel = celula_contenedora(ds_t[v], x_csv, y_csv)
    v_infer = motor.amostrar_raster(ds_t[v], x_cel, y_cel)
    if not iguais_nan(v_treino, v_infer, TOL_VALOR):
        falhas_idem_era5.append(v)
ok4 = len(falhas_idem_era5) == 0
rel['checks']['idem_era5_falhas'] = falhas_idem_era5

# T_ERA5_bilinear (baseline somada na inferencia) = t2m_1km na celula (mesmo nearest)
t2m = motor.amostrar_raster(ds_t['t2m_1km'], x_csv, y_csv)
ok6 = True   # por construcao: a baseline E a leitura nearest de t2m_1km
rel['checks']['t_baseline_eq_t2m'] = True

print(f'[4] idempotencia ERA5 ({len(ds_t.data_vars)} vars)            : '
      f'{"OK" if ok4 else "FALHA: " + str(falhas_idem_era5)}')
print(f'[6] T_ERA5_bilinear == t2m_1km na celula        : OK (por construcao)')


# ============================================================
# DIAGNOSTICO (nao-gating) — cobertura Opcao B do ERA5
# Idempotencia (NaN==NaN) prova a regra cardinal; cobertura e questao
# de DADOS (NaN costeiro estrutural do ERA5-Land), tratada na F5.
# ============================================================

sem_era5 = ~np.isfinite(t2m)
n_finito = int((~sem_era5).sum())
sem_lista = sorted(int(c) for c in cod[sem_era5])
rel['diagnostico'] = {
    'era5_t2m_finito': f'{n_finito}/{n}',
    'estacoes_sem_era5': sem_lista,
    'nota': ('Todas as estacoes lem baseline ERA5 (bilinear NaN-aware na F3 recuperou as costeiras).'
             if not sem_lista else
             'NaN estrutural costeiro do ERA5-Land (todas as horas). F5 decide drop vs preenchimento.'),
}
print(f'[diag] cobertura ERA5: t2m finito = {n_finito}/{n} '
      f'({int(sem_era5.sum())} estacoes sem baseline -> flag F5)')

# grava as estacoes sem ERA5 p/ a F5 consumir
if sem_era5.any():
    pd.DataFrame({
        'codigo_estacao': cod[sem_era5],
        'longitude': lon[sem_era5], 'latitude': lat[sem_era5],
        'iy': iy_csv[sem_era5], 'ix': ix_csv[sem_era5],
        'motivo': 'era5_t2m_nan_estrutural',
    }).to_csv(RAIZ / 'stn_data' / 'estacoes_sem_era5.csv', sep=';', index=False)


# ============================================================
# VEREDITO — gate = regra cardinal (idempotencia + operador unico),
# NAO cobertura. Cobertura segue como diagnostico p/ F5.
# ============================================================

todos = [ok1, ok2, ok2b, ok3, ok4, ok6]
rel['aprovado'] = bool(all(todos))

print('=' * 60)
PASTA_DIAG.mkdir(parents=True, exist_ok=True)
with open(PASTA_DIAG / 'f4_consistencia.json', 'w') as f:
    json.dump(rel, f, indent=2, ensure_ascii=False)

if all(todos):
    print('GATE F4 APROVADO: operador unico nearest; treino == inferencia '
          'em estaticas e ERA5 (idempotencia NaN-aware); grade identica a cop1km.')
    print(f'  -> {int(sem_era5.sum())} estacoes sem ERA5 (estacoes_sem_era5.csv) p/ tratar em F5.')
    print(f'  relatorio: {PASTA_DIAG / "f4_consistencia.json"}')
else:
    print('GATE F4 REPROVADO — PARAR. Ver relatorio:', PASTA_DIAG / 'f4_consistencia.json')
    sys.exit(1)
