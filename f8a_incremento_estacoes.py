#!/usr/bin/env python3

# ============================================================
# F8a — Incremento das ESTACOES p/ ancorar um periodo FORA do treino (hindcast).
# Produz o parquet [cell_id, data_hora_utc, y_residuo, pred_residuo] que o F8
# consome em --ancora-oof. O OOF leave-cell-out (oof_residuo_loco) so existe p/
# 2020-2025; p/ datas novas (ex.: 2026) o incremento honesto e out-of-time:
#   pred_residuo = modelo(features reconstruidas na celula-estacao)      [residuo, degC]
#   y_residuo    = T_obs - (t2m_1km - 273.15)          [obs - baseline ERA5, degC]
#   inc (no F8)  = pred_residuo - y_residuo = T_final_estacao - T_obs
# (mesma logica out-of-time do oper: previsao na estacao vs obs fresca — como o
# periodo NAO foi treinado, nao ha vazamento in-sample que subestime a correcao.)
# Features reconstruidas EXATAMENTE como no F8/treino (funcoes importadas do F8).
# Leve: so as ~200 celulas-estacao, sem chunk. ERA5 e 1 arquivo/mes; a janela
# PODE cruzar meses (carrega o span de meses e concatena no eixo de tempo).
# Uso: python f8a_incremento_estacoes.py --init 2026-01-01 --end 2026-01-10 \
#        --modelo /.../modelo_p0100_meta.json
#      -> saidas/modelos/inc_estacoes_20260101T00_20260110T00.parquet ; depois:
#      python f8_infere_grade.py --modelo .../modelo_p0100_meta.json \
#        --init 2026-01-01 --end 2026-01-10 --ancora \
#        --ancora-oof saidas/modelos/inc_estacoes_20260101T00_20260110T00.parquet --ancora-Lz 500
# Rodar no env py312 (rioxarray/lightgbm/scipy).
# ============================================================

import os
os.environ.setdefault('OMP_NUM_THREADS', '22')   # NUMA: fixar antes de numpy/lightgbm

import sys
import json
import time
import calendar
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray   # noqa: F401  (registra .rio)
import lightgbm as lgb

# reconstrucao de features IDENTICA ao F8 (fonte unica) — importa a maquinaria.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _root import ROOT
from f8_infere_grade import (
    abrir_era5, exigir, desacumular_fluxo, InterpoladorGrade, ler_raster,
    carregar_estaticas, codificar_tempo, derivar_dinamica,
    RAW_DEPS, RAW_SRC, DYN_FEATURES, TEMP_MEM,
    ERA5_RAW, ARQ_TERMICO, ARQ_VAR, ARQ_RAD, PASTA_GRID, PASTA_MODEL,
)

RAIZ      = Path(ROOT)
ARQ_OBS   = RAIZ / 'stn_data' / 'estacoes_qc.parquet'
ARQ_TREIN = RAIZ / 'stn_data' / 'estacoes_treinaveis.csv'
ARQ_ORO   = PASTA_GRID / 'z_era5_orog_1km.tif'


def shift_rows(a, k):
    # a[t-k] com NaN antes do inicio (t<k) — dt2m/d2t2m no eixo temporal (igual oper).
    out = np.full_like(a, np.nan)
    if 0 < k < a.shape[0]:
        out[k:] = a[:-k]
    elif k == 0:
        out[:] = a
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Incremento das estacoes p/ F8 --ancora (periodo fora do treino).')
    ap.add_argument('--init', required=True, help='data inicial YYYY-MM-DD (inclusive)')
    ap.add_argument('--end',  default=None, help='data final YYYY-MM-DD (inclusive; default = fim do mes)')
    ap.add_argument('--modelo', default=str(PASTA_MODEL / 'modelo_p0100_meta.json'),
                    help='meta json do modelo (usa o ramo completo=t2m p/ pred_residuo)')
    ap.add_argument('--obs',   default=str(ARQ_OBS), help='parquet de obs QC (temperatura_c_qc)')
    ap.add_argument('--saida', default=None, help='parquet de saida (default inc_estacoes_AAAA_MM.parquet)')
    ap.add_argument('--excluir', default=None,
                    help='codigos de estacao a EXCLUIR do OI (csv); obs delas nao viram ancora. '
                         'Mede influencia real do OI vizinho sobre a estacao segurada.')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 22)))
    args = ap.parse_args()
    t0 = time.time()
    excluir = {str(c).strip() for c in args.excluir.split(',')} if args.excluir else set()

    d_ini = pd.Timestamp(args.init)
    d_fim = pd.Timestamp(args.end) if args.end else (d_ini + pd.offsets.MonthEnd(0))
    if d_fim < d_ini:
        sys.exit('ERRO: --end deve ser >= --init.')
    meses = pd.period_range(d_ini.normalize(), d_fim.normalize(), freq='M')  # span (pode cruzar meses)
    ano, mes = d_ini.year, d_ini.month   # so p/ prints; nome de saida usa init/end

    # --- modelo: ramo completo (L2) -> pred_residuo ---
    modelo_p = Path(args.modelo)
    meta = json.loads(modelo_p.read_text())
    feats = meta['features']
    lapse_rate = float(meta.get('lapse_rate', 0.0065))
    if meta.get('tipo') in ('segmentado_quantil', 'segmentado_quantil_v2'):
        ramo = 'completo' if 'completo' in meta['ramos'] else list(meta['ramos'])[0]
        f_mod = modelo_p.parent / meta['ramos'][ramo]['modelo']
    else:                                   # booster monolitico (F7): meta ao lado do .txt
        f_mod = modelo_p.with_name(meta.get('modelo', modelo_p.stem.replace('_meta', '') + '.txt'))
    if not f_mod.exists():
        sys.exit(f'ERRO: booster nao encontrado: {f_mod}')
    booster = lgb.Booster(model_file=str(f_mod))
    print(f'Modelo: {f_mod.name} | {len(feats)} features', flush=True)

    # --- grade + celulas-estacao unicas (iy,ix) ---
    da_z = rioxarray.open_rasterio(PASTA_GRID / 'cop1km.tif', masked=True).squeeze()
    ny, nx = da_z.shape
    g = np.load(PASTA_GRID / 'centros_wgs84.npz')
    lat_flat = g['lat_centros'].ravel(); lon_flat = g['lon_centros'].ravel()
    est = pd.read_csv(ARQ_TREIN, sep=';')
    est['codigo_estacao'] = est['codigo_estacao'].astype(str)
    if excluir:
        faltando = excluir - set(est['codigo_estacao'])
        if faltando:
            print(f'AVISO: --excluir codigos ausentes em treinaveis (nao fazem nada): '
                  f'{sorted(faltando)}', flush=True)
    cel = est[['iy', 'ix']].astype(int).drop_duplicates().reset_index(drop=True)
    cel['flat']    = cel['iy'] * nx + cel['ix']
    cel['cell_id'] = cel['iy'].astype(str) + '_' + cel['ix'].astype(str)
    flats = cel['flat'].to_numpy()
    ncel = len(cel)
    lat_c = lat_flat[flats]; lon_c = lon_flat[flats]
    print(f'Grade {ny}x{nx} | {ncel} celulas-estacao unicas | mes {ano}-{mes:02d}', flush=True)

    # --- classes de features (igual F8) ---
    estat_feats = [f for f in feats if f not in DYN_FEATURES and f not in TEMP_MEM
                   and not f.startswith(('hora_', 'doy_')) and f not in ('coord_x', 'coord_y')]
    dyn_feats   = [f for f in feats if f in DYN_FEATURES]
    temp_feats  = [f for f in feats if f.startswith(('hora_', 'doy_'))]
    coord_feats = [f for f in feats if f in ('coord_x', 'coord_y')]
    mem_feats   = [f for f in feats if f in TEMP_MEM]

    estaticas, _ = carregar_estaticas(set(estat_feats) | {'delta_z'}, ny, nx)
    if 'delta_z' not in estaticas:
        z_alvo = ler_raster(PASTA_GRID / 'cop1km.tif', ny, nx)
        estaticas['delta_z'] = z_alvo - ler_raster(ARQ_ORO, ny, nx)
    estaticas = {k: v[flats] for k, v in estaticas.items()}
    delta_z = estaticas['delta_z']
    coords = {}
    if coord_feats:
        coords['coord_x'] = g['X_grid'].ravel()[flats]
        coords['coord_y'] = g['Y_grid'].ravel()[flats]
    col_estatica = {**estaticas, **coords}

    need_raw = {'t2m'}
    for f in dyn_feats:
        need_raw.update(RAW_DEPS[f])

    # --- ERA5 bruto por MES no span (janela pode cruzar meses) ---
    # 1 arquivo/mes: carrega cada mes, de-acumula fluxo POR MES (reset diario
    # intacto) e concatena no eixo de tempo. Memoria temporal (dt2m) fica continua.
    coarse_parts = {v: [] for v in need_raw}
    tempos_parts = []
    for per in meses:
        a_m, m_m = per.year, per.month
        c = {'term': ERA5_RAW / ARQ_TERMICO.format(ano=a_m, mes=m_m),
             'var':  ERA5_RAW / ARQ_VAR.format(ano=a_m, mes=m_m),
             'rad':  ERA5_RAW / ARQ_RAD.format(ano=a_m, mes=m_m)}
        for k, p in c.items():
            if not p.exists():
                sys.exit(f'ERRO: ERA5 bruto ausente ({k}): {p}')
        fonte = {g: abrir_era5(c[g]) for g in c}
        nt_m = fonte['term'].sizes['time']
        if nt_m != calendar.monthrange(a_m, m_m)[1] * 24:
            sys.exit(f'ERRO: ERA5 {a_m}-{m_m:02d}: {nt_m}h != mes completo.')
        tempos_parts.append(pd.date_range(f'{a_m}-{m_m:02d}-01', periods=nt_m, freq='h'))
        for v in sorted(need_raw):
            grp, deac = RAW_SRC[v]
            da = exigir(fonte[grp], v, c[grp])
            coarse_parts[v].append(desacumular_fluxo(da) if deac else da)
    if len(meses) > 1:
        coarse = {v: xr.concat(coarse_parts[v], dim='time') for v in need_raw}
        tempos = tempos_parts[0].append(tempos_parts[1:])
    else:
        coarse = {v: coarse_parts[v][0] for v in need_raw}
        tempos = tempos_parts[0]
    nt = len(tempos)

    interp = InterpoladorGrade(lat_c, lon_c, da_mascara=coarse['t2m'], oceano='nearest')
    raw = {v: interp.interp(coarse[v]) for v in need_raw}      # (nt, ncel)
    baseline_c = raw['t2m'] - 273.15                           # T_ERA5 na celula-estacao

    mem_vals = {}
    if mem_feats:
        t2m = raw['t2m']
        for dh in (1, 2, 3):
            if f'dt2m_{dh}h' in mem_feats or f'd2t2m_{dh}h' in mem_feats:
                p1 = shift_rows(t2m, dh); p2 = shift_rows(t2m, 2 * dh)
                mem_vals[f'dt2m_{dh}h']  = t2m - p1
                mem_vals[f'd2t2m_{dh}h'] = t2m - 2 * p1 + p2

    # --- monta X (nt*ncel, nfeat) na ordem C (t externo, celula interna) ---
    temp_all = codificar_tempo(tempos) if temp_feats else {}
    X = np.empty((nt * ncel, len(feats)), dtype='float32')
    for j, f in enumerate(feats):
        if f in mem_vals:
            col = mem_vals[f].astype('float32').ravel()
        elif f in DYN_FEATURES:
            col = derivar_dinamica(f, raw, delta_z, lapse_rate).astype('float32').ravel()
        elif f in col_estatica:
            col = np.broadcast_to(col_estatica[f].astype('float32'), (nt, ncel)).ravel()
        elif f in temp_all:
            col = np.repeat(temp_all[f].astype('float32'), ncel)
        else:
            sys.exit(f'ERRO: feature sem fonte definida: {f}')
        X[:, j] = col
    pred_res = booster.predict(X, num_threads=args.threads).reshape(nt, ncel)

    # --- DataFrame longo (celula x hora) + recorte da janela ---
    df = pd.DataFrame({'cell_id': np.tile(cel['cell_id'].to_numpy(), nt),
                       'data_hora_utc': np.repeat(tempos.values, ncel),
                       'baseline_c': baseline_c.ravel(),
                       'pred_residuo': pred_res.ravel()})
    fim_excl = d_fim.normalize() + pd.Timedelta(days=1)
    df = df[(df['data_hora_utc'] >= d_ini.normalize()) & (df['data_hora_utc'] < fim_excl)]

    # --- obs frescas -> agrega por celula+hora (media das estacoes da celula) ---
    obs = pd.read_parquet(args.obs, columns=['codigo_estacao', 'data_hora_utc', 'temperatura_c_qc'])
    obs['codigo_estacao'] = obs['codigo_estacao'].astype(str)
    if excluir:
        n0 = len(obs)
        obs = obs[~obs['codigo_estacao'].isin(excluir)]
        print(f'OI SEM {sorted(excluir)}: -{n0-len(obs):,} obs removidas '
              '(celula fica sem ancora se nao houver outra estacao nela)', flush=True)
    obs = obs.merge(est[['codigo_estacao', 'iy', 'ix']].astype({'iy': int, 'ix': int}),
                    on='codigo_estacao', how='inner')
    obs['cell_id'] = obs['iy'].astype(str) + '_' + obs['ix'].astype(str)
    obs = obs.dropna(subset=['temperatura_c_qc'])
    obs_cell = (obs.groupby(['cell_id', 'data_hora_utc'], as_index=False)['temperatura_c_qc']
                .mean())

    df = df.merge(obs_cell, on=['cell_id', 'data_hora_utc'], how='inner')
    df['y_residuo'] = df['temperatura_c_qc'].astype('float64') - df['baseline_c'].astype('float64')
    out = df[['cell_id', 'data_hora_utc', 'y_residuo', 'pred_residuo']].reset_index(drop=True)
    if out.empty:
        sys.exit('ERRO: nenhuma obs casou com celula-estacao na janela — nada a gravar.')

    tag = f'{d_ini:%Y%m%dT%H}_{d_fim:%Y%m%dT%H}'   # arquivo por data+hora do range
    if excluir:                                    # nao sobrescreve o inc operacional
        tag += '_excl' + '-'.join(sorted(excluir))
    saida = Path(args.saida) if args.saida else PASTA_MODEL / f'inc_estacoes_{tag}.parquet'
    saida.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(saida, index=False)

    inc = out['pred_residuo'].to_numpy() - out['y_residuo'].to_numpy()   # T_final - obs
    print(f'\nSalvo: {saida}', flush=True)
    print(f'  {len(out):,} linhas | {out["cell_id"].nunique()} celulas com obs | '
          f'{out["data_hora_utc"].nunique()} horas', flush=True)
    print(f'  janela {d_ini.date()} .. {d_fim.date()} | inc=T_final-obs: '
          f'media={inc.mean():+.3f} |med|={np.median(np.abs(inc)):.3f} degC | '
          f'tempo {time.time()-t0:.1f}s', flush=True)
    print(f'\nUse no F8:\n  --ancora --ancora-oof {saida} --ancora-Lz 500', flush=True)


if __name__ == '__main__':
    main()
