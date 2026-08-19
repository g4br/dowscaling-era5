#!/usr/bin/env python3

# ============================================================
# F8b — Benchmark da regra de interpolacao da ANCORAGEM (leave-cell-out no OOF).
# Compara, SEM rodar o F8, as regras de peso usadas p/ espalhar o incremento
# OOF das celulas-estacao (inc = pred_residuo - y_residuo = T_final - obs):
#   - idw   : w = 1/d^p nos k vizinhos (regra atual do F8; k=20 p=1 = C0)
#   - idw+vert  : w = (1/d^p)*exp(-dz^2/Lz^2) nos k vizinhos (hibrido)
#   - gauss : w = exp(-d^2/L^2)                      (Barnes horizontal)
#   - gauss+vert: w = exp(-d^2/L^2)*exp(-dz^2/Lz^2)  (Barnes com peso vertical)
# Leave-cell-out: o incremento de CADA celula-estacao e previsto so pelas
# OUTRAS (diagonal da matriz de pesos zerada) — mesma normalizacao NaN-aware
# de aplicar_ancoragem (f8). Metricas identicas ao F6b:
#   err_anc  = inc - inc_hat (= T_final_ancorado - obs); err_base = -y_residuo;
#   SS = 1 - RMSE / RMSE_ERA5.
# Validacao: com --oof oof_ensaio.parquet e idw k=20 p=1 deve reproduzir o C0
#   (RMSE 1.465 -> 1.245, SS 0.2177 -> 0.3348; sessao-20260701.md).
# Estratos: faixa de temperatura observada (frio<=p10 / meio / quente>p90,
# obs via matriz_treino) e hora UTC — mostram ONDE o termo vertical ganha.
# --por-hora: alem do global, escolhe Lz*(h) por hora do dia (argmin do RMSE
#   por estrato, idw k=20 p=1) e grava lz_por_hora_<tag>.json p/ o F8 consumir
#   via --ancora-Lz-json. Validado em leave-one-year-out: escolha estavel
#   (identica em 21/24 h nos 6 folds), ganho held-out +0,0006 SS vs Lz global.
# Saida: tabelas no stdout + csv em saidas/modelos/bench_ancoragem/.
# Uso: python f8b_bench_ancoragem.py [--oof FILE] [--L 15,25,...] [--Lz 0,100,...] [--por-hora]
# Rodar no env com pyarrow/rioxarray/scipy (py312). Leve (~1-2 min).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
# VM 44 vCPU / 4 NUMA: limitar BLAS antes de importar numpy (ver memoria do projeto)
os.environ.setdefault('OMP_NUM_THREADS', '22')

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
from _root import ROOT, DIR_GRID, DIR_MATRIZ


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ        = Path(ROOT)
PASTA_GRID  = DIR_GRID
PASTA_MODEL = RAIZ / 'saidas' / 'modelos'
MATRIZ      = DIR_MATRIZ / 'matriz_treino.parquet'

ARQ_COP     = PASTA_GRID / 'cop1km.tif'          # altitude 1 km (grade canonica)
ARQ_CENTROS = PASTA_GRID / 'centros_wgs84.npz'   # X_grid/Y_grid em EPSG:31982

NY, NX = 560, 807                                # grade canonica (f0)


# ============================================================
# FUNCOES — metricas (identicas ao F6b)
# ============================================================

def metricas(err):
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'me': np.nan, 'n': 0}
    return {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'me': float(np.mean(err)), 'n': int(err.size)}


def skill(err_down, err_base):
    md, mb = metricas(err_down), metricas(err_base)
    ss = 1.0 - md['rmse'] / mb['rmse'] if mb['rmse'] and mb['rmse'] > 0 else float('nan')
    return md, mb, ss


# ============================================================
# FUNCOES — pesos leave-cell-out (estacao x estacao)
# ============================================================

def ler_raster(caminho, ny, nx):
    arr = rioxarray.open_rasterio(caminho, masked=True).squeeze()
    if arr.shape != (ny, nx):
        raise ValueError(f'{caminho}: shape {arr.shape} != ({ny},{nx})')
    return np.asarray(arr.values, dtype='float64').ravel()


def pesos_idw(d_km, k, power, dz_m=None, Lz_m=0.0):
    # regra atual do F8: 1/d^p nos k vizinhos mais proximos (excluindo a propria).
    # Com Lz>0, modula verticalmente: w = (1/d^p)*exp(-dz^2/Lz^2) (hibrido).
    k = int(k)                       # pode chegar como float (coluna com NaN)
    nest = d_km.shape[0]
    w = np.zeros_like(d_km)
    for s in range(nest):
        d = d_km[s].copy()
        d[s] = np.inf
        viz = np.argsort(d)[:min(k, nest - 1)]
        w[s, viz] = 1.0 / np.maximum(d[viz], 1e-6) ** power
    if Lz_m and Lz_m > 0:
        w = w * np.exp(-(dz_m / Lz_m) ** 2)
    return w


def pesos_gauss(d_km, dz_m, L_km, Lz_m):
    # w = exp(-d^2/L^2) * exp(-dz^2/Lz^2); Lz=0 -> sem termo vertical.
    # Sem corte em k: a propria gaussiana localiza (peso ~0 alem de ~2-3 L).
    w = np.exp(-(d_km / L_km) ** 2)
    if Lz_m and Lz_m > 0:
        w = w * np.exp(-(dz_m / Lz_m) ** 2)
    np.fill_diagonal(w, 0.0)
    return w


def loo_inc_hat(inc_mat, w):
    # inc_mat (nt, nest) com NaN onde a estacao falta; w (nest, nest) diag=0.
    # inc_hat(t,s) = sum_j w[s,j]*inc(t,j) / sum_j w[s,j], só sobre j com obs
    # (mesma normalizacao NaN-aware de aplicar_ancoragem no f8).
    fin = np.isfinite(inc_mat)
    num = np.where(fin, inc_mat, 0.0) @ w.T
    den = fin.astype('float64') @ w.T
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Benchmark leave-cell-out da ancoragem (F8b).')
    ap.add_argument('--oof', default=str(PASTA_MODEL / 'oof_residuo_loco_55feat.parquet'),
                    help='parquet OOF (cell_id,data_hora_utc,y_residuo,pred_residuo). '
                         'Com oof_ensaio.parquet, idw k=20 p=1 reproduz o C0 (RMSE 1.245).')
    ap.add_argument('--matriz', default=str(MATRIZ),
                    help='matriz de treino (obs p/ estrato por faixa de temperatura)')
    ap.add_argument('--k', default='10,20,40', help='lista de k do idw (20 = C0)')
    ap.add_argument('--power', default='1,2', help='lista de expoentes do idw (1 = C0)')
    ap.add_argument('--L',  default='15,25,35,50,75,100',
                    help='lista de L horizontais (km) da gaussiana')
    ap.add_argument('--Lz', default='0,100,200,300,500,800',
                    help='lista de Lz verticais (m); 0 = sem termo vertical')
    ap.add_argument('--por-hora', action='store_true',
                    help='escolhe Lz*(h) por hora do dia (idw k=20 p=1) e grava json p/ o F8 '
                         '(--ancora-Lz-json). Ganho held-out ~+0,0006 SS vs Lz global.')
    ap.add_argument('--saida', default=str(PASTA_MODEL / 'bench_ancoragem'),
                    help='diretorio das tabelas csv')
    args = ap.parse_args()
    t0 = time.time()

    Ks  = [int(v) for v in args.k.split(',')]
    Ps  = [float(v) for v in args.power.split(',')]
    Ls  = [float(v) for v in args.L.split(',')]
    Lzs = [float(v) for v in args.Lz.split(',')]
    oof_p = Path(args.oof)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    tag = oof_p.stem.replace('oof_', '')

    # --- OOF -> linhas validas + matriz de incremento (nt, nest) ---
    print(f'Lendo OOF: {oof_p}', flush=True)
    df = pd.read_parquet(oof_p, columns=['cell_id', 'data_hora_utc', 'y_residuo', 'pred_residuo'])
    n0 = len(df)
    df = df[np.isfinite(df['y_residuo']) & np.isfinite(df['pred_residuo'])].reset_index(drop=True)
    df['inc'] = df['pred_residuo'].astype('float64') - df['y_residuo'].astype('float64')
    print(f'  {n0:,} linhas | {n0 - len(df):,} descartadas (NaN) | {len(df):,} usadas', flush=True)

    piv = df.pivot_table(index='data_hora_utc', columns='cell_id', values='inc')
    tempos, cells = piv.index, list(piv.columns)
    inc_mat = piv.values.astype('float64')                  # (nt, nest)
    nest = len(cells)
    print(f'  {nest} celulas-estacao | {len(tempos):,} horas | '
          f'{tempos.min()} -> {tempos.max()}', flush=True)

    # indexadores linha -> (tempo, estacao) p/ ler inc_hat de volta nas linhas
    it = tempos.get_indexer(pd.DatetimeIndex(df['data_hora_utc']))
    js = pd.Index(cells).get_indexer(df['cell_id'])
    if (it < 0).any() or (js < 0).any():
        raise RuntimeError('indexacao linha->pivot falhou')

    # --- coords (EPSG:31982) + altitude (cop1km) das celulas-estacao ---
    iyix = np.array([list(map(int, c.split('_'))) for c in cells])
    flat = iyix[:, 0] * NX + iyix[:, 1]
    g = np.load(ARQ_CENTROS)
    xg, yg = g['X_grid'].ravel(), g['Y_grid'].ravel()
    z_full = ler_raster(ARQ_COP, NY, NX)
    xe, ye, ze = xg[flat], yg[flat], z_full[flat]
    if not np.all(np.isfinite(ze)):
        raise ValueError('altitude NaN em celula-estacao (cop1km)')
    d_km = np.sqrt((xe[:, None] - xe[None, :]) ** 2 + (ye[:, None] - ye[None, :]) ** 2) / 1000.0
    dz_m = np.abs(ze[:, None] - ze[None, :])
    print(f'  altitude estacoes: {ze.min():.0f} a {ze.max():.0f} m | '
          f'dist media entre pares: {d_km[np.triu_indices(nest, 1)].mean():.0f} km', flush=True)

    # --- erros de referencia (F6b): baseline ERA5 e downscaling sem ancoragem ---
    err_base = (-df['y_residuo']).values.astype('float64')  # T_ERA5  - obs
    inc_rows = df['inc'].values                             # T_final - obs (sem ancora)

    # --- estratos: faixa de temperatura observada (obs da matriz) + hora UTC ---
    df['hora'] = pd.DatetimeIndex(df['data_hora_utc']).hour
    df['faixa'] = 'sem_obs'
    try:
        m = pd.read_parquet(args.matriz, columns=['cell_id', 'data_hora_utc', 'temperatura_c_qc'])
        df = df.merge(m, on=['cell_id', 'data_hora_utc'], how='left')
        obs = df['temperatura_c_qc'].values
        p10, p90 = np.nanpercentile(obs, [10, 90])
        df['faixa'] = np.select([obs <= p10, obs > p90], ['frio<=p10', 'quente>p90'], 'meio')
        print(f'  faixas por obs: p10={p10:.1f} degC, p90={p90:.1f} degC '
              f'({np.isnan(obs).sum():,} linhas sem obs)', flush=True)
    except Exception as e:
        print(f'  AVISO: sem estrato por faixa de temperatura ({e})', flush=True)

    # --- grade de regras: idw (k x p x Lz, Lz=0 = atual) + gauss (L x Lz) ---
    regras = [('idw', {'k': k, 'power': p, 'Lz': Lz}) for k in Ks for p in Ps for Lz in Lzs]
    regras += [('gauss', {'L': L, 'Lz': Lz}) for L in Ls for Lz in Lzs]
    print(f'\nAvaliando {len(regras)} regras (leave-cell-out, {len(tempos):,} h x {nest} est)...',
          flush=True)

    def err_da_regra(nome, prm):
        # erro por linha do OOF (T_final_ancorado - obs) da regra pedida
        if nome == 'idw':
            w = pesos_idw(d_km, prm['k'], prm['power'], dz_m, prm['Lz'])
        else:
            w = pesos_gauss(d_km, dz_m, prm['L'], prm['Lz'])
        inc_hat = loo_inc_hat(inc_mat, w)
        return inc_rows - inc_hat[it, js]

    def rotulo(nome, prm):
        vert = f' Lz={prm["Lz"]:g}m' if prm['Lz'] else ''
        if nome == 'idw':
            return f'idw k={prm["k"]} p={prm["power"]:g}{vert}'
        return f'gauss L={prm["L"]:g}km{vert}'

    linhas = []
    for nome, prm in regras:
        rot = rotulo(nome, prm)
        md, mb, ss = skill(err_da_regra(nome, prm), err_base)
        linhas.append({'regra': nome, **prm, 'rotulo': rot, 'n': md['n'],
                       'rmse': round(md['rmse'], 4), 'mae': round(md['mae'], 4),
                       'me': round(md['me'], 4), 'SS': round(ss, 4)})
        print(f'  {rot:34s} RMSE={md["rmse"]:.4f}  SS={ss:.4f}', flush=True)

    # referencias: baseline ERA5 e downscaling puro (sem ancoragem)
    md0, mb0, ss0 = skill(inc_rows, err_base)
    tab = pd.DataFrame(linhas).sort_values('rmse').reset_index(drop=True)

    print('\n=== GLOBAL (leave-cell-out, vs baseline ERA5) ===', flush=True)
    print(f'  baseline ERA5   : RMSE={mb0["rmse"]:.4f}', flush=True)
    print(f'  sem ancoragem   : RMSE={md0["rmse"]:.4f}  SS={ss0:.4f}', flush=True)
    print(tab[['rotulo', 'rmse', 'mae', 'me', 'SS']].to_string(index=False), flush=True)

    # --- estratos: vencedor vs regra atual (idw k=20 p=1 sem vert = C0) ---
    ref = tab[(tab['regra'] == 'idw') & (tab['k'] == 20) & (tab['power'] == 1.0) & (tab['Lz'] == 0)]
    ref = ref.iloc[0] if len(ref) else tab[tab['regra'] == 'idw'].iloc[0]
    win = tab.iloc[0] if tab['rotulo'].iloc[0] != ref['rotulo'] else tab.iloc[1]
    rot_idw, rot_win = ref['rotulo'], win['rotulo']
    err_ref = err_da_regra(ref['regra'], {c: ref[c] for c in ('k', 'power', 'L', 'Lz') if pd.notna(ref.get(c))})
    err_win = err_da_regra(win['regra'], {c: win[c] for c in ('k', 'power', 'L', 'Lz') if pd.notna(win.get(c))})
    est_linhas = []
    for col in ['faixa', 'hora']:
        for grupo, idx in df.groupby(col, observed=True).groups.items():
            ii = df.index.get_indexer(idx)
            mi, _, ss_i = skill(err_ref[ii], err_base[ii])
            mw, _, ss_w = skill(err_win[ii], err_base[ii])
            est_linhas.append({'estrato': col, 'grupo': grupo, 'n': mw['n'],
                               'rmse_idw': round(mi['rmse'], 4), 'rmse_win': round(mw['rmse'], 4),
                               'SS_idw': round(ss_i, 4), 'SS_win': round(ss_w, 4),
                               'dSS': round(ss_w - ss_i, 4)})
    est = pd.DataFrame(est_linhas)
    print(f'\n=== ESTRATOS: [{rot_win}] vs [{rot_idw}] ===', flush=True)
    for col in ['faixa', 'hora']:
        print(f'\n--- por {col} ---', flush=True)
        print(est[est['estrato'] == col].drop(columns='estrato').to_string(index=False), flush=True)

    # --- salva ---
    tab.to_csv(saida / f'global_{tag}.csv', sep=';', index=False)
    est.to_csv(saida / f'estratos_{tag}.csv', sep=';', index=False)
    print(f'\nSalvo: {saida}/global_{tag}.csv | estratos_{tag}.csv', flush=True)

    # --- Lz por hora do dia (opcional): argmin do RMSE por estrato, idw k=20 p=1 ---
    if args.por_hora:
        lzs_h = sorted(set(Lzs) | {0.0})
        err_lz = {lz: err_da_regra('idw', {'k': 20, 'power': 1.0, 'Lz': lz}).astype('float32')
                  for lz in lzs_h}
        horas = df['hora'].values
        esc = {}
        print('\n=== Lz por hora (idw k=20 p=1, argmin RMSE) ===', flush=True)
        for h in range(24):
            m = horas == h
            esc[h] = min(lzs_h, key=lambda lz: float(np.mean(err_lz[lz][m].astype('float64') ** 2)))
            print(f'  hora {h:02d}: Lz*={esc[h]:g} m', flush=True)
        arq_json = saida / f'lz_por_hora_{tag}.json'
        arq_json.write_text(json.dumps(
            {'metodo': 'idw', 'k': 20, 'power': 1.0, 'fonte_oof': oof_p.name,
             'lz_por_hora': {str(h): esc[h] for h in range(24)}}, indent=1))
        print(f'Salvo: {arq_json}  (use no F8: --ancora-Lz-json {arq_json})', flush=True)

    melhor = tab.iloc[0]
    print(f'\nVENCEDOR: {melhor["rotulo"]}  (RMSE {melhor["rmse"]:.4f}, SS {melhor["SS"]:.4f}; '
          f'regra atual [{rot_idw}]: RMSE {ref["rmse"]:.4f}, SS {ref["SS"]:.4f})', flush=True)
    if melhor['regra'] == 'gauss':
        sug = f'--ancora-metodo gauss --ancora-L {melhor["L"]:g} --ancora-Lz {melhor["Lz"]:g}'
    else:
        sug = (f'--ancora-metodo idw --ancora-k {int(melhor["k"])} '
               f'--ancora-power {melhor["power"]:g} --ancora-Lz {melhor["Lz"]:g}')
    print(f'Comando F8 sugerido:\n  python f8_infere_grade.py --ano AAAA --mes MM --ancora {sug}',
          flush=True)
    print(f'Tempo total: {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main()
