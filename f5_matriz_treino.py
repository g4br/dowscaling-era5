#!/usr/bin/env python3

# ============================================================
# F5 — Matriz de treino (LightGBM, mesmo schema da inferencia F7).
# Cada linha = celula-contenedora UNICA, no instante UTC. Quando varias
# estacoes caem na MESMA celula (221 estacoes -> 206 celulas), as features
# sao IDENTICAS (regra cardinal: leem a mesma celula) e o ALVO e a MEDIA das
# observacoes das estacoes daquela celula naquele instante.
# Alvo: y_residuo = T_obs_qc(media) - T_ERA5_bilinear, com a baseline lida na
# celula (= t2m_1km - 273.15), a mesma somada de volta na inferencia.
# Extracao via motor de F4 (operador unico nearest / celula-contenedora):
#   estaticas 1x por celula (no CENTRO); ERA5 por mes (amostrar_era5_celulas),
#   em paralelo (N_WORKERS) — cada stack decompacta 14 grades 3D.
# PARALELISMO SEM SOBREPOSICAO: a chave do alvo (cell_id, instante) pertence a
# UM unico (ano,mes), logo cada worker emite um bloco DISJUNTO de linhas; os
# workers so LEEM os globais (fork) e cada um escreve em memoria propria.
# 57 features = 16 ERA5 + 31 estaticas + 4 temporais + 6 deltas t2m:
#   ERA5 inclui ssr/str individuais (alem de rad_liquida=ssr+str);
#   1a ordem dt2m_{1,2,3}h e 2a ordem d2t2m_{1,2,3}h.
# NaN em features (HAND parcial, fracoes de borda, bowen noturno) e
# preservado: LightGBM roteia NaN nativamente.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import sys
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f4_motor_extracao as motor
from _root import ROOT, DIR_GRID, DIR_MATRIZ


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ        = Path(ROOT)
PARQUET     = RAIZ / 'stn_data' / 'estacoes_qc.parquet'
TREINAVEIS  = RAIZ / 'stn_data' / 'estacoes_treinaveis.csv'
PASTA_GRID  = DIR_GRID
PASTA_ERA5  = RAIZ / 'saidas' / 'era5land_interp1km_celulas'
PASTA_SAIDA = DIR_MATRIZ

N_FEATURES_ESPERADO = 57   # +ssr +str (radiacao solar/termal individuais)
N_WORKERS = 24
# corte de outliers do alvo: |y_residuo| > LIM_RESIDUO (°C) e tratado como erro
# de observacao / mismatch grosseiro com a baseline ERA5 e removido do treino.
LIM_RESIDUO = 10.0
# chaves = colunas NAO-feature. cell_id = celula-contenedora (grupo da CV
# leave-cell-out em F6); n_est_cel = estacoes mediadas naquela celula/instante.
CHAVES = ['cell_id', 'data_hora_utc', 'temperatura_c_qc', 'n_est_cel',
          'T_ERA5_bilinear', 'y_residuo']

# globais SOMENTE-LEITURA herdados pelos workers no fork (celulas UNICAS).
# Definidos em main() ANTES de criar o Pool; nunca reatribuidos num worker.
IY_CEL = IX_CEL = CELL2POS = None


# ============================================================
# FUNCOES
# ============================================================

def codificar_tempo(serie):
    hora = serie.dt.hour
    doy  = serie.dt.dayofyear
    return pd.DataFrame({
        'hora_sin': np.sin(2 * np.pi * hora / 24),
        'hora_cos': np.cos(2 * np.pi * hora / 24),
        'doy_sin':  np.sin(2 * np.pi * doy / 365.25),
        'doy_cos':  np.cos(2 * np.pi * doy / 365.25),
    }, index=serie.index)


def meses_disponiveis():
    out = set()
    for f in glob.glob(str(PASTA_ERA5 / 'era5land_interp1km_celulas_*.nc')):
        a, m = Path(f).stem.split('_')[-2:]
        out.add((int(a), int(m)))
    return out


def processar_mes(task):
    # worker (1 mes): extrai ERA5 nas celulas-contenedoras UNICAS p/ as obs ja
    # AGREGADAS pela media por celula. So LE os globais IY_CEL/IX_CEL/CELL2POS;
    # abre seu proprio .nc e devolve um DataFrame independente (bloco disjunto).
    ano, mes, grp = task
    ds = motor.abrir_era5_stack(ano, mes)
    vars_ = list(ds.data_vars)
    times = pd.DatetimeIndex(ds['time'].values)
    era5  = motor.amostrar_era5_celulas(ds, IY_CEL, IX_CEL)   # var -> (nt,206) ou (206,)
    ds.close()

    t_idx = times.get_indexer(grp['data_hora_utc'].values)
    ok = t_idx >= 0
    g = grp[ok]; ti = t_idx[ok]
    ci = np.array([CELL2POS[c] for c in g['cell_id'].values], dtype=int)

    d = {'cell_id':          g['cell_id'].values,
         'data_hora_utc':    g['data_hora_utc'].values,
         'temperatura_c_qc': g['temperatura_c_qc'].values.astype('float32'),
         'n_est_cel':        g['n_est_cel'].values.astype('int16')}
    for v in vars_:
        a = era5[v]
        d[v] = (a[ti, ci] if a.ndim == 2 else a[ci]).astype('float32')

    # memoria temporal do ERA5 na MESMA celula: tendencia (1a ordem) e curvatura
    # (2a ordem) de t2m_1km em 1h, 2h e 3h.
    #   1a: dt2m_dh  = t2m(t) - t2m(t-dh)                  (variacao)
    #   2a: d2t2m_dh = t2m(t) - 2*t2m(t-dh) + t2m(t-2*dh)  (variacao da variacao)
    # Lookup explicito de (obs_time - k*dh) no eixo de tempo do mes (robusto a
    # buracos); no inicio do mes os lags caem no mes anterior, ausentes no stack
    # do worker => NaN (fracao minima; LightGBM roteia NaN). Reusa o t2m amostrado.
    t_now   = pd.DatetimeIndex(g['data_hora_utc'].values)
    a_t2m   = era5['t2m_1km']
    t2m_now = d['t2m_1km']

    def _t2m_lag(h):                          # t2m_1km na MESMA celula, h horas antes
        j   = times.get_indexer(t_now - pd.Timedelta(hours=h))
        okj = j >= 0
        out = np.full(len(g), np.nan)
        out[okj] = a_t2m[j[okj], ci[okj]]
        return out

    for dh in (1, 2, 3):
        p1 = _t2m_lag(dh)                      # t2m(t-dh)
        p2 = _t2m_lag(2 * dh)                  # t2m(t-2*dh)
        d[f'dt2m_{dh}h']  = (t2m_now - p1).astype('float32')              # 1a ordem
        d[f'd2t2m_{dh}h'] = (t2m_now - 2 * p1 + p2).astype('float32')     # 2a ordem

    return pd.DataFrame(d), int((~ok).sum()), (ano, mes)


# ============================================================
# CELULAS-CONTENEDORAS (221 estacoes -> 206 celulas) E ESTATICAS (1x/celula)
# ============================================================

def main():
    global IY_CEL, IX_CEL, CELL2POS

    print('Lendo estacoes treinaveis e colapsando em celulas-contenedoras...')
    tre = pd.read_csv(TREINAVEIS, sep=';', dtype={'codigo_estacao': str})
    tre['cell_id'] = tre['iy'].astype(str) + '_' + tre['ix'].astype(str)
    n_est = len(tre); n_cel = tre['cell_id'].nunique()
    print(f'  {n_est} estacoes -> {n_cel} celulas-contenedoras unicas '
          f'({n_est - n_cel} estacoes compartilham celula; alvo = media por celula)')

    # mapa estacao -> celula (p/ agregar as observacoes ao nivel da celula)
    est2cel = tre.set_index('codigo_estacao')['cell_id']

    # celulas unicas: indice canonico (iy,ix) e CENTRO da celula (xs_1km/ys_1km).
    # Estaticas amostradas no CENTRO == valor na estacao (idempotencia, gate F4):
    # ficam bem-definidas mesmo quando varias estacoes caem na mesma celula, e
    # batem com a inferencia (que tambem preve no centro da celula).
    grid = np.load(PASTA_GRID / 'centros_wgs84.npz')
    xs_1km, ys_1km = grid['xs_1km'], grid['ys_1km']
    cels = (tre.drop_duplicates('cell_id')
               .sort_values('cell_id')[['cell_id', 'iy', 'ix']]
               .reset_index(drop=True))
    IY_CEL   = cels['iy'].values.astype(int)
    IX_CEL   = cels['ix'].values.astype(int)
    CELL2POS = {c: i for i, c in enumerate(cels['cell_id'].values)}
    x_cel = xs_1km[IX_CEL].astype('float64')
    y_cel = ys_1km[IY_CEL].astype('float64')

    camadas    = motor.carregar_estaticas()
    da_fracoes = motor.carregar_fracoes()
    estaticas  = motor.amostrar_estaticas_estacoes(camadas, da_fracoes, x_cel, y_cel)
    z_alvo = motor.amostrar_raster(
        rioxarray.open_rasterio(PASTA_GRID / 'cop1km.tif', masked=True).squeeze(), x_cel, y_cel)

    df_est = pd.DataFrame(estaticas)
    df_est['z_alvo']  = z_alvo
    df_est['coord_x'] = x_cel               # centro da celula (consistente c/ inferencia)
    df_est['coord_y'] = y_cel
    df_est['cell_id'] = cels['cell_id'].values
    print(f'  estaticas: {df_est.shape[1] - 1} features x {n_cel} celulas')

    cobertura = {k: int((~np.isfinite(v)).sum()) for k, v in estaticas.items()
                 if np.issubdtype(np.asarray(v).dtype, np.floating) and (~np.isfinite(v)).any()}
    if cobertura:
        print(f'  AVISO NaN em estaticas (celulas/{n_cel}, roteado pelo LightGBM):',
              {k: cobertura[k] for k in sorted(cobertura)})

    # ========================================================
    # OBSERVACOES (QC) — agregadas por celula: MEDIA do alvo
    # ========================================================

    print('Lendo observacoes QC...')
    obs = pd.read_parquet(PARQUET, columns=['codigo_estacao', 'data_hora_utc', 'temperatura_c_qc'])
    obs['codigo_estacao'] = obs['codigo_estacao'].astype(str)
    obs = obs[obs['codigo_estacao'].isin(est2cel.index) & obs['temperatura_c_qc'].notna()].copy()
    obs['cell_id'] = obs['codigo_estacao'].map(est2cel)
    obs['data_hora_utc'] = pd.to_datetime(obs['data_hora_utc'])

    # MEDIA POR CELULA: estacoes na MESMA celula-contenedora leem features
    # identicas (regra cardinal); o alvo e a media das observacoes no mesmo
    # (celula, instante). n_est_cel = nº de estacoes que contribuiram.
    # Esta agregacao roda no processo-pai, ANTES do fork -> cada (celula,
    # instante) ja esta resolvido como UMA linha antes de ir p/ os workers.
    obs = (obs.groupby(['cell_id', 'data_hora_utc'], sort=False)
              .agg(temperatura_c_qc=('temperatura_c_qc', 'mean'),
                   n_est_cel=('codigo_estacao', 'nunique'))
              .reset_index())
    obs['ano'] = obs['data_hora_utc'].dt.year
    obs['mes'] = obs['data_hora_utc'].dt.month

    disp = meses_disponiveis()
    obs = obs[[k in disp for k in zip(obs['ano'], obs['mes'])]].copy()
    n_multi = int((obs['n_est_cel'] > 1).sum())
    print(f'  obs-celula com stack: {len(obs):,} | {obs["cell_id"].nunique()} celulas '
          f'| {n_multi:,} instantes-celula com media de >1 estacao '
          f'| {obs["data_hora_utc"].min()} -> {obs["data_hora_utc"].max()}')

    # ========================================================
    # ERA5 POR MES (paralelo, particao DISJUNTA) E MONTAGEM
    # ========================================================
    # Cada tarefa = um (ano,mes). Como data_hora_utc determina (ano,mes) de forma
    # unica, nenhuma chave (cell_id, instante) cai em duas tarefas: os workers
    # produzem blocos sem interseccao e o concat apenas os empilha. A garantia e
    # checada em runtime pelo assert de unicidade abaixo.
    tarefas = [(ano, mes, grp[['cell_id', 'data_hora_utc', 'temperatura_c_qc', 'n_est_cel']])
               for (ano, mes), grp in obs.groupby(['ano', 'mes'], sort=True)]
    print(f'Amostrando ERA5 1 km em {len(tarefas)} meses ({N_WORKERS} workers)...')

    linhas = []; n_drop = 0; feito = 0
    with mp.Pool(N_WORKERS) as pool:
        for df_mes, nd, (a, m) in pool.imap_unordered(processar_mes, tarefas):
            linhas.append(df_mes); n_drop += nd; feito += 1
            if df_mes.empty and nd > 0:   # mes inteiro sem casar no eixo de tempo do stack
                print(f'  AVISO: {a}-{m:02d} sem match no eixo de tempo — '
                      f'todas {nd:,} obs descartadas (stack com tempo corrompido?)', flush=True)
            if feito % 10 == 0 or feito == len(tarefas):
                print(f'  {feito}/{len(tarefas)} meses', flush=True)

    df = pd.concat(linhas, ignore_index=True)

    # GUARDA ANTI-SOBREPOSICAO: cada (celula, instante) tem de aparecer 1x. Se
    # falhar, a particao paralela teve overlap (nunca deveria, por construcao).
    dups = int(df.duplicated(['cell_id', 'data_hora_utc']).sum())
    if dups:
        raise ValueError(f'{dups} pares (cell_id, instante) duplicados apos o concat '
                         f'— particao paralela com sobreposicao (abortar).')
    print(f'  linhas: {len(df):,} | descartadas (instante fora do stack): {n_drop:,} '
          f'| (cell_id, instante) unicos: OK')

    # ========================================================
    # ALVO, TEMPORAIS, JUNCAO ESTATICAS
    # ========================================================

    print('Residuo, temporais e juncao das estaticas...')
    df['T_ERA5_bilinear'] = df['t2m_1km'] - 273.15
    df['y_residuo']       = df['temperatura_c_qc'] - df['T_ERA5_bilinear']
    df = pd.concat([df.reset_index(drop=True), codificar_tempo(df['data_hora_utc'])], axis=1)
    df = df.merge(df_est, on='cell_id', how='left')

    # ordem determinista (workers chegam fora de ordem): saida byte-reproduzivel
    df = df.sort_values(['cell_id', 'data_hora_utc']).reset_index(drop=True)

    # ========================================================
    # DIAGNOSTICO REPRESENTATIVIDADE (z_alvo vs altitude real, por celula)
    # ========================================================

    alt = (pd.read_parquet(PARQUET, columns=['codigo_estacao', 'altitude'])
           .assign(codigo_estacao=lambda d: d['codigo_estacao'].astype(str))
           .drop_duplicates('codigo_estacao'))
    alt['cell_id'] = alt['codigo_estacao'].map(est2cel)
    alt_cel = (alt.dropna(subset=['cell_id'])
                  .groupby('cell_id')
                  .agg(altitude=('altitude', 'mean'), n_est_cel=('codigo_estacao', 'nunique'))
                  .reset_index())
    diag = df_est[['cell_id', 'z_alvo']].merge(alt_cel, on='cell_id', how='left')
    diag['delta_representatividade_m'] = diag['z_alvo'] - diag['altitude']

    # ========================================================
    # CHECAGENS E SAIDA
    # ========================================================

    n_features = len([c for c in df.columns if c not in CHAVES])
    nao_finito = int((~np.isfinite(df['y_residuo'])).sum())
    print(f'Features: {n_features} (esperado {N_FEATURES_ESPERADO}) | y_residuo nao-finito: {nao_finito}')
    if n_features != N_FEATURES_ESPERADO:
        raise ValueError(f'Colunas de feature = {n_features}, esperado {N_FEATURES_ESPERADO} (revisar D1)')
    if nao_finito:
        df = df[np.isfinite(df['y_residuo'])].copy()

    # CORTE DE OUTLIERS DO ALVO: |y_residuo| > LIM_RESIDUO (°C) indica provavel
    # erro de observacao ou descasamento grosseiro com a baseline ERA5; essas
    # amostras contaminam o treino. Aplicado apos o corte de nao-finitos, com o
    # residuo ja bem-definido.
    n_outlier = int((df['y_residuo'].abs() > LIM_RESIDUO).sum())
    if n_outlier:
        frac = 100.0 * n_outlier / len(df)
        print(f'Outliers |y_residuo| > {LIM_RESIDUO:g} °C removidos: {n_outlier:,} ({frac:.3f}%)')
        df = df[df['y_residuo'].abs() <= LIM_RESIDUO].copy()

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PASTA_SAIDA / 'matriz_treino.parquet', index=False)
    diag.to_csv(PASTA_SAIDA / 'diagnostico_representatividade.csv',
                sep=';', index=False, encoding='utf-8', float_format='%.3f')

    # cobertura de features (NaN por coluna na matriz final)
    nan_col = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    with open(PASTA_SAIDA / 'cobertura_features.json', 'w') as f:
        json.dump({'n_linhas': len(df), 'n_features': n_features,
                   'celulas': int(df['cell_id'].nunique()),
                   'estacoes_originais': int(n_est),
                   'instantes_celula_multi_estacao': int((df['n_est_cel'] > 1).sum()),
                   'lim_residuo': LIM_RESIDUO,
                   'outliers_residuo_removidos': n_outlier,
                   'nan_por_coluna': nan_col,
                   'estaticas_nan_celulas': cobertura}, f, indent=2, ensure_ascii=False)

    print(f'Matriz: {df.shape} | features: {n_features} | celulas: {df["cell_id"].nunique()}')
    print(f'y_residuo: media={df["y_residuo"].mean():.3f} std={df["y_residuo"].std():.3f} '
          f'min={df["y_residuo"].min():.2f} max={df["y_residuo"].max():.2f}')
    print(f'Salvos em {PASTA_SAIDA}: matriz_treino.parquet, diagnostico_representatividade.csv, cobertura_features.json')


if __name__ == '__main__':
    main()
