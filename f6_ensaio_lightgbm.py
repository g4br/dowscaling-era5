#!/usr/bin/env python3

# ============================================================
# F6 (ensaio) — Downscaling LightGBM do RESIDUO sobre a baseline ERA5.
# Alvo: y_residuo = T_obs - T_ERA5_bilinear. Inferencia: T_final = T_ERA5 + f(X).
# Validacao: leave-cell-out (GroupKFold por cell_id) — o produto preve onde NAO
# ha estacao; CV aleatoria vazaria. Agrupar por CELULA (e nao por estacao) evita
# o vazamento de estacoes que partilham a MESMA celula-contenedora (X identico).
# Early stopping num conjunto de validacao tambem agrupado por celula (carregado
# do treino, sem tocar o teste).
# Metricas SEMPRE vs baseline ERA5 (SS = 1 - RMSE_down/RMSE_ERA5).
# NaN nas features e nativo do LightGBM (nao imputar).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from _root import ROOT, DIR_MATRIZ


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ    = Path(ROOT)
MATRIZ  = DIR_MATRIZ / 'matriz_treino.parquet'
SAIDA   = RAIZ / 'saidas' / 'modelos'

CHAVES = ['cell_id', 'data_hora_utc', 'temperatura_c_qc', 'n_est_cel', 'T_ERA5_bilinear', 'y_residuo']

# Conjunto base decidido em selecao_variaveis.md (reducao por multicolinearidade
# + remocoes manuais n_est_cel, coord_x). Mantidos por criterio fisico apesar da
# colinearidade: sp (~z_alvo) e dt_ar_pele (~rad_liquida). Acrescida a memoria
# temporal do ERA5 (t2m_1km + dt2m_{1,2,3}h de 1a ordem e d2t2m_{1,2,3}h de 2a
# ordem, geradas em F5) -> 53 features no total.
FEATURES_53 = [
    'lapse', 'u10', 'v10', 'vel_vento', 'dep_orvalho', 'dt_ar_solo', 'rad_liquida',
    'dt_ar_pele', 'razao_bowen', 'delta_z',
    'dev_r1000_p10_1km', 'dev_r3000_mean_1km', 'dev_r300_p10_1km', 'eastness_mean_1km',
    'hand_mean_1km', 'mrvbf_p90_1km', 'northness_mean_1km', 'tri_mean_1km', 'fracao_t1',
    'fracao_t3', 'fracao_t4', 'fracao_t5', 'fracao_t6', 'fracao_t7', 'fracao_t8',
    'z_alvo', 'sp', 'dist_oceano',
    't2m_1km',                            # T absoluta (regime), gerada em F5
    'dt2m_1h', 'dt2m_2h', 'dt2m_3h',      # memoria temporal ERA5 — 1a ordem / tendencia (1/2/3h)
    'd2t2m_1h', 'd2t2m_2h', 'd2t2m_3h',   # memoria temporal ERA5 — 2a ordem / curvatura (1/2/3h)
    'dev_r1000_mean_1km', 'dev_r3000_p10_1km',
    'dev_r300_mean_1km','hand_p10_1km','slope_max_1km',
    'slope_mean_1km', 'svf_mean_1km', 'svf_p10_1km','tri_max_1km',
    'tri_std_1km', 'umidade_relativa', 'z_std_1km',
    'hora_sin', 'hora_cos', 'doy_sin', 'doy_cos',
    'coord_x', 'coord_y'
]

# conjunto podado (33) de selecao_variaveis.md (poda por multicolinearidade + remocoes manuais).
FEATURES_33 = [
    'lapse', 'u10', 'v10', 'vel_vento', 'dep_orvalho', 'dt_ar_solo', 'rad_liquida',
    'dt_ar_pele', 'razao_bowen', 'delta_z', 'hora_sin', 'hora_cos', 'doy_sin', 'doy_cos',
    'dev_r1000_p10_1km', 'dev_r3000_mean_1km', 'dev_r300_p10_1km', 'eastness_mean_1km',
    'hand_mean_1km', 'mrvbf_p90_1km', 'northness_mean_1km', 'tri_mean_1km', 'fracao_t1',
    'fracao_t3', 'fracao_t4', 'fracao_t5', 'fracao_t6', 'fracao_t7', 'fracao_t8', 'z_alvo',
    'sp', 'dist_oceano', 'coord_y'
]
# 53 + radiacao solar (onda-curta) e termal (onda-longa) individuais.
FEATURES_55 = FEATURES_53 + ['ssr', 'str']
CONJUNTOS = {'33': FEATURES_33, '53': FEATURES_53, '55': FEATURES_55}

PARAMS = {
    'objective':          'regression_l2',
    'metric':             ['l2', 'l1'],
    'learning_rate':      0.02,   # passo menor generaliza melhor (early stopping corta)
    'num_leaves':         63,     # menos variancia p/ extrapolacao espacial (leave-cell-out)
    'min_child_samples':  500,    # folhas grandes -> robusto ao ruido do residuo
    'feature_fraction':   0.8,    # pode amostrar mais por split
    'bagging_fraction':   0.7,
    'bagging_freq':       5,      # era 1: re-bagging a cada round so encarece; 5 basta
    'lambda_l1':          1.0,
    'lambda_l2':          5.0,    # mais shrinkage L2 contra ruido
    'max_bin':            127,    # era 255 (default): histograma mais leve, ~nulo p/ acuracia
    'verbosity':          -1,
    'seed':               42,
    'num_threads':        22,     # CRITICO: 0 (=44 threads) cruza os 4 nos NUMA da VM -> ~43x mais lento
}
EARLY      = 300       # paciencia coerente com lr 0.02
LOG_PERIOD = 100       # imprime metricas (l2/l1) da validacao a cada N epocas de boosting
FRAC_VALID = 0.20      # fracao das celulas de treino p/ early stopping (devolve dados ao fit)
SEED       = 42


# ============================================================
# CLI — conjunto de features, folds, rounds e tag de saida (defaults reproduzem o run 53/10-fold)
# ============================================================

def _resolver_features(spec):
    # spec: '33'|'53'|'55' (nomeado), lista 'a,b,c', ou '@arquivo' (1+ por linha,
    # separados por virgula; texto apos '#' na linha e comentario)
    if spec in CONJUNTOS:
        return list(CONJUNTOS[spec]), spec
    if spec.startswith('@'):
        lst = []
        for linha in Path(spec[1:]).read_text().splitlines():
            linha = linha.split('#', 1)[0].strip()      # remove comentario inline
            lst += [t.strip() for t in linha.split(',') if t.strip()]
        return lst, spec
    return [t.strip() for t in spec.split(',') if t.strip()], 'custom'

ap = argparse.ArgumentParser(description='Ensaio LightGBM (F6) — CV leave-cell-out do residuo.')
ap.add_argument('--features', default='53', help="conjunto: 33|53|55, lista 'a,b,c', ou @arquivo")
ap.add_argument('--folds', type=int, default=10, help='nº de folds do GroupKFold (leave-cell-out)')
ap.add_argument('--num-round', type=int, default=16000, help='rounds de boosting (early stopping limita)')
ap.add_argument('--tag', default='', help='sufixo nas saidas (oof_ensaio{tag}.parquet, etc)')
ap.add_argument('--threads', type=int, default=22, help='num_threads LightGBM (nunca 0 nesta VM)')
args = ap.parse_args()

FEATURES, FEAT_SPEC = _resolver_features(args.features)
N_SPLITS  = args.folds
NUM_ROUND = args.num_round
TAG       = args.tag
PARAMS['num_threads'] = args.threads
print(f'F6: conjunto={FEAT_SPEC} ({len(FEATURES)} feats) | folds={N_SPLITS} | '
      f'num_round={NUM_ROUND} | tag="{TAG}"', flush=True)


# ============================================================
# FUNCOES
# ============================================================

def metricas(err):
    err = err[np.isfinite(err)]
    return {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'me': float(np.mean(err)), 'n': int(err.size)}


def skill(err_down, err_base):
    md, mb = metricas(err_down), metricas(err_base)
    ss = 1.0 - md['rmse'] / mb['rmse'] if mb['rmse'] > 0 else float('nan')
    return md, mb, ss


def estrato(df, col_grupo, err_down, err_base):
    linhas = []
    for g, idx in df.groupby(col_grupo).groups.items():
        ii = df.index.get_indexer(idx)
        md, mb, ss = skill(err_down[ii], err_base[ii])
        linhas.append({col_grupo: g, 'n': md['n'], 'rmse_era5': round(mb['rmse'], 3),
                       'rmse_down': round(md['rmse'], 3), 'mae_down': round(md['mae'], 3),
                       'SS': round(ss, 4)})
    return pd.DataFrame(linhas)


# ============================================================
# LEITURA
# ============================================================

print('Lendo matriz...', flush=True)
df = pd.read_parquet(MATRIZ)
faltando = [c for c in FEATURES if c not in df.columns]
assert not faltando, f'features ausentes na matriz: {faltando}'
feats = FEATURES
X = df[feats]
y = df['y_residuo'].values.astype('float32')
grp = df['cell_id'].values
print(f'  {len(df):,} linhas | {len(feats)} features | {df["cell_id"].nunique()} celulas', flush=True)


# ============================================================
# LEAVE-STATION-OUT (GroupKFold) — predicoes out-of-fold
# ============================================================

oof = np.full(len(df), np.nan, dtype='float64')
imp_gain = np.zeros(len(feats)); imp_split = np.zeros(len(feats))
best_iters = []
mae_tr, bias_tr = [], []      # MAE e BIAS in-sample (fit) por fold -> diagnostico de overfit
tempos = []                   # tempo (s) de cada fold (carve + dataset + treino + predicao)
rng = np.random.default_rng(SEED)

gkf = GroupKFold(n_splits=N_SPLITS)
for fold, (tr, te) in enumerate(gkf.split(X, y, grp), 1):
    t0 = time.time()
    # carve valid agrupado por celula a partir do treino (early stopping)
    cel_tr = np.unique(grp[tr])
    n_val = max(1, int(round(FRAC_VALID * len(cel_tr))))
    cel_val = set(rng.choice(cel_tr, size=n_val, replace=False).tolist())
    mask_val = np.array([g in cel_val for g in grp[tr]])
    i_val = tr[mask_val]; i_fit = tr[~mask_val]

    dfit = lgb.Dataset(X.iloc[i_fit], label=y[i_fit], free_raw_data=False)
    dval = lgb.Dataset(X.iloc[i_val], label=y[i_val], reference=dfit, free_raw_data=False)
    print(f'\n=== fold {fold}/{N_SPLITS} — treinando (estatisticas da validacao a cada {LOG_PERIOD} epocas) ===', flush=True)
    model = lgb.train(PARAMS, dfit, num_boost_round=NUM_ROUND, valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(EARLY, verbose=True),
                                 lgb.log_evaluation(LOG_PERIOD)])
    oof[te] = model.predict(X.iloc[te], num_iteration=model.best_iteration)
    best_iters.append(int(model.best_iteration))
    imp_gain  += model.feature_importance('gain')
    imp_split += model.feature_importance('split')

    # treino in-sample (fit) — diagnostico de overfit (gap treino vs OOF)
    e_tr = model.predict(X.iloc[i_fit], num_iteration=model.best_iteration) - y[i_fit]
    m_tr = metricas(e_tr)
    mae_tr.append(m_tr['mae']); bias_tr.append(m_tr['me'])

    dt = time.time() - t0
    tempos.append(dt)

    e_d = oof[te] - y[te]; e_b = -y[te].astype('float64')
    md, mb, ss = skill(e_d, e_b)
    print(f'  fold {fold}/{N_SPLITS}: {len(np.unique(grp[te]))} celulas teste | '
          f'best_iter={model.best_iteration} | RMSE ERA5={mb["rmse"]:.3f} -> down={md["rmse"]:.3f} | SS={ss:.4f}'
          f' | treino MAE={m_tr["mae"]:.3f} BIAS={m_tr["me"]:+.3f}'
          f' | tempo={dt:.1f}s ({dt/60:.1f} min)',
          flush=True)


# ============================================================
# METRICAS GLOBAIS E ESTRATIFICADAS (vs baseline ERA5)
# ============================================================

err_down = oof - y                      # T_final - obs
err_base = -y.astype('float64')         # T_ERA5 - obs
md, mb, ss = skill(err_down, err_base)
print('\n=== ENSAIO (leave-station-out) — global ===', flush=True)
print(f'  baseline ERA5 : MAE={mb["mae"]:.3f} RMSE={mb["rmse"]:.3f} ME={mb["me"]:.3f}', flush=True)
print(f'  downscaling   : MAE={md["mae"]:.3f} RMSE={md["rmse"]:.3f} ME={md["me"]:.3f}', flush=True)
print(f'  treino (fit)  : MAE={np.mean(mae_tr):.3f} BIAS={np.mean(bias_tr):+.3f}  (media dos folds)', flush=True)
print(f'  SKILL SCORE   : SS = {ss:.4f}  (RMSE -{100*(1-md["rmse"]/mb["rmse"]):.1f}%)', flush=True)
print(f'  TEMPO         : total={sum(tempos):.1f}s ({sum(tempos)/60:.1f} min) | '
      f'media/fold={np.mean(tempos):.1f}s | por fold={[round(t, 1) for t in tempos]}', flush=True)

df = df.reset_index(drop=True)
df['hora'] = df['data_hora_utc'].dt.hour
df['mes']  = df['data_hora_utc'].dt.month
df['faixa_elev'] = pd.cut(df['z_alvo'], [-1, 200, 500, 1000, 2000],
                          labels=['0-200', '200-500', '500-1000', '>1000'])
df['faixa_costa'] = pd.cut(df['dist_oceano'] / 1000, [-1, 10, 50, 150, 1000],
                           labels=['0-10km', '10-50km', '50-150km', '>150km'])

estr = {c: estrato(df, c, err_down, err_base) for c in ['hora', 'faixa_elev', 'faixa_costa', 'mes']}
print('\n--- SS por hora UTC (noite SC ~ 03-09 UTC) ---', flush=True)
print(estr['hora'].to_string(index=False), flush=True)
print('\n--- SS por faixa de elevacao ---', flush=True)
print(estr['faixa_elev'].to_string(index=False), flush=True)
print('\n--- SS por faixa de distancia a costa ---', flush=True)
print(estr['faixa_costa'].to_string(index=False), flush=True)


# ============================================================
# IMPORTANCIA E SAIDAS
# ============================================================

imp = (pd.DataFrame({'feature': feats, 'gain': imp_gain / N_SPLITS, 'split': imp_split / N_SPLITS})
       .sort_values('gain', ascending=False).reset_index(drop=True))
print('\n--- top 15 features (gain medio) ---', flush=True)
print(imp.head(15).to_string(index=False), flush=True)

SAIDA.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'cell_id': df['cell_id'], 'data_hora_utc': df['data_hora_utc'],
              'y_residuo': y, 'pred_residuo': oof}).to_parquet(SAIDA / f'oof_ensaio{TAG}.parquet', index=False)
imp.to_csv(SAIDA / f'importancia_ensaio{TAG}.csv', sep=';', index=False, float_format='%.1f')
rel = {'cv': 'leave-cell-out GroupKFold', 'n_splits': N_SPLITS, 'best_iters': best_iters,
       'conjunto': FEAT_SPEC, 'n_features': len(feats), 'features': list(feats),
       'params': PARAMS, 'num_round': NUM_ROUND, 'early': EARLY,
       'tempo': {'total_s': round(sum(tempos), 1), 'media_fold_s': round(float(np.mean(tempos)), 1),
                 'por_fold_s': [round(t, 1) for t in tempos]},
       'global': {'baseline': mb, 'downscaling': md, 'skill_score': ss},
       'treino': {'mae': float(np.mean(mae_tr)), 'bias': float(np.mean(bias_tr)),
                  'mae_por_fold': [round(x, 3) for x in mae_tr],
                  'bias_por_fold': [round(x, 3) for x in bias_tr]},
       'por_hora': estr['hora'].to_dict('records'),
       'por_elevacao': estr['faixa_elev'].to_dict('records'),
       'por_costa': estr['faixa_costa'].to_dict('records')}
with open(SAIDA / f'metricas_ensaio{TAG}.json', 'w') as f:
    json.dump(rel, f, indent=2, ensure_ascii=False)
print(f'\nSalvos em {SAIDA}: oof_ensaio{TAG}.parquet, importancia_ensaio{TAG}.csv, metricas_ensaio{TAG}.json', flush=True)
