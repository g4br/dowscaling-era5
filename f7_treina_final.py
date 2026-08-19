#!/usr/bin/env python3

# ============================================================
# F7 — Treino do modelo FINAL (LightGBM) para producao/inferencia.
# O F6 so faz validacao cruzada (leave-cell-out) e NAO persiste modelo. Este
# script treina UMA vez sobre a matriz COMPLETA e salva o booster + metadados,
# que a inferencia (F8) carrega para prever na grade de 1 km.
# Alvo: y_residuo = T_obs - T_ERA5_bilinear.  Inferencia: T_final = T_ERA5 + f(X).
# Mesmos PARAMS do F6 (inclui bagging_freq=5 e max_bin=127). Na CV o early stopping
# nunca foi vinculante (best_iter ~= 16000 = NUM_ROUND em todos os folds), logo o
# padrao aqui e ajustar NUM_ROUND fixo (=16000) em TODOS os dados (sem reservar
# validacao). Com --early N, em vez disso reserva 20% das CELULAS p/ early stopping
# e salva o modelo no best_iteration (treinado so na parte restante).
# Features: --features {28|33|53|55} (default 53 = conjunto que gerou os artefatos
# do ensaio F6; 55 = 53 + ssr/str individuais, vencedor da ablacao 5-fold, SS 0,224;
# 28 = reducao planejada em selecao_variaveis.md, ainda NAO validada).
# NaN nas features e nativo do LightGBM (nao imputar).
# Uso: python f7_treina_final.py [--features 53] [--num-round 16000] [--early 0]
# Rodar no env com lightgbm/pyarrow (py312). Treino e pesado (~horas).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from _root import ROOT, DIR_MATRIZ


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ   = Path(ROOT)
MATRIZ = DIR_MATRIZ / 'matriz_treino.parquet'
SAIDA  = RAIZ / 'saidas' / 'modelos'

# Conjunto de 28 features = reducao planejada em selecao_variaveis.md (ainda NAO
# validada por ensaio). 53 = conjunto EXATO que gerou os artefatos do ensaio F6
# (metricas_ensaio.json / importancia_ensaio.csv) — mesma ordem do F6.
FEATURES_28 = [
    'lapse', 'u10', 'v10', 'vel_vento', 'dep_orvalho', 'dt_ar_solo', 'rad_liquida',
    'dt_ar_pele', 'razao_bowen', 'delta_z',
    'dev_r1000_p10_1km', 'dev_r3000_mean_1km', 'dev_r300_p10_1km', 'eastness_mean_1km',
    'hand_mean_1km', 'mrvbf_p90_1km', 'northness_mean_1km', 'tri_mean_1km', 'fracao_t1',
    'fracao_t3', 'fracao_t4', 'fracao_t5', 'fracao_t6', 'fracao_t7', 'fracao_t8',
    'z_alvo', 'sp', 'dist_oceano',
]
FEATURES_33 = FEATURES_28 + ['hora_sin', 'hora_cos', 'doy_sin', 'doy_cos', 'coord_y']
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
    'dev_r300_mean_1km', 'hand_p10_1km', 'slope_max_1km',
    'slope_mean_1km', 'svf_mean_1km', 'svf_p10_1km', 'tri_max_1km',
    'tri_std_1km', 'umidade_relativa', 'z_std_1km',
    'hora_sin', 'hora_cos', 'doy_sin', 'doy_cos',
    'coord_x', 'coord_y',
]
# 55 = 53 + radiacao solar (onda-curta) e termal (onda-longa) individuais.
FEATURES_55 = FEATURES_53 + ['ssr', 'str']

# baseline e alvo (mesma convencao do F5/F6) — a inferencia precisa destes
LAPSE_RATE   = 0.0065
BASELINE_VAR = 't2m_1km'           # T_ERA5 = t2m_1km - 273.15
ALVO         = 'y_residuo'

PARAMS = {
    'objective':          'regression_l2',
    'metric':             ['l2', 'l1'],
    'learning_rate':      0.02,
    'num_leaves':         63,
    'min_child_samples':  500,
    'feature_fraction':   0.8,
    'bagging_fraction':   0.7,
    'bagging_freq':       5,      # casa com o ensaio F6 (era 1)
    'lambda_l1':          1.0,
    'lambda_l2':          5.0,
    'max_bin':            127,    # casa com o ensaio F6 (default e 255)
    'verbosity':          -1,
    'seed':               42,
    'num_threads':        22,     # VM 44 vCPU / 4 NUMA: 0 (todas) e ~43x mais lento; 22 = otimo
}
NUM_ROUND  = 16000               # = teto do ensaio; best_iter dos folds ~= 16000 (early nao vinculou)
LOG_PERIOD = 100
FRAC_VALID = 0.20
SEED       = 42


# ============================================================
# FUNCOES
# ============================================================

def metricas(err):
    err = err[np.isfinite(err)]
    return {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'me': float(np.mean(err)), 'n': int(err.size)}


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Treino do modelo final LightGBM (F7).')
    ap.add_argument('--features', choices=['28', '33', '53', '55'], default='53')
    ap.add_argument('--num-round', type=int, default=NUM_ROUND)
    ap.add_argument('--threads', type=int, default=PARAMS['num_threads'],
                    help='num_threads do LightGBM (nesta VM, 22; nao usar 0 = ~43x mais lento)')
    ap.add_argument('--early', type=int, default=0,
                    help='paciencia early stopping (0=off; reserva 20%% das celulas p/ validar)')
    ap.add_argument('--matriz', default=str(MATRIZ))
    ap.add_argument('--saida',  default=str(SAIDA))
    ap.add_argument('--nome',   default='modelo_final', help='prefixo dos arquivos de saida')
    args = ap.parse_args()

    feats = {'28': FEATURES_28, '33': FEATURES_33, '53': FEATURES_53,
             '55': FEATURES_55}[args.features]
    PARAMS['num_threads'] = args.threads      # NUMA: nunca 0 nesta VM (ver memoria do projeto)
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    f_modelo = saida / f'{args.nome}.txt'
    f_meta   = saida / f'{args.nome}_meta.json'

    print(f'Lendo matriz: {args.matriz}', flush=True)
    cols = list(dict.fromkeys(feats + [ALVO, 'cell_id']))
    df = pd.read_parquet(args.matriz, columns=cols)
    faltando = [c for c in feats if c not in df.columns]
    assert not faltando, f'features ausentes na matriz: {faltando}'

    X = df[feats]
    y = df[ALVO].values.astype('float32')
    fin = np.isfinite(y)
    if not fin.all():
        X = X[fin]; y = y[fin]; df = df[fin]
    print(f'  {len(df):,} linhas | {len(feats)} features ({args.features}) | '
          f'{df["cell_id"].nunique()} celulas', flush=True)

    t0 = time.time()
    if args.early > 0:
        # reserva 20% das CELULAS p/ early stopping (leave-cell-out, sem vazamento)
        grp = df['cell_id'].values
        rng = np.random.default_rng(SEED)
        cel = np.unique(grp)
        cel_val = set(rng.choice(cel, size=max(1, int(round(FRAC_VALID * len(cel)))),
                                 replace=False).tolist())
        m_val = np.array([gg in cel_val for gg in grp])
        dfit = lgb.Dataset(X[~m_val], label=y[~m_val], free_raw_data=False)
        dval = lgb.Dataset(X[m_val],  label=y[m_val], reference=dfit, free_raw_data=False)
        print(f'Treinando com early stopping (paciencia {args.early}) | '
              f'fit={int((~m_val).sum()):,}  val={int(m_val.sum()):,} ...', flush=True)
        model = lgb.train(PARAMS, dfit, num_boost_round=args.num_round,
                          valid_sets=[dval], valid_names=['val'],
                          callbacks=[lgb.early_stopping(args.early, verbose=True),
                                     lgb.log_evaluation(LOG_PERIOD)])
        best_it = int(model.best_iteration)
    else:
        # ajuste em TODOS os dados, NUM_ROUND fixo (CV: best_iter ~= cap)
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        print(f'Treinando em todos os dados | {args.num_round} rounds (sem early stopping) ...', flush=True)
        model = lgb.train(PARAMS, dtrain, num_boost_round=args.num_round,
                          valid_sets=[dtrain], valid_names=['train'],
                          callbacks=[lgb.log_evaluation(LOG_PERIOD)])
        best_it = int(model.current_iteration())
    dt = time.time() - t0

    # diagnostico in-sample (T_final - obs no proprio treino)
    e_tr = model.predict(X, num_iteration=best_it) - y
    m_tr = metricas(e_tr)
    print(f'\nTreino concluido em {dt/60:.1f} min | best_iter={best_it} | '
          f'in-sample MAE={m_tr["mae"]:.3f} RMSE={m_tr["rmse"]:.3f} BIAS={m_tr["me"]:+.3f}', flush=True)

    # importancia (sanity)
    imp = (pd.DataFrame({'feature': feats,
                         'gain': model.feature_importance('gain'),
                         'split': model.feature_importance('split')})
           .sort_values('gain', ascending=False).reset_index(drop=True))
    print('\n--- top 12 features (gain) ---', flush=True)
    print(imp.head(12).to_string(index=False), flush=True)

    # ========================================================
    # SAIDA: booster + metadados (contrato com a inferencia F8)
    # ========================================================

    model.save_model(str(f_modelo), num_iteration=best_it)
    imp.to_csv(saida / f'{args.nome}_importancia.csv', sep=';', index=False, float_format='%.1f')

    meta = {
        'modelo': f_modelo.name,
        'criado_em': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'features': feats,
        'n_features': len(feats),
        'conjunto_features': args.features,
        'alvo': ALVO,
        'baseline': {'var': BASELINE_VAR, 'offset_celsius': -273.15,
                     'formula': 'T_ERA5 = t2m_1km - 273.15 ; T_final = T_ERA5 + pred_residuo'},
        'lapse_rate': LAPSE_RATE,
        'params': PARAMS,
        'num_round': args.num_round,
        'best_iteration': best_it,
        'early_stopping': args.early,
        'n_train': int(len(df)),
        'n_celulas': int(df['cell_id'].nunique()),
        'matriz': str(args.matriz),
        'in_sample': m_tr,
        'tempo_min': round(dt / 60, 1),
    }
    f_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nSalvos em {saida}:', flush=True)
    print(f'  {f_modelo.name} | {f_meta.name} | {args.nome}_importancia.csv', flush=True)


if __name__ == '__main__':
    main()
