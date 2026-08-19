#!/usr/bin/env python3

# ============================================================
# F7q — Treino QUANTILICO do residuo (pinball) na MATRIZ COMPLETA.
# CORRIGE o experimento anterior (L2 por SUBCONJUNTO de temperatura): segmentar os
# dados de treino por t2m_1km e ajustar um L2 (media) por ramo NAO produz quantis —
# 'frio' e 'quente' viravam duas medias quase iguais no mesmo pixel, entao a banda
# [t2m_p16, t2m_p84] colapsava (cobertura OOF ~2-7% vs ~68% esperado). O contrato dos
# scripts de avaliacao (f12/f13: t2m_p16->alpha0.16, t2m_p16a84->0.50, t2m_p84->0.84;
# winkler/cobertura sobre a banda p16-p84) SEMPRE foi quantilico — era o treino que
# estava errado.
#
# A rodada de PRODUCAO (v2) tem QUATRO boosters, TODOS na matriz completa:
#   frio     : objective=quantile alpha=q_lo (default 0.16)  -> var t2m_p16   (banda inf.)
#   meio     : objective=quantile alpha=0.50 (mediana)       -> var t2m_p16a84
#   quente   : objective=quantile alpha=q_hi (default 0.84)  -> var t2m_p84   (banda sup.)
#   completo : objective=regression_l2 (MEDIA — inalterado)  -> var t2m       (central)
# O alvo e o residuo: y_residuo = T_obs - T_ERA5_bilinear ; T_q = T_ERA5 + pred_residuo.
# Cobertura no residuo == cobertura na temperatura (soma-se o mesmo T_ERA5 aos dois).
#
# Na INFERENCIA (F8/oper v2) NAO ha roteamento: os 4 boosters preveem a grade inteira,
# cada um numa variavel NetCDF propria. A banda p16-p84 vira um intervalo de predicao de
# ~68% e a variavel 't2m' (L2) segue como estimativa central (SS +0,33, ja validada — o
# ramo 'completo' NAO muda). MESMO tipo/saida_vars do meta -> F8/oper/f10-f14 inalterados.
#
# NOTA (monotonicidade): os 3 quantis sao boosters INDEPENDENTES e podem cruzar em
# alguns pixels (p16>p84). O F8 hoje NAO ordena; se a taxa de cruzamento do --cv for
# relevante, aplicar np.sort no eixo dos quantis por pixel no F8 (nao impor no treino).
#
# NOTA (regularizacao da cauda): lambda_l2=5 e min_child_samples=500 herdados do F7
# comprimem os quantis extremos p/ o centro (sub-cobertura). --lambda-l2 / --min-child
# relaxam SO os ramos quantilicos (completo mantem o F7). O --cv diz se precisa.
#
# NOTA (num_round): o best_iter ~16000 do F6/F7 e do L2. A pinball tem hessiana
# degenerada e converge diferente — NAO assumir 16000. Rode --cv (early stopping na
# metrica 'quantile' por ramo) p/ ler o best_iter e o VEREDITO de cobertura; depois
# treine o final com --early (cada booster acha o seu) ou --num-round fixo no valor lido.
#
# Anti-vazamento no --cv: limiares informativos e quantis sao do fold de treino; o OOF
# de cada quantil e previsto no fold de teste. NaN nas features e nativo do LightGBM
# (nao imputar); t2m_1km com NaN e falha ALTO (baseline, tem que existir em toda linha).
# Rodar no env py312 (lightgbm/pyarrow/sklearn/matplotlib).
#
# Uso (validar):  python f7q_treina_quantil.py --cv --features 55 --folds 5 --q-lo 0.16 --q-hi 0.84
# Uso (final v2): python f7q_treina_quantil.py --features 55 --q-lo 0.16 --q-hi 0.84 --early 300
#   -> gera modelo_quantil_q16_q84_{frio,meio,quente,completo}.txt + modelo_quantil_q16_q84_meta.json
#
# --cv --apenas-completo SUBSTITUI o F6: CV leave-cell-out SO do ramo L2 central,
# gera oof_residuo_loco_{n}feat.parquet (drop-in p/ o f8b --oof) + relatorio
# estratificado (SS por hora/elevacao/costa/mes, importancia media/fold, gap
# treino-vs-OOF). Fonte unica de features/params = F7 (importada acima).
# Uso (ensaio L2): python f7q_treina_quantil.py --cv --apenas-completo --features 55 --folds 10
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
from sklearn.model_selection import GroupKFold

# fonte unica das features/baseline/alvo/params — importar do F7 evita drift.
# (f7_treina_final guarda toda a logica em main(), entao importa-lo nao dispara treino.)
from _root import ROOT, DIR_MATRIZ
from f7_treina_final import (
    FEATURES_28, FEATURES_33, FEATURES_53, FEATURES_55,
    LAPSE_RATE, BASELINE_VAR, ALVO,
    PARAMS as F7_PARAMS, NUM_ROUND as F7_NUM_ROUND,
    FRAC_VALID, SEED, LOG_PERIOD,
)


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ   = Path(ROOT)
MATRIZ = DIR_MATRIZ / 'matriz_treino.parquet'
SAIDA  = RAIZ / 'saidas' / 'modelos'

CONJUNTOS = {'28': FEATURES_28, '33': FEATURES_33, '53': FEATURES_53, '55': FEATURES_55}

# 3 boosters quantilicos (banda p16/p50/p84) + 1 L2 (central). Todos na matriz COMPLETA.
RAMOS_Q       = ('frio', 'meio', 'quente')          # ramos quantilicos (alpha q_lo/0.50/q_hi)
RAMO_COMPLETO = 'completo'                           # 4o ramo: regression_l2 (media), como o F7
RAMOS_V2      = RAMOS_Q + (RAMO_COMPLETO,)           # ordem das 4 saidas do treino final v2
ALPHA_MEIO    = 0.50                                 # mediana (casa com t2m_p16a84->0.50 no f12/f13)
KELVIN        = 273.15
EARLY_CV      = 300      # paciencia do early stopping no --cv (coerente com lr 0.02)
TOL_COB       = 0.03     # tolerancia do VEREDITO de cobertura (|cob - alvo|)

# cortes de estratificacao — IDENTICOS ao F6 (para casar as tabelas por faixa)
ELEV_BINS,  ELEV_LABS  = [-1, 200, 500, 1000, 2000], ['0-200', '200-500', '500-1000', '>1000']
COSTA_BINS, COSTA_LABS = [-1, 10, 50, 150, 1000],    ['0-10km', '10-50km', '50-150km', '>150km']
# foco declarado: vale alto (>500 m) + noite (03-09 UTC) — regime de geada, onde o q16
# de temperatura minima TEM de estar calibrado p/ a banda valer a pena.
VAN_Z_MIN, VAN_HORA = 500.0, (3, 9)


# ============================================================
# FUNCOES — quantis, metricas e utilidades
# ============================================================

def alpha_ramo(ramo, q_lo, q_hi):
    # alpha do objective quantile por ramo; None => regression_l2 (ramo 'completo').
    return {'frio': q_lo, 'meio': ALPHA_MEIO, 'quente': q_hi, RAMO_COMPLETO: None}[ramo]


def limiares(t2m, q_lo, q_hi):
    # temperaturas (K) nos quantis q_lo/q_hi de t2m_1km. So INFORMATIVO no v2 (sem
    # roteamento); persistido no meta p/ compatibilidade com o F8 (le corte.t_lo_K/t_hi_K).
    return float(np.quantile(t2m, q_lo)), float(np.quantile(t2m, q_hi))


def nomes_saida(q_lo, q_hi):
    # ramo -> nome da variavel NetCDF no F8 v2 (contrato com f10-f14, NAO mudar):
    #   frio->t2m_p16 ; meio->t2m_p16a84 ; quente->t2m_p84 ; completo->t2m.
    lo, hi = int(round(q_lo * 100)), int(round(q_hi * 100))
    return {'frio': f't2m_p{lo}', 'meio': f't2m_p{lo}a{hi}',
            'quente': f't2m_p{hi}', RAMO_COMPLETO: 't2m'}


def params_ramo(threads, alpha=None, lambda_l2=None, min_child=None):
    # PARAMS do F7; alpha!=None troca p/ objective='quantile' (com metrica pinball).
    # lambda_l2/min_child sobrescrevem SO os ramos quantilicos (relaxar a cauda).
    p = dict(F7_PARAMS)                 # copia — nao mutar o dict do F7
    p['num_threads'] = int(threads)     # NUMA: nunca 0 nesta VM (~43x mais lento)
    if alpha is not None:
        p['objective'] = 'quantile'
        p['alpha']     = float(alpha)
        p['metric']    = 'quantile'     # pinball (substitui o ['l2','l1'] do F7)
        if lambda_l2 is not None:
            p['lambda_l2'] = float(lambda_l2)
        if min_child is not None:
            p['min_child_samples'] = int(min_child)
    return p


def pinball(y, q, alpha):
    # perda pinball media (metrica-alvo do quantil; menor = melhor).
    e = np.asarray(y, 'float64') - np.asarray(q, 'float64')
    return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))


def cobertura(y, q):
    # P(y <= q): fracao de residuos observados abaixo do quantil previsto (alvo = alpha).
    return float(np.mean(np.asarray(y, 'float64') <= np.asarray(q, 'float64')))


def metricas(err):
    err = err[np.isfinite(err)]
    return {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'me': float(np.mean(err)), 'n': int(err.size)}


def skill(err_down, err_base):
    # SS = 1 - RMSE_down / RMSE_ERA5 (identico ao F6/F6b).
    md, mb = metricas(err_down), metricas(err_base)
    ss = 1.0 - md['rmse'] / mb['rmse'] if mb['rmse'] > 0 else float('nan')
    return md, mb, ss


def estrato(df, col_grupo, err_down, err_base):
    # SS por nivel de col_grupo (porte do F6, para --cv --apenas-completo).
    linhas = []
    for g, idx in df.groupby(col_grupo, observed=True).groups.items():
        ii = df.index.get_indexer(idx)
        md, mb, ss = skill(err_down[ii], err_base[ii])
        linhas.append({col_grupo: g, 'n': md['n'], 'rmse_era5': round(mb['rmse'], 3),
                       'rmse_down': round(md['rmse'], 3), 'mae_down': round(md['mae'], 3),
                       'SS': round(ss, 4)})
    return pd.DataFrame(linhas)


def linha_cob(estrato, faixa, y, p_lo, p_med, p_hi):
    # cobertura das 3 bandas + do intervalo [p_lo,p_hi] + largura media, num subconjunto.
    return {'estrato': estrato, 'faixa': str(faixa), 'n': int(y.size),
            'cob_lo': cobertura(y, p_lo), 'cob_med': cobertura(y, p_med),
            'cob_hi': cobertura(y, p_hi),
            'cob_int': float(np.mean((y >= p_lo) & (y <= p_hi))) if y.size else float('nan'),
            'larg': float(np.mean(p_hi - p_lo)) if y.size else float('nan')}


def df_para_md(df, casas=4):
    # tabela markdown sem depender de 'tabulate' (mesmo helper do F6d).
    def fmt(v):
        return f'{v:.{casas}f}' if isinstance(v, float) else str(v)
    cols = list(df.columns)
    linhas = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        linhas.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
    return '\n'.join(linhas)


# ============================================================
# LEITURA DA MATRIZ (compartilhada pelos dois modos)
# ============================================================

def carregar_matriz(caminho, feats, com_estratos):
    # falha ALTO se a matriz, uma feature ou o baseline faltar. Descarta linhas de
    # y NaN; t2m_1km NaN (baseline) e erro, nao descarte silencioso.
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f'matriz nao encontrada: {caminho}')
    extra = ['data_hora_utc', 'z_alvo', 'dist_oceano'] if com_estratos else []
    cols = list(dict.fromkeys(feats + [ALVO, 'cell_id', BASELINE_VAR] + extra))
    print(f'Lendo matriz: {caminho}', flush=True)
    df = pd.read_parquet(caminho, columns=cols)
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        raise KeyError(f'colunas ausentes na matriz: {faltando}')

    y = df[ALVO].to_numpy('float32')
    fin = np.isfinite(y)
    n0 = len(df)
    df = df[fin].reset_index(drop=True) if not fin.all() else df.reset_index(drop=True)

    t2m = df[BASELINE_VAR].to_numpy('float64')
    n_nan = int((~np.isfinite(t2m)).sum())
    if n_nan:
        raise ValueError(f'{BASELINE_VAR} com {n_nan} NaN — baseline tem que existir em '
                         f'toda linha (checar F5).')
    print(f'  {n0:,} linhas | {n0 - len(df):,} descartadas (y NaN) | {len(df):,} usadas | '
          f'{len(feats)} features | {df["cell_id"].nunique()} celulas', flush=True)
    return df


# ============================================================
# TREINO DE UM BOOSTER (compartilhado pelo --cv e pelo final)
# ============================================================

def carve_valid(grp_tr, rng):
    # reserva FRAC_VALID das CELULAS do treino p/ early stopping (sem vazamento).
    cel = np.unique(grp_tr)
    n_val = max(1, int(round(FRAC_VALID * len(cel))))
    cel_val = set(rng.choice(cel, size=n_val, replace=False).tolist())
    return np.array([g in cel_val for g in grp_tr])


def treina_ramo(X, y, grp, idx, num_round, early, threads, rng,
                alpha=None, lambda_l2=None, min_child=None):
    # treina UM booster nas linhas `idx`. alpha=None -> L2; alpha -> quantile.
    # early>0 reserva FRAC_VALID das celulas p/ early stopping; senao num_round fixo.
    params = params_ramo(threads, alpha, lambda_l2, min_child)
    if early > 0:
        m_val = carve_valid(grp[idx], rng)
        i_val, i_fit = idx[m_val], idx[~m_val]
        dfit = lgb.Dataset(X.iloc[i_fit], label=y[i_fit], free_raw_data=False)
        dval = lgb.Dataset(X.iloc[i_val], label=y[i_val], reference=dfit, free_raw_data=False)
        model = lgb.train(params, dfit, num_boost_round=num_round,
                          valid_sets=[dval], valid_names=['val'],
                          callbacks=[lgb.early_stopping(early, verbose=False),
                                     lgb.log_evaluation(LOG_PERIOD)])
        return model, int(model.best_iteration)
    dtrain = lgb.Dataset(X.iloc[idx], label=y[idx], free_raw_data=False)
    model = lgb.train(params, dtrain, num_boost_round=num_round,
                      valid_sets=[dtrain], valid_names=['train'],
                      callbacks=[lgb.log_evaluation(LOG_PERIOD)])
    return model, int(model.current_iteration())


# ============================================================
# MODO --cv — validacao leave-cell-out (decide o aceite: cobertura da banda)
# ============================================================

def figuras_cv(figs, cob_global, alvos, tab_elev, best_iters):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cores = {'frio': '#4C72B0', 'meio': '#55A868', 'quente': '#C44E52'}

    # --- fig_cobertura: cobertura global por quantil vs alvo (alpha) ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(len(RAMOS_Q))
    a1.bar(xs, [cob_global[r] for r in RAMOS_Q], color=[cores[r] for r in RAMOS_Q],
           edgecolor='white')
    for i, r in enumerate(RAMOS_Q):
        a1.plot([i - 0.4, i + 0.4], [alvos[r], alvos[r]], 'k--', lw=1.2)
        a1.text(i, cob_global[r], f'{cob_global[r]:.3f}', ha='center', va='bottom', fontsize=9)
    a1.set_xticks(xs); a1.set_xticklabels([f'{r}\n(a={alvos[r]:.2f})' for r in RAMOS_Q])
    a1.set_ylabel('cobertura P(y<=q)'); a1.set_ylim(0, 1)
    a1.set_title('Cobertura OOF por quantil (traco = alvo)'); a1.grid(axis='y', alpha=0.3)
    # intervalo
    a2.bar([0], [cob_global['intervalo']], color='#2c7fb8', edgecolor='white', width=0.5)
    a2.axhline(alvos['intervalo'], color='k', ls='--', lw=1.2,
               label=f'alvo {alvos["intervalo"]:.2f}')
    a2.text(0, cob_global['intervalo'], f'{cob_global["intervalo"]:.3f}',
            ha='center', va='bottom', fontsize=10)
    a2.set_xticks([0]); a2.set_xticklabels(['banda p16-p84'])
    a2.set_ylabel('cobertura do intervalo'); a2.set_ylim(0, 1)
    a2.set_title('Cobertura da banda'); a2.grid(axis='y', alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'fig_cobertura.png', dpi=140); plt.close(fig)

    # --- fig_cobertura_elev: cobertura da banda por faixa de elevacao ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tab_elev['faixa'], tab_elev['cob_int'], 'o-', color='#2c7fb8', label='banda p16-p84')
    ax.axhline(alvos['intervalo'], color='k', ls='--', lw=1.0, label=f'alvo {alvos["intervalo"]:.2f}')
    ax.set_ylabel('cobertura do intervalo'); ax.set_ylim(0, 1)
    ax.set_title('Cobertura da banda por faixa de elevacao'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'fig_cobertura_elev.png', dpi=140); plt.close(fig)

    # --- fig_best_iter: best_iter por fold em cada ramo quantilico ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ramo in RAMOS_Q:
        bis = best_iters[ramo]
        ax.plot(np.arange(1, len(bis) + 1), bis, 'o-', color=cores[ramo], label=ramo)
    ax.set_xlabel('fold'); ax.set_ylabel('best_iteration')
    ax.set_title('best_iter por fold (early stopping pinball por ramo)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(figs / 'fig_best_iter.png', dpi=140); plt.close(fig)


def rodar_cv(args, feats):
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    tag = f'q{int(args.q_lo * 100):02d}q{int(args.q_hi * 100):02d}'
    aval = saida / f'avaliacao_quantil_{tag}'
    figs = aval / 'figuras'
    figs.mkdir(parents=True, exist_ok=True)

    df = carregar_matriz(args.matriz, feats, com_estratos=True)
    n_grp = df['cell_id'].nunique()
    if n_grp < args.folds:
        raise ValueError(f'poucas celulas p/ GroupKFold: {n_grp} < folds={args.folds}')
    X = df[feats]
    y32 = df[ALVO].to_numpy('float32')
    y = df[ALVO].to_numpy('float64')
    t2m = df[BASELINE_VAR].to_numpy('float64')
    grp = df['cell_id'].to_numpy()

    alvos_ramo = {'frio': args.q_lo, 'meio': ALPHA_MEIO, 'quente': args.q_hi}

    print(f'\n=== --cv leave-cell-out | {args.folds} folds | quantile alpha '
          f'{args.q_lo}/{ALPHA_MEIO}/{args.q_hi} (matriz completa, sem roteamento) ===', flush=True)
    # OOF de cada quantil (todo teste recebe previsao dos 3 boosters — sem roteamento)
    oof = {r: np.full(len(df), np.nan, dtype='float64') for r in RAMOS_Q}
    best_iters = {r: [] for r in RAMOS_Q}
    rng = np.random.default_rng(SEED)
    gkf = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, grp), 1):
        print(f'fold {fold}/{args.folds}: fit={tr.size:,} te={te.size:,} | '
              f'{len(np.unique(grp[te]))} celulas teste', flush=True)
        for ramo in RAMOS_Q:
            t0 = time.time()
            model, bi = treina_ramo(X, y32, grp, tr, args.num_round, EARLY_CV, args.threads,
                                    rng, alpha_ramo(ramo, args.q_lo, args.q_hi),
                                    args.lambda_l2, args.min_child)
            best_iters[ramo].append(bi)
            oof[ramo][te] = model.predict(X.iloc[te], num_iteration=bi)
            cob = cobertura(y[te], oof[ramo][te])
            print(f'  [{ramo:6s} a={alvos_ramo[ramo]:.2f}] best_iter={bi} | '
                  f'cob(oof)={cob:.3f} (alvo {alvos_ramo[ramo]:.2f}) | {time.time() - t0:.1f}s',
                  flush=True)

    if any(not np.isfinite(oof[r]).all() for r in RAMOS_Q):
        raise RuntimeError('OOF incompleto — ha linhas sem previsao em algum quantil.')

    p_lo, p_med, p_hi = oof['frio'], oof['meio'], oof['quente']
    alvos = dict(alvos_ramo, intervalo=args.q_hi - args.q_lo)

    # OOF cru -> parquet (contrato de diagnostico: 3 quantis do residuo)
    oof_df = pd.DataFrame({'cell_id': df['cell_id'], 'data_hora_utc': df['data_hora_utc'],
                           'y_residuo': y, BASELINE_VAR: t2m,
                           'pred_p_lo': p_lo, 'pred_p_med': p_med, 'pred_p_hi': p_hi})
    oof_path = saida / f'oof_quantil_{tag}.parquet'
    oof_df.to_parquet(oof_path, index=False)

    # ---- metricas OOF globais: cobertura, pinball, cruzamento, largura ----
    cob_global = {'frio': cobertura(y, p_lo), 'meio': cobertura(y, p_med),
                  'quente': cobertura(y, p_hi),
                  'intervalo': float(np.mean((y >= p_lo) & (y <= p_hi)))}
    pb = {'frio': pinball(y, p_lo, args.q_lo), 'meio': pinball(y, p_med, ALPHA_MEIO),
          'quente': pinball(y, p_hi, args.q_hi)}
    cruzam = float(np.mean((p_lo > p_med) | (p_med > p_hi)))
    larg = float(np.mean(p_hi - p_lo))

    print('\n=== OOF global ===', flush=True)
    for r in RAMOS_Q:
        print(f'  {r:6s} (alvo {alvos_ramo[r]:.2f}): cob={cob_global[r]:.3f}  pinball={pb[r]:.4f}',
              flush=True)
    print(f'  banda p16-p84: cob={cob_global["intervalo"]:.3f} (alvo {alvos["intervalo"]:.2f}) | '
          f'largura media={larg:.3f} °C | cruzamento={cruzam*100:.2f}%', flush=True)

    # ---- estratos (hora / elevacao / costa) — mesmos cortes do F6 ----
    df['hora'] = df['data_hora_utc'].dt.hour
    df['faixa_elev']  = pd.cut(df['z_alvo'], ELEV_BINS, labels=ELEV_LABS)
    df['faixa_costa'] = pd.cut(df['dist_oceano'] / 1000, COSTA_BINS, labels=COSTA_LABS)
    estr_tabs = {}
    for col in ['hora', 'faixa_elev', 'faixa_costa']:
        linhas = []
        for lvl, idx in df.groupby(col, observed=True).groups.items():
            ii = df.index.get_indexer(idx)
            linhas.append(linha_cob(col, lvl, y[ii], p_lo[ii], p_med[ii], p_hi[ii]))
        estr_tabs[col] = pd.DataFrame(linhas)

    # subconjunto vale-alto-noturno (foco: geada — o q16 tem de estar calibrado aqui)
    mask_van = (df['z_alvo'].to_numpy() > VAN_Z_MIN) & \
               df['hora'].between(VAN_HORA[0], VAN_HORA[1]).to_numpy()
    van = linha_cob('vale_alto_noturno',
                    f'z>{VAN_Z_MIN:.0f}m & {VAN_HORA[0]:02d}-{VAN_HORA[1]:02d}UTC',
                    y[mask_van], p_lo[mask_van], p_med[mask_van], p_hi[mask_van])

    print('\n--- cobertura da banda por faixa de elevacao ---', flush=True)
    print(estr_tabs['faixa_elev'].to_string(index=False), flush=True)
    print(f'\nvale-alto-noturno: n={van["n"]:,} | cob_lo={van["cob_lo"]:.3f} '
          f'cob_hi={van["cob_hi"]:.3f} cob_int={van["cob_int"]:.3f}', flush=True)

    # ---- CSV cobertura (global + van + estratificado) ----
    g_row = linha_cob('global', 'all', y, p_lo, p_med, p_hi)
    cob_csv = pd.concat([pd.DataFrame([g_row, van])] + list(estr_tabs.values()),
                        ignore_index=True)
    cob_path = aval / 'cobertura_quantil.csv'
    cob_csv.to_csv(cob_path, sep=';', index=False, float_format='%.4f')

    # ---- figuras ----
    figuras_cv(figs, cob_global, alvos, estr_tabs['faixa_elev'], best_iters)

    # ---- VEREDITO (por cauda; foco global E vale-alto-noturno) ----
    bi_medio = {r: int(round(float(np.mean(best_iters[r])))) for r in RAMOS_Q}

    def veredito_cauda(nome, alpha, cob_g, cob_van):
        # sub-cobre a banda: cauda inferior alta demais (cob>alpha) OU superior baixa (cob<alpha).
        calibr = abs(cob_g - alpha) <= TOL_COB and abs(cob_van - alpha) <= TOL_COB
        if calibr:
            return (f'{nome} (a={alpha:.2f}) CALIBRADO: cob_global={cob_g:.3f} '
                    f'van={cob_van:.3f}. Treinar final com best_iter={bi_medio[nome]}.')
        aperta = (cob_g > alpha) if nome == 'frio' else (cob_g < alpha)
        if aperta:
            return (f'{nome} (a={alpha:.2f}) SUB-COBRE a banda: cob_global={cob_g:.3f} '
                    f'van={cob_van:.3f}. Relaxar cauda: --lambda-l2 menor e/ou --min-child '
                    f'menor; re-rodar --cv.')
        return (f'{nome} (a={alpha:.2f}) SOBRE-COBRE: cob_global={cob_g:.3f} '
                f'van={cob_van:.3f}. Banda larga demais; endurecer regularizacao.')

    vereditos = [
        veredito_cauda('frio', args.q_lo, cob_global['frio'], van['cob_lo']),
        veredito_cauda('quente', args.q_hi, cob_global['quente'], van['cob_hi']),
        f'banda p16-p84: cob_global={cob_global["intervalo"]:.3f} (alvo {alvos["intervalo"]:.2f}); '
        f'vale-alto-noturno={van["cob_int"]:.3f}; largura={larg:.3f} °C.',
    ]
    if cruzam > 0.01:
        vereditos.append(f'ATENCAO: cruzamento de quantis {cruzam*100:.2f}% — ordenar (np.sort) '
                         f'os quantis por pixel no F8.')
    print('\n>>> VEREDITO', flush=True)
    for v in vereditos:
        print(f'  - {v}', flush=True)

    # ---- relatorio markdown ----
    cols_cob = ['estrato', 'faixa', 'n', 'cob_lo', 'cob_med', 'cob_hi', 'cob_int', 'larg']
    linhas = [
        '# Validacao leave-cell-out do treino quantilico (F7q)', '',
        f'- Matriz: `{Path(args.matriz).name}` | features: **{args.features}** ({len(feats)}).',
        f'- Boosters: frio (quantile a={args.q_lo}), meio (a={ALPHA_MEIO}), '
        f'quente (a={args.q_hi}) — TODOS na matriz completa, SEM roteamento; alvo `{ALVO}`.',
        f'- CV: GroupKFold por cell_id, **{args.folds}** folds | carve {int(FRAC_VALID*100)}% das '
        f'celulas p/ early stopping (paciencia {EARLY_CV}, metrica pinball).',
        f'- Regularizacao da cauda: lambda_l2={args.lambda_l2 if args.lambda_l2 is not None else "F7"} '
        f'min_child={args.min_child if args.min_child is not None else "F7"}.',
        f'- OOF cru (3 quantis do residuo) em `{oof_path.name}`.', '',
        '## Global (OOF)', '',
        '| quantil | alvo | cobertura | pinball |', '|---|---|---|---|',
        *[f'| {r} | {alvos_ramo[r]:.2f} | {cob_global[r]:.4f} | {pb[r]:.4f} |' for r in RAMOS_Q],
        f'| banda p16-p84 | {alvos["intervalo"]:.2f} | {cob_global["intervalo"]:.4f} | — |', '',
        f'- largura media da banda: **{larg:.4f} °C** | cruzamento de quantis: **{cruzam*100:.2f}%**.', '',
        '## Cobertura por faixa de elevacao', '',
        df_para_md(estr_tabs['faixa_elev'][cols_cob]), '',
        '## Cobertura por faixa de distancia a costa', '',
        df_para_md(estr_tabs['faixa_costa'][cols_cob]), '',
        '## Cobertura por hora UTC', '',
        df_para_md(estr_tabs['hora'][cols_cob]), '',
        '## Vale-alto-noturno (foco: geada)', '',
        df_para_md(pd.DataFrame([van])[cols_cob]), '',
        '## best_iter por fold (early stopping pinball por ramo)', '',
        '| ramo | best_iters | media |', '|---|---|---|',
        *[f'| {r} | {[int(x) for x in best_iters[r]]} | {bi_medio[r]} |' for r in RAMOS_Q], '',
        f'## Veredito (TOL_COB={TOL_COB})', '',
        *[f'- **{v}**' for v in vereditos], '',
        '## Saidas', '',
        f'- `{cob_path.name}` (global + vale-alto-noturno + estratificado)',
        '- `figuras/fig_cobertura.png`, `figuras/fig_cobertura_elev.png`, `figuras/fig_best_iter.png`',
        f'- OOF: `../{oof_path.name}`', '',
    ]
    rel_path = aval / 'relatorio_quantil.md'
    rel_path.write_text('\n'.join(linhas), encoding='utf-8')

    print(f'\nSalvos em {aval}:', flush=True)
    print(f'  {cob_path.name} | {rel_path.name} | figuras/*.png', flush=True)
    print(f'OOF: {oof_path}', flush=True)


# ============================================================
# MODO --cv --apenas-completo — CV leave-cell-out do L2 central (SUBSTITUI o F6)
# ============================================================

def rodar_cv_completo(args, feats):
    # Porte do F6: OOF do ramo L2 (pred_residuo) + relatorio estratificado. Gera
    # oof_residuo_loco_{n}feat.parquet (drop-in p/ o f8b) + importancia + metricas.
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    df = carregar_matriz(args.matriz, feats, com_estratos=True)
    n_grp = df['cell_id'].nunique()
    if n_grp < args.folds:
        raise ValueError(f'poucas celulas p/ GroupKFold: {n_grp} < folds={args.folds}')
    X = df[feats]
    y32 = df[ALVO].to_numpy('float32')
    y = df[ALVO].to_numpy('float64')
    grp = df['cell_id'].to_numpy()

    print(f'\n=== --cv --apenas-completo leave-cell-out | {args.folds} folds | '
          f'regression_l2 (substitui o F6) ===', flush=True)
    oof = np.full(len(df), np.nan, dtype='float64')
    imp_gain = np.zeros(len(feats)); imp_split = np.zeros(len(feats))
    best_iters, mae_tr, bias_tr, tempos = [], [], [], []
    rng = np.random.default_rng(SEED)
    gkf = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y32, grp), 1):
        t0 = time.time()
        # alpha=None -> L2; EARLY_CV reserva FRAC_VALID das celulas p/ early stopping.
        model, bi = treina_ramo(X, y32, grp, tr, args.num_round, EARLY_CV, args.threads,
                                rng, alpha=None)
        oof[te] = model.predict(X.iloc[te], num_iteration=bi)
        best_iters.append(bi)
        imp_gain  += model.feature_importance('gain')
        imp_split += model.feature_importance('split')
        # treino (fit) in-sample no fold p/ o gap treino-vs-OOF. Sobre `tr` (inclui as
        # celulas de early stop) — diferenca vs i_fit e desprezivel p/ um diagnostico.
        m_tr = metricas(model.predict(X.iloc[tr], num_iteration=bi) - y[tr])
        mae_tr.append(m_tr['mae']); bias_tr.append(m_tr['me'])
        dt = time.time() - t0; tempos.append(dt)
        md, mb, ss = skill(oof[te] - y[te], -y[te])
        print(f'  fold {fold}/{args.folds}: {len(np.unique(grp[te]))} celulas teste | '
              f'best_iter={bi} | RMSE ERA5={mb["rmse"]:.3f} -> down={md["rmse"]:.3f} | '
              f'SS={ss:.4f} | treino MAE={m_tr["mae"]:.3f} BIAS={m_tr["me"]:+.3f} | '
              f'tempo={dt:.1f}s', flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError('OOF incompleto — ha linhas sem previsao.')

    err_down = oof - y
    err_base = -y
    md, mb, ss = skill(err_down, err_base)
    print('\n=== ENSAIO (leave-cell-out) — global ===', flush=True)
    print(f'  baseline ERA5 : MAE={mb["mae"]:.3f} RMSE={mb["rmse"]:.3f} ME={mb["me"]:.3f}', flush=True)
    print(f'  downscaling   : MAE={md["mae"]:.3f} RMSE={md["rmse"]:.3f} ME={md["me"]:.3f}', flush=True)
    print(f'  treino (fit)  : MAE={np.mean(mae_tr):.3f} BIAS={np.mean(bias_tr):+.3f}  (media dos folds)', flush=True)
    print(f'  SKILL SCORE   : SS = {ss:.4f}  (RMSE -{100*(1-md["rmse"]/mb["rmse"]):.1f}%)', flush=True)
    print(f'  TEMPO         : total={sum(tempos):.1f}s | por fold={[round(t, 1) for t in tempos]}', flush=True)

    df = df.reset_index(drop=True)
    df['hora'] = df['data_hora_utc'].dt.hour
    df['mes']  = df['data_hora_utc'].dt.month
    df['faixa_elev']  = pd.cut(df['z_alvo'], ELEV_BINS, labels=ELEV_LABS)
    df['faixa_costa'] = pd.cut(df['dist_oceano'] / 1000, COSTA_BINS, labels=COSTA_LABS)
    estr = {c: estrato(df, c, err_down, err_base) for c in ['hora', 'faixa_elev', 'faixa_costa', 'mes']}
    print('\n--- SS por hora UTC (noite SC ~ 03-09 UTC) ---', flush=True)
    print(estr['hora'].to_string(index=False), flush=True)
    print('\n--- SS por faixa de elevacao ---', flush=True)
    print(estr['faixa_elev'].to_string(index=False), flush=True)
    print('\n--- SS por faixa de distancia a costa ---', flush=True)
    print(estr['faixa_costa'].to_string(index=False), flush=True)

    imp = (pd.DataFrame({'feature': feats, 'gain': imp_gain / args.folds,
                         'split': imp_split / args.folds})
           .sort_values('gain', ascending=False).reset_index(drop=True))
    print('\n--- top 15 features (gain medio) ---', flush=True)
    print(imp.head(15).to_string(index=False), flush=True)

    # --- saidas: OOF drop-in p/ o f8b (--oof) + importancia + metricas ---
    base = f'oof_residuo_loco_{len(feats)}feat'
    oof_path = saida / f'{base}.parquet'
    pd.DataFrame({'cell_id': df['cell_id'], 'data_hora_utc': df['data_hora_utc'],
                  'y_residuo': y, 'pred_residuo': oof}).to_parquet(oof_path, index=False)
    imp.to_csv(saida / f'{base}_importancia.csv', sep=';', index=False, float_format='%.1f')
    rel = {'cv': 'leave-cell-out GroupKFold', 'n_splits': args.folds, 'best_iters': best_iters,
           'conjunto': args.features, 'n_features': len(feats), 'features': list(feats),
           'params': params_ramo(args.threads, None), 'num_round': args.num_round, 'early': EARLY_CV,
           'tempo': {'total_s': round(sum(tempos), 1),
                     'por_fold_s': [round(t, 1) for t in tempos]},
           'global': {'baseline': mb, 'downscaling': md, 'skill_score': ss},
           'treino': {'mae': float(np.mean(mae_tr)), 'bias': float(np.mean(bias_tr))},
           'por_hora': estr['hora'].to_dict('records'),
           'por_elevacao': estr['faixa_elev'].to_dict('records'),
           'por_costa': estr['faixa_costa'].to_dict('records'),
           'por_mes': estr['mes'].to_dict('records')}
    with open(saida / f'{base}_metricas.json', 'w') as f:
        json.dump(rel, f, indent=2, ensure_ascii=False)
    print(f'\nSalvos em {saida}: {base}.parquet (drop-in f8b --oof) | '
          f'{base}_importancia.csv | {base}_metricas.json', flush=True)


# ============================================================
# MODO PADRAO (sem --cv) — treino final dos 4 boosters (matriz COMPLETA)
# ============================================================

def rodar_final(args, feats):
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    df = carregar_matriz(args.matriz, feats, com_estratos=False)
    X = df[feats]
    y32 = df[ALVO].to_numpy('float32')
    y = df[ALVO].to_numpy('float64')
    t2m = df[BASELINE_VAR].to_numpy('float64')
    grp = df['cell_id'].to_numpy()

    t_lo, t_hi = limiares(t2m, args.q_lo, args.q_hi)   # informativo (meta/compat F8)
    # --apenas-completo: SO o ramo p0-100 (L2 central -> t2m). Produto reduzido rapido
    # (1 booster no treino E na inferencia), sem a banda quantilica.
    ramos_treino = (RAMO_COMPLETO,) if args.apenas_completo else RAMOS_V2
    saida_vars = {r: nomes_saida(args.q_lo, args.q_hi)[r] for r in ramos_treino}
    idx_full = np.arange(len(df))
    modo = 'so p0-100 (L2)' if args.apenas_completo else 'QUANTILICO'
    print(f'\nTreino {modo} na matriz completa ({len(df):,} linhas). '
          f'Temperaturas dos quantis de {BASELINE_VAR} (informativo): '
          f'q{int(args.q_lo*100)}={t_lo - KELVIN:.2f}°C q{int(args.q_hi*100)}={t_hi - KELVIN:.2f}°C',
          flush=True)
    for ramo in ramos_treino:
        a = alpha_ramo(ramo, args.q_lo, args.q_hi)
        obj = f'quantile a={a:.2f}' if a is not None else 'regression_l2 (media)'
        print(f'  {ramo:9s} -> {saida_vars[ramo]:12s} | {obj}', flush=True)

    rng = np.random.default_rng(SEED)
    preds = {}          # p/ diagnostico in-sample da banda (cobertura/cruzamento)
    info_ramos = {}
    for ramo in ramos_treino:
        a = alpha_ramo(ramo, args.q_lo, args.q_hi)
        lam, mc = (args.lambda_l2, args.min_child) if a is not None else (None, None)
        t0 = time.time()
        print(f'\n[{ramo}] treinando ({idx_full.size:,} linhas | early={args.early}) ...', flush=True)
        model, best_it = treina_ramo(X, y32, grp, idx_full, args.num_round, args.early,
                                     args.threads, rng, a, lam, mc)
        dt = time.time() - t0
        pred = model.predict(X, num_iteration=best_it)
        preds[ramo] = pred

        # diagnostico in-sample (sanity so): cobertura/pinball p/ quantil; MAE/RMSE p/ L2.
        if a is not None:
            cob, pb = cobertura(y, pred), pinball(y, pred, a)
            diag = {'alpha': a, 'cobertura_in': cob, 'pinball_in': pb}
            print(f'[{ramo}] concluido em {dt/60:.1f} min | best_iter={best_it} | '
                  f'in-sample cob={cob:.3f} (alvo {a:.2f}) pinball={pb:.4f}', flush=True)
        else:
            m_tr = metricas(pred - y)
            diag = {'alpha': None, 'in_sample': m_tr}
            print(f'[{ramo}] concluido em {dt/60:.1f} min | best_iter={best_it} | '
                  f'in-sample MAE={m_tr["mae"]:.3f} RMSE={m_tr["rmse"]:.3f} '
                  f'BIAS={m_tr["me"]:+.3f}', flush=True)

        f_modelo = saida / f'{args.nome}_{ramo}.txt'
        model.save_model(str(f_modelo), num_iteration=best_it)
        imp = (pd.DataFrame({'feature': feats,
                             'gain': model.feature_importance('gain'),
                             'split': model.feature_importance('split')})
               .sort_values('gain', ascending=False).reset_index(drop=True))
        imp.to_csv(saida / f'{args.nome}_{ramo}_importancia.csv',
                   sep=';', index=False, float_format='%.1f')
        info_ramos[ramo] = {
            'modelo': f_modelo.name, 'saida_var': saida_vars[ramo],
            'objective': 'quantile' if a is not None else 'regression_l2',
            'alpha': a, 'best_iteration': best_it, 'n_train': int(idx_full.size),
            'diagnostico_in_sample': diag, 'tempo_min': round(dt / 60, 1),
        }

    # banda in-sample (sanity): so faz sentido com os dois ramos de cauda treinados.
    if {'frio', 'quente'} <= set(preds):
        p_lo, p_hi = preds['frio'], preds['quente']
        cob_int = float(np.mean((y >= p_lo) & (y <= p_hi)))
        cruzam = float(np.mean(p_lo > p_hi))
        print(f'\nbanda in-sample [p16,p84]: cobertura={cob_int:.3f} (alvo {args.q_hi-args.q_lo:.2f}) | '
              f'cruzamento={cruzam*100:.2f}% (in-sample e otimista; ver --cv)', flush=True)

    # ========================================================
    # META — CONTRATO com o F8 (mesmo tipo/saida_vars; F8/oper/f10-f14 inalterados)
    # ========================================================

    meta = {
        'criado_em': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'tipo': 'segmentado_quantil_v2',   # mantido p/ compat F8/oper; metodo real abaixo
        'metodo': 'p0_100_L2' if args.apenas_completo else 'regressao_quantilica_v2',
        'features': feats,
        'n_features': len(feats),
        'conjunto_features': args.features,
        'alvo': ALVO,
        'ramos': info_ramos,           # {'frio': {'modelo':..., 'saida_var':'t2m_p16', 'alpha':0.16, ...}, ...}
        'saida_vars': saida_vars,      # ramo -> nome da variavel NetCDF no F8
        'quantis': {'frio': args.q_lo, 'meio': ALPHA_MEIO, 'quente': args.q_hi, RAMO_COMPLETO: None},
        'corte': {
            # v2 SEM roteamento: t_lo_K/t_hi_K sao INFORMATIVOS (temperaturas dos quantis de
            # t2m_1km). Persistidos porque o F8 le corte.t_lo_K/t_hi_K/q_lo/q_hi.
            'var': BASELINE_VAR, 'q_lo': args.q_lo, 'q_hi': args.q_hi,
            't_lo_K': round(t_lo, 4), 't_hi_K': round(t_hi, 4),
            't_lo_C': round(t_lo - KELVIN, 4), 't_hi_C': round(t_hi - KELVIN, 4),
            'regra': ('QUANTILICO na matriz completa (sem roteamento): frio=quantile alpha=q_lo, '
                      'meio=quantile alpha=0.50, quente=quantile alpha=q_hi, completo=regression_l2 '
                      '(media). t_lo_K/t_hi_K sao so referencia (quantis de t2m_1km).'),
        },
        'baseline': {
            'var': BASELINE_VAR, 'offset_celsius': -273.15,
            'formula': 'T_ERA5 = t2m_1km - 273.15 ; T_q = T_ERA5 + pred_residuo',
        },
        'roteamento_no_f8': ('v2 SEM roteamento: o F8 aplica os 4 boosters a TODOS os pixels de '
                             'SC e grava t2m_p16 (q_lo), t2m_p16a84 (mediana), t2m_p84 (q_hi) e '
                             't2m (media L2). A banda [t2m_p16, t2m_p84] e um intervalo de ~68%.'),
        'monotonicidade': ('quantis independentes podem cruzar; se o cruzamento do --cv for '
                           'relevante, aplicar np.sort por pixel no F8 (t2m_p16<=t2m_p16a84<=t2m_p84).'),
        'params': params_ramo(args.threads, None if args.apenas_completo else args.q_lo,
                              args.lambda_l2, args.min_child),
        'lapse_rate': LAPSE_RATE,
        'num_round': args.num_round,
        'early_stopping': args.early,
        'n_train': int(len(df)),
        'n_celulas': int(df['cell_id'].nunique()),
        'matriz': str(args.matriz),
    }
    f_meta = saida / f'{args.nome}_meta.json'
    f_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nSalvos em {saida}:', flush=True)
    for ramo, info in info_ramos.items():
        print(f'  {info["modelo"]} | {args.nome}_{ramo}_importancia.csv | '
              f'best_iter={info["best_iteration"]}', flush=True)
    print(f'  {f_meta.name} (contrato com o F8)', flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description='Treino quantilico do residuo (F7q): boosters p16/p50/p84 + L2 central.')
    ap.add_argument('--features', choices=['28', '33', '53', '55'], default='55',
                    help='conjunto de features (default 55 = vencedor da ablacao, propaga ssr/str)')
    ap.add_argument('--q-lo', type=float, default=0.16, dest='q_lo',
                    help='alpha do quantil inferior (default 0.16 = banda p16 de producao)')
    ap.add_argument('--q-hi', type=float, default=0.84, dest='q_hi',
                    help='alpha do quantil superior (default 0.84 = banda p84 de producao)')
    ap.add_argument('--num-round', type=int, default=F7_NUM_ROUND,
                    help='teto de rounds (pinball converge diferente do L2; ver --cv)')
    ap.add_argument('--early', type=int, default=0,
                    help='paciencia early stopping no treino final (0=off; reserva 20%% das celulas)')
    ap.add_argument('--lambda-l2', type=float, default=None, dest='lambda_l2',
                    help='lambda_l2 SO dos ramos quantilicos (default: herda do F7=5; menor = '
                         'cauda mais larga)')
    ap.add_argument('--min-child', type=int, default=None, dest='min_child',
                    help='min_child_samples SO dos ramos quantilicos (default: herda F7=500)')
    ap.add_argument('--apenas-completo', action='store_true', dest='apenas_completo',
                    help='SO o ramo L2 central (-> t2m). Sem --cv: treino final reduzido (1 booster). '
                         'Com --cv: CV leave-cell-out do L2, SUBSTITUI o F6 (gera oof_residuo_loco).')
    ap.add_argument('--cv', action='store_true',
                    help='modo validacao leave-cell-out (NAO persiste modelo). Sem --apenas-completo: '
                         'cobertura da banda quantilica; com: SS do L2 central (porte do F6)')
    ap.add_argument('--folds', type=int, default=5, help='folds do GroupKFold por cell_id (--cv)')
    ap.add_argument('--threads', type=int, default=int(F7_PARAMS['num_threads']),
                    help='num_threads LightGBM (NUMA: nunca 0 nesta VM, ~43x mais lento)')
    ap.add_argument('--matriz', default=str(MATRIZ))
    ap.add_argument('--saida',  default=str(SAIDA))
    ap.add_argument('--nome',   default=None,
                    help='prefixo dos modelos (default: modelo_quantil_qXX_qYY, dos quantis)')
    args = ap.parse_args()

    feats = CONJUNTOS[args.features]
    if not (0.0 < args.q_lo < ALPHA_MEIO < args.q_hi < 1.0):
        raise ValueError(f'quantis invalidos: exige 0 < q_lo < 0.5 < q_hi < 1 '
                         f'(recebi {args.q_lo}/{args.q_hi})')
    if args.nome is None:            # auto: modelo_p0100 (reduzido) ou modelo_quantil_qXX_qYY (quantis)
        args.nome = 'modelo_p0100' if args.apenas_completo else \
            f'modelo_quantil_q{int(round(args.q_lo*100)):02d}_q{int(round(args.q_hi*100)):02d}'

    print(f'F7q: modo={"CV" if args.cv else "FINAL"} | features={args.features} ({len(feats)}) | '
          f'quantile alpha {args.q_lo}/{ALPHA_MEIO}/{args.q_hi} + L2 central | '
          f'threads={args.threads}', flush=True)

    if args.cv and args.apenas_completo:
        rodar_cv_completo(args, feats)      # porte do F6: CV do L2 central
    elif args.cv:
        rodar_cv(args, feats)
    else:
        rodar_final(args, feats)


if __name__ == '__main__':
    main()
