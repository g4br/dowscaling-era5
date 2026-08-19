#!/usr/bin/env python3

# ============================================================
# F6b — Avaliacao do ensaio LightGBM (le as saidas do F6).
# Entradas (saidas/modelos): oof_ensaio.parquet, importancia_ensaio.csv,
#   metricas_ensaio.json.
# Tudo SEMPRE vs baseline ERA5: err_down = pred_residuo - y_residuo
#   (= T_final - obs); err_base = -y_residuo (= T_ERA5 - obs).
#   SS = 1 - RMSE_down / RMSE_ERA5 (mesma definicao do F6).
# Recalcula a partir do OOF (verdade out-of-fold, sem vazamento) as metricas
# globais e estratificadas (hora, mes, celula) e cruza com metricas_ensaio.json.
# Gera tabelas (csv), figuras (png) e um resumo em markdown.
# NaN nas predicoes/alvo sao ignorados nas metricas.
# Uso: python f6b_avalia_ensaio.py  [--modelos DIR] [--oof FILE]
# Rodar no env com pyarrow/matplotlib (py312).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _root import ROOT, DIR_GRID


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ        = Path(ROOT)
PASTA_MODEL = RAIZ / 'saidas' / 'modelos'
GRID_NPZ    = DIR_GRID / 'centros_wgs84.npz'

TOP_FEATURES = None    # nº de features no grafico de importancia (None = todas)


# ============================================================
# FUNCOES — metricas (identicas ao F6)
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


def df_para_md(df):
    # tabela markdown sem depender de 'tabulate'
    cols = list(df.columns)
    linhas = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        linhas.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
    return '\n'.join(linhas)


def estrato(df, col_grupo, err_down, err_base):
    # SS por grupo (mesma logica do F6, mas vetorizada por groupby de indices).
    linhas = []
    for g, idx in df.groupby(col_grupo, observed=True).groups.items():
        ii = df.index.get_indexer(idx)
        md, mb, ss = skill(err_down[ii], err_base[ii])
        linhas.append({col_grupo: g, 'n': md['n'],
                       'rmse_era5': round(mb['rmse'], 3), 'rmse_down': round(md['rmse'], 3),
                       'mae_era5': round(mb['mae'], 3), 'mae_down': round(md['mae'], 3),
                       'me_down': round(md['me'], 3), 'SS': round(ss, 4)})
    return pd.DataFrame(linhas).sort_values(col_grupo).reset_index(drop=True)


# ============================================================
# LEITURA
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Avaliacao do ensaio LightGBM (F6).')
    ap.add_argument('--modelos', default=str(PASTA_MODEL),
                    help='diretorio com oof_ensaio.parquet / importancia / metricas')
    ap.add_argument('--oof', default=None, help='caminho do parquet OOF (sobrepoe --modelos/--tag)')
    ap.add_argument('--tag', default='', help='sufixo das entradas/saidas (casa com o --tag do F6): '
                    'le oof_ensaio{tag}.parquet etc e grava em avaliacao{tag}/')
    args = ap.parse_args()

    pasta = Path(args.modelos)
    tag = args.tag
    oof_path = Path(args.oof) if args.oof else pasta / f'oof_ensaio{tag}.parquet'
    imp_path = pasta / f'importancia_ensaio{tag}.csv'
    met_path = pasta / f'metricas_ensaio{tag}.json'
    saida    = pasta / (f'avaliacao{tag}' if tag else 'avaliacao')
    figs     = saida / 'figuras'
    figs.mkdir(parents=True, exist_ok=True)

    print(f'Lendo OOF: {oof_path}', flush=True)
    df = pd.read_parquet(oof_path)
    df = df.reset_index(drop=True)
    n0 = len(df)
    df = df[np.isfinite(df['y_residuo']) & np.isfinite(df['pred_residuo'])].copy().reset_index(drop=True)
    print(f'  {n0:,} linhas | {n0 - len(df):,} descartadas (NaN) | {len(df):,} usadas', flush=True)

    met_json = json.loads(met_path.read_text()) if met_path.exists() else {}
    imp = pd.read_csv(imp_path, sep=';') if imp_path.exists() else None

    # erros vs baseline ERA5
    err_down = (df['pred_residuo'] - df['y_residuo']).values.astype('float64')   # T_final - obs
    err_base = (-df['y_residuo']).values.astype('float64')                       # T_ERA5  - obs

    # eixos de estratificacao
    dh = pd.to_datetime(df['data_hora_utc'])
    df['hora'] = dh.dt.hour
    df['mes']  = dh.dt.month

    # ========================================================
    # GLOBAL
    # ========================================================

    md, mb, ss = skill(err_down, err_base)
    r = np.corrcoef(df['pred_residuo'], df['y_residuo'])[0, 1]
    print('\n=== GLOBAL (out-of-fold, vs baseline ERA5) ===', flush=True)
    print(f'  baseline ERA5 : MAE={mb["mae"]:.3f}  RMSE={mb["rmse"]:.3f}  ME={mb["me"]:+.3f}', flush=True)
    print(f'  downscaling   : MAE={md["mae"]:.3f}  RMSE={md["rmse"]:.3f}  ME={md["me"]:+.3f}', flush=True)
    print(f'  SKILL SCORE   : SS={ss:.4f}  (RMSE -{100*(1-md["rmse"]/mb["rmse"]):.1f}%  |  '
          f'MAE -{100*(1-md["mae"]/mb["mae"]):.1f}%)', flush=True)
    print(f'  corr(pred_residuo, y_residuo) = {r:.4f}  (R2={r**2:.4f})', flush=True)
    if met_json.get('global'):
        gj = met_json['global']
        print(f'  [json F6] SS={gj["skill_score"]:.4f}  RMSE_down={gj["downscaling"]["rmse"]:.3f}  '
              f'RMSE_era5={gj["baseline"]["rmse"]:.3f}  (deve bater com o acima)', flush=True)

    # ========================================================
    # ESTRATIFICADO: hora, mes, celula
    # ========================================================

    est_hora = estrato(df, 'hora', err_down, err_base)
    est_mes  = estrato(df, 'mes',  err_down, err_base)

    # por celula: cell_id = "iy_ix" -> coords (lon/lat) p/ mapa
    est_cel = estrato(df, 'cell_id', err_down, err_base)
    iyix = est_cel['cell_id'].str.split('_', expand=True).astype(int)
    est_cel['iy'] = iyix[0]; est_cel['ix'] = iyix[1]
    g = np.load(GRID_NPZ)
    est_cel['lon'] = g['lon_centros'][est_cel['iy'].values, est_cel['ix'].values]
    est_cel['lat'] = g['lat_centros'][est_cel['iy'].values, est_cel['ix'].values]
    frac_pos = float((est_cel['SS'] > 0).mean())

    print('\n--- SS por hora UTC (madrugada SC ~ 03-09 UTC) ---', flush=True)
    print(est_hora[['hora', 'n', 'rmse_era5', 'rmse_down', 'mae_down', 'SS']].to_string(index=False), flush=True)
    print('\n--- SS por mes ---', flush=True)
    print(est_mes[['mes', 'n', 'rmse_era5', 'rmse_down', 'mae_down', 'SS']].to_string(index=False), flush=True)
    print(f'\n--- por celula: {len(est_cel)} celulas | SS>0 em {100*frac_pos:.1f}% | '
          f'SS mediano={est_cel["SS"].median():.4f} ---', flush=True)

    est_hora.to_csv(saida / 'metricas_por_hora.csv', sep=';', index=False, float_format='%.4f')
    est_mes.to_csv(saida / 'metricas_por_mes.csv', sep=';', index=False, float_format='%.4f')
    est_cel.to_csv(saida / 'metricas_por_celula.csv', sep=';', index=False, float_format='%.4f')

    # ========================================================
    # FIGURAS
    # ========================================================

    print('\nGerando figuras...', flush=True)

    # 1) SS por hora
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(est_hora['hora'], est_hora['SS'], color='#2c7fb8')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='hora UTC', ylabel='Skill Score', title=f'SS por hora UTC (global={ss:.3f})')
    ax.set_xticks(range(0, 24, 2)); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / 'fig_ss_por_hora.png', dpi=130); plt.close(fig)

    # 2) RMSE por hora (ERA5 vs downscaling)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(est_hora['hora'], est_hora['rmse_era5'], 'o-', label='ERA5 (baseline)', color='#d95f02')
    ax.plot(est_hora['hora'], est_hora['rmse_down'], 's-', label='downscaling', color='#1b9e77')
    ax.set(xlabel='hora UTC', ylabel='RMSE (degC)', title='RMSE por hora UTC')
    ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(figs / 'fig_rmse_por_hora.png', dpi=130); plt.close(fig)

    # 3) SS por mes
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(est_mes['mes'], est_mes['SS'], color='#7570b3')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='mes', ylabel='Skill Score', title='SS por mes')
    ax.set_xticks(range(1, 13)); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / 'fig_ss_por_mes.png', dpi=130); plt.close(fig)

    # 4) histograma dos erros (down vs baseline)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = np.nanpercentile(np.abs(np.concatenate([err_down, err_base])), 99.5)
    bins = np.linspace(-lim, lim, 120)
    ax.hist(err_base, bins=bins, alpha=0.5, label=f'ERA5 (RMSE={mb["rmse"]:.2f})', color='#d95f02', density=True)
    ax.hist(err_down, bins=bins, alpha=0.5, label=f'downscaling (RMSE={md["rmse"]:.2f})', color='#1b9e77', density=True)
    ax.axvline(0, color='k', lw=0.8)
    ax.set(xlabel='erro = previsto - observado (degC)', ylabel='densidade', title='Distribuicao dos erros')
    ax.legend(); fig.tight_layout(); fig.savefig(figs / 'fig_hist_erros.png', dpi=130); plt.close(fig)

    # 5) hexbin pred_residuo vs y_residuo (1:1)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    lo = float(np.nanpercentile(df['y_residuo'], 0.5)); hi = float(np.nanpercentile(df['y_residuo'], 99.5))
    hb = ax.hexbin(df['y_residuo'], df['pred_residuo'], gridsize=80, bins='log',
                   extent=(lo, hi, lo, hi), cmap='viridis', mincnt=1)
    ax.plot([lo, hi], [lo, hi], 'r-', lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='residuo observado (obs - ERA5)',
           ylabel='residuo previsto', title=f'Residuo previsto vs observado (R2={r**2:.3f})')
    fig.colorbar(hb, ax=ax, label='log10(contagem)')
    fig.tight_layout(); fig.savefig(figs / 'fig_scatter_residuo.png', dpi=130); plt.close(fig)

    # 6) importancia (gain)
    if imp is not None:
        top = imp.sort_values('gain', ascending=False)
        if TOP_FEATURES:
            top = top.head(TOP_FEATURES)
        top = top.iloc[::-1]
        titulo = f'Importancia — top {len(top)} (gain)' if TOP_FEATURES else \
                 f'Importancia — todas as {len(top)} features (gain)'
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.2))
        bars = ax.barh(top['feature'], top['gain'], color='#41ab5d')
        ax.bar_label(bars, labels=[f'{v/1e6:.1f}M' for v in top['gain']],
                     padding=2, fontsize=7)
        ax.margins(x=0.10)   # espaco a direita p/ as labels
        ax.set(xlabel='gain medio (folds)', title=titulo)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout(); fig.savefig(figs / 'fig_importancia.png', dpi=130); plt.close(fig)

    # 7) mapa de SS por celula
    for col, cmap, vlim, nome, titulo in [
        ('SS',        'RdYlGn', (-0.3, 0.3), 'fig_mapa_ss_celula.png',   'SS por celula (estacao)'),
        ('rmse_down', 'viridis', None,       'fig_mapa_rmse_celula.png', 'RMSE downscaling por celula (degC)')]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        kw = dict(c=est_cel[col], cmap=cmap, s=28, edgecolor='k', linewidth=0.2)
        if vlim: kw.update(vmin=vlim[0], vmax=vlim[1])
        sc = ax.scatter(est_cel['lon'], est_cel['lat'], **kw)
        ax.set(xlabel='lon', ylabel='lat', title=titulo); ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label=col)
        fig.tight_layout(); fig.savefig(figs / nome, dpi=130); plt.close(fig)

    # ========================================================
    # RESUMO MARKDOWN
    # ========================================================

    linhas = [
        '# Avaliacao do ensaio LightGBM (F6)', '',
        f'- OOF: `{oof_path.name}` | linhas usadas: **{len(df):,}** | celulas: **{est_cel.shape[0]}**',
        f'- Periodo: {dh.min()} -> {dh.max()}', '',
        '## Global (out-of-fold, vs baseline ERA5)', '',
        '| metrica | ERA5 (baseline) | downscaling | ganho |',
        '|---|---|---|---|',
        f'| MAE  (degC) | {mb["mae"]:.3f} | {md["mae"]:.3f} | -{100*(1-md["mae"]/mb["mae"]):.1f}% |',
        f'| RMSE (degC) | {mb["rmse"]:.3f} | {md["rmse"]:.3f} | -{100*(1-md["rmse"]/mb["rmse"]):.1f}% |',
        f'| ME   (degC) | {mb["me"]:+.3f} | {md["me"]:+.3f} | — |',
        '',
        f'- **Skill Score global: {ss:.4f}**',
        f'- corr(residuo previsto, observado) = {r:.4f} (R2={r**2:.4f})',
        f'- celulas com SS>0 (modelo bate a baseline): **{100*frac_pos:.1f}%** ({int((est_cel["SS"]>0).sum())}/{len(est_cel)})',
        f'- SS por celula: mediano={est_cel["SS"].median():.4f}, p10={est_cel["SS"].quantile(.1):.4f}, p90={est_cel["SS"].quantile(.9):.4f}',
        '',
        '## Por hora UTC', '', df_para_md(est_hora[['hora','n','rmse_era5','rmse_down','mae_down','SS']]),
        '', '## Por mes', '', df_para_md(est_mes[['mes','n','rmse_era5','rmse_down','mae_down','SS']]),
        '', '## Figuras (em ./figuras/)', '',
        '- `fig_ss_por_hora.png`, `fig_rmse_por_hora.png`, `fig_ss_por_mes.png`',
        '- `fig_hist_erros.png`, `fig_scatter_residuo.png`, `fig_importancia.png`',
        '- `fig_mapa_ss_celula.png`, `fig_mapa_rmse_celula.png`', '',
        '## Tabelas', '',
        '- `metricas_por_hora.csv`, `metricas_por_mes.csv`, `metricas_por_celula.csv`', '',
    ]
    (saida / 'resumo_avaliacao.md').write_text('\n'.join(linhas), encoding='utf-8')

    print(f'\nSalvo em {saida}:', flush=True)
    print('  resumo_avaliacao.md | metricas_por_{hora,mes,celula}.csv | figuras/*.png', flush=True)


if __name__ == '__main__':
    main()
