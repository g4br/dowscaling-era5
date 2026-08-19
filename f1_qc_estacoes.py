#!/usr/bin/env python3

# ============================================================
# F1 — QC e harmonizacao das estacoes
# Alvo: T_obs limpo (residuo = T_obs - T_ERA5_bilinear).
# Dado real: sep ',', colunas dt_regis/md_valor/md_final (hora LOCAL UTC-3).
# md_final: SO 4 e 5 sao validos; qualquer outra flag (9, sentinelas, suspeitos) -> descartar.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import argparse
import datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from _root import ROOT


# ============================================================
# CONFIGURACOES
# ============================================================

# STN_DIR sobrepoe a pasta dos CSVs (mesmo layout <id>_<yyyymm>.csv). Usado p/ rodar o
# MESMO QC nas estacoes da aba Exclusao (stn_data_exclusao/192) -> parquet separado via --saida.
PASTA_ESTACOES = Path(os.environ.get('STN_DIR') or ROOT + '/stn_data/192')
CAMINHO_META   = Path(ROOT + '/stn_data/localidades_estacoes.csv')
PASTA_SAIDA    = Path(ROOT + '/stn_data')
# SUF sufixa as saidas auxiliares (mesma ideia do f3): com STN_DIR=stn_data_exclusao/192
# use SUF=_exclusao p/ NAO sobrescrever aprovadas/sem_metadados da rodada principal.
SUF            = os.environ.get('SUF', '')

SEP_CSV  = ','
COL_DATA = 'dt_regis'
COL_TEMP = 'md_valor'
COL_FLAG = 'md_final'

META_KEY  = 'Estacao'
META_ALT  = 'Altitude'
META_LON  = 'Longitude'
META_LAT  = 'Latitude'

FUSO_LOCAL_H      = -3        # SC = UTC-3; dt_regis e LOCAL (confirmado por ciclo diurno)
FLAGS_VALIDAS     = (4, 5)    # so md_final in (4,5) e valido; resto -> descartar
TEMP_MIN_FIS      = -15.0
TEMP_MAX_FIS      = 45.0
SALTO_MAX_H       = 15.0
HORAS_TRAVADO     = 6
MIN_HORAS_VALIDAS = 24 * 365  # ~1 ano


# ============================================================
# FUNCOES
# ============================================================

def aplicar_qc_temperatura(df_estacao):
    # Retorna (df, contagem) onde contagem atribui cada remocao a PRIMEIRA etapa que a derruba.
    temp = pd.to_numeric(df_estacao[COL_TEMP], errors='coerce').copy()
    contagem = {}

    # 0) temperatura originalmente ausente / nao-numerica (nao contabilizada nas etapas seguintes)
    contagem['t_ausente'] = int(temp.isna().sum())
    antes = temp.notna()

    # 1) flag do provedor: mantem so md_final in (4,5); resto -> NaN (remove 9/sentinelas/suspeitos)
    flag = pd.to_numeric(df_estacao[COL_FLAG], errors='coerce')
    temp[~flag.isin(FLAGS_VALIDAS)] = np.nan
    agora = temp.notna()
    contagem['flag_invalida'] = int((antes & ~agora).sum())
    antes = agora

    # 2) faixa fisica plausivel para SC (backstop)
    temp[(temp < TEMP_MIN_FIS) | (temp > TEMP_MAX_FIS)] = np.nan
    agora = temp.notna()
    contagem['fora_faixa'] = int((antes & ~agora).sum())
    antes = agora

    # 3) spikes horarios
    salto = temp.diff().abs()
    temp[salto > SALTO_MAX_H] = np.nan
    agora = temp.notna()
    contagem['spike'] = int((antes & ~agora).sum())
    antes = agora

    # 4) valores travados (>= HORAS_TRAVADO h identicas)
    travado = (temp.diff() == 0).rolling(HORAS_TRAVADO).sum() >= HORAS_TRAVADO
    temp[travado] = np.nan
    agora = temp.notna()
    contagem['travado'] = int((antes & ~agora).sum())

    contagem['sobrevive'] = int(temp.notna().sum())
    df_estacao['temperatura_c_qc'] = temp
    return df_estacao, contagem


def id_da_estacao(caminho_csv):
    # <id>_<yyyymm>.csv -> id
    return caminho_csv.stem.rsplit('_', 1)[0]


def janela_mask(utc, ini, fim):
    # utc: Series datetime64 (UTC). ini inclusivo, fim EXCLUSIVO; None = sem limite.
    sel = pd.Series(True, index=utc.index)
    if ini is not None:
        sel &= utc >= ini
    if fim is not None:
        sel &= utc < fim
    return sel


def merge_janela(antigo, novo, ini, fim):
    # Substitui SO a janela [ini, fim) no parquet antigo pelas linhas novas; preserva o resto.
    fora = ~janela_mask(antigo['data_hora_utc'], ini, fim)
    return pd.concat([antigo[fora], novo], ignore_index=True)


def _self_check():
    idx = pd.date_range('2026-01-01', periods=5, freq='D')          # 01..05 -> v=0..4
    utc = pd.Series(idx)
    ini, fim = pd.Timestamp('2026-01-02'), pd.Timestamp('2026-01-04')   # [02,04) -> 02,03
    assert janela_mask(utc, ini, fim).tolist() == [False, True, True, False, False]
    antigo = pd.DataFrame({'data_hora_utc': idx, 'v': range(5)})
    novo   = pd.DataFrame({'data_hora_utc': [pd.Timestamp('2026-01-02 12:00')], 'v': [99]})
    m = merge_janela(antigo, novo, ini, fim).sort_values('data_hora_utc')
    assert m['v'].tolist() == [0, 99, 3, 4]      # 01,novo,04,05 ; 02/03 antigos trocados


# ============================================================
# ARGUMENTOS
# ============================================================

ap = argparse.ArgumentParser(description='F1 — QC e harmonizacao das estacoes.')
ap.add_argument('--init', default=None,
                help='data inicial UTC YYYY-MM-DD (inclusive): filtra data_hora_utc e faz MERGE '
                     'so nesta janela do parquet (preserva os outros periodos)')
ap.add_argument('--end', default=None, help='data final UTC YYYY-MM-DD (inclusive)')
ap.add_argument('--min-horas', type=int, default=None,
                help='cobertura min. de horas validas/estacao; default 24*365 (run completo) '
                     'ou 0 quando ha --init/--end (janela curta nao atinge 1 ano)')
ap.add_argument('--saida', default=str(PASTA_SAIDA / 'estacoes_qc.parquet'), help='parquet de saida')
ap.add_argument('--self-check', action='store_true', help='roda asserts internos e sai (nao le disco)')
args = ap.parse_args()

if args.self_check:
    _self_check()
    print('self-check OK'); sys.exit(0)

INI_UTC   = pd.Timestamp(args.init) if args.init else None
FIM_UTC   = pd.Timestamp(args.end) + pd.Timedelta(days=1) if args.end else None  # inclusivo -> exclusivo
JANELA    = INI_UTC is not None or FIM_UTC is not None
MIN_HORAS = args.min_horas if args.min_horas is not None else (0 if JANELA else MIN_HORAS_VALIDAS)
ARQ_SAIDA = Path(args.saida)


# ============================================================
# CAMINHOS E CHECAGENS
# ============================================================

arquivos_csv = sorted(PASTA_ESTACOES.glob('*.csv'))
if len(arquivos_csv) == 0:
    raise FileNotFoundError('Nenhum CSV de estacao em: ' + str(PASTA_ESTACOES))

if not CAMINHO_META.exists():
    raise FileNotFoundError('Metadados ausentes: ' + str(CAMINHO_META))


# ============================================================
# LEITURA E AGRUPAMENTO POR ESTACAO
# ============================================================

print('Lendo e agrupando', len(arquivos_csv), 'CSVs por estacao...')

por_estacao = {}
for caminho_csv in arquivos_csv:
    df_mes = pd.read_csv(caminho_csv, sep=SEP_CSV)
    for col in (COL_DATA, COL_TEMP, COL_FLAG):
        if col not in df_mes.columns:
            raise KeyError('Coluna "' + col + '" ausente em ' + str(caminho_csv) +
                           ' (cols: ' + str(list(df_mes.columns)) + ')')
    df_mes[COL_DATA] = pd.to_datetime(df_mes[COL_DATA])
    por_estacao.setdefault(id_da_estacao(caminho_csv), []).append(df_mes)

print('Estacoes distintas (por id de arquivo):', len(por_estacao))


# ============================================================
# PROCESSAMENTO — QC E FILTRO DE COBERTURA
# ============================================================

print('Aplicando QC e filtro de cobertura (>=', MIN_HORAS, 'h validas)...'
      + (f' | janela UTC [{INI_UTC}, {FIM_UTC})' if JANELA else ''))

lista_df  = []
aprovadas = []
descartadas_cobertura = []
contagem_flags   = Counter()   # auditoria: linhas por valor de md_final (todas as estacoes lidas)
contagem_validas = Counter()   # auditoria: linhas por flag que SOBREVIVEM ao QC completo (temperatura_c_qc nao-NaN)
contagem_etapas  = Counter()   # auditoria: remocoes por etapa do QC (atribuidas a 1a etapa que derruba)
for codigo, partes in por_estacao.items():
    df_estacao = pd.concat(partes, ignore_index=True).sort_values(COL_DATA).reset_index(drop=True)
    df_estacao, contagem_qc = aplicar_qc_temperatura(df_estacao)   # QC na serie CHEIA (continuidade)
    contagem_etapas.update(contagem_qc)
    df_estacao['codigo_estacao'] = codigo

    if JANELA:   # recorta a janela SO depois do QC (spike/travado usam vizinhos)
        utc_est = df_estacao[COL_DATA] - dt.timedelta(hours=FUSO_LOCAL_H)
        df_estacao = df_estacao[janela_mask(utc_est, INI_UTC, FIM_UTC)].reset_index(drop=True)

    flag_vals = pd.to_numeric(df_estacao[COL_FLAG], errors='coerce')
    sobrevive = df_estacao['temperatura_c_qc'].notna()
    for val, cnt in flag_vals.value_counts(dropna=False).items():
        chave = 'NaN/nao-numerico' if pd.isna(val) else int(val)
        contagem_flags[chave] += int(cnt)
    for val, cnt in flag_vals[sobrevive].value_counts(dropna=False).items():
        chave = 'NaN/nao-numerico' if pd.isna(val) else int(val)
        contagem_validas[chave] += int(cnt)

    n_validas = int(df_estacao['temperatura_c_qc'].notna().sum())
    if n_validas < MIN_HORAS:
        descartadas_cobertura.append((codigo, n_validas))
        continue

    lista_df.append(df_estacao)
    aprovadas.append(codigo)

print('Descartadas por cobertura:', len(descartadas_cobertura))
if not lista_df:
    sys.exit('Nenhuma estacao sobreviveu ao QC/cobertura no periodo — nada a salvar.')
df_obs = pd.concat(lista_df, ignore_index=True)

# UTC para casar com ERA5-Land (LOCAL UTC-3 -> UTC = local - (-3) = local + 3h)
df_obs['data_hora']     = df_obs[COL_DATA]
df_obs['data_hora_utc'] = df_obs[COL_DATA] - dt.timedelta(hours=FUSO_LOCAL_H)


# ============================================================
# JUNCAO COM METADADOS (coords reais p/ diagnostico F5 — NAO sao feature)
# ============================================================

print('Juntando metadados...')
df_meta = pd.read_csv(CAMINHO_META, sep=SEP_CSV)
for col in (META_KEY, META_ALT, META_LON, META_LAT):
    if col not in df_meta.columns:
        raise KeyError('Coluna de metadados "' + col + '" ausente (cols: ' + str(list(df_meta.columns)) + ')')

df_meta['codigo_estacao'] = df_meta[META_KEY].astype(str)
n_dup = int(df_meta['codigo_estacao'].duplicated().sum())
if n_dup > 0:
    print('AVISO: metadados com', n_dup, 'codigos duplicados -> mantendo o primeiro')
    df_meta = df_meta.drop_duplicates('codigo_estacao', keep='first')

df_meta_sel = df_meta[['codigo_estacao', META_ALT, META_LON, META_LAT]].rename(
    columns={META_ALT: 'altitude', META_LON: 'longitude', META_LAT: 'latitude'})

df_obs = df_obs.merge(df_meta_sel, on='codigo_estacao', how='left')


# ============================================================
# ESTACOES SEM COORDENADA -> excluir + reportar (sem coords nao ha extracao)
# ============================================================

sem_meta = sorted(df_obs.loc[df_obs['latitude'].isna(), 'codigo_estacao'].unique().tolist())
if len(sem_meta) > 0:
    print('Estacoes SEM metadados de coordenada (excluidas):', sem_meta)
    df_obs    = df_obs[~df_obs['codigo_estacao'].isin(sem_meta)].copy()
    aprovadas = [c for c in aprovadas if c not in sem_meta]
    pd.DataFrame({'codigo_estacao': sem_meta}).to_csv(
        PASTA_SAIDA / f'estacoes_sem_metadados{SUF}.csv', sep=';', index=False, encoding='utf-8')


# ============================================================
# SAIDA
# ============================================================

print('Salvando tabela estacao-hora limpa e lista de aprovadas...')

colunas_saida = ['codigo_estacao', 'data_hora', 'data_hora_utc',
                 COL_TEMP, COL_FLAG, 'temperatura_c_qc',
                 'altitude', 'longitude', 'latitude']
saida_df = df_obs[colunas_saida]
if JANELA and ARQ_SAIDA.exists():   # preserva os outros periodos do parquet; troca so a janela
    antigo = pd.read_parquet(ARQ_SAIDA)
    saida_df = merge_janela(antigo, saida_df, INI_UTC, FIM_UTC)
    print(f'MERGE janela [{INI_UTC}, {FIM_UTC}) em {ARQ_SAIDA.name}: '
          f'{len(antigo)} antigas -> {len(saida_df)} apos merge')
saida_df.to_parquet(ARQ_SAIDA, index=False)

pd.DataFrame({'codigo_estacao': sorted(aprovadas)}).to_csv(
    PASTA_SAIDA / f'estacoes_aprovadas{SUF}.csv', sep=';', index=False, encoding='utf-8')


# ============================================================
# RELATORIO
# ============================================================

n_validas_total = int(df_obs['temperatura_c_qc'].notna().sum())
print('---')
print('Distribuicao md_final x QC final (todas as estacoes lidas):')
print('  flag : total | sobrevive_qc | NaN_final | %sobrevive -> status')
total_flags = sum(contagem_flags.values())
for chave in sorted(contagem_flags, key=lambda k: (isinstance(k, str), k)):
    n_total = contagem_flags[chave]
    n_ok    = contagem_validas.get(chave, 0)   # sobrevivem ao QC completo (so possivel p/ flags validas)
    n_nan   = n_total - n_ok
    pct_ok  = round(100.0 * n_ok / n_total, 2) if n_total else 0.0
    status  = 'VALIDO' if (isinstance(chave, int) and chave in FLAGS_VALIDAS) else 'descartado'
    print('  flag', chave, ':', n_total, '|', n_ok, '|', n_nan, '|',
          str(pct_ok) + '% ->', status)
print('  (flags descartadas devem ter sobrevive_qc=0; nas validas, NaN_final = etapas 2-4 + T ausente)')
print('---')
print('Remocoes por etapa do QC (atribuidas a 1a etapa que derruba; todas as estacoes lidas):')
ETAPAS_ORDEM = [
    ('t_ausente',     'temperatura ausente/nao-numerica'),
    ('flag_invalida', 'md_final fora de (4,5)'),
    ('fora_faixa',    'fora da faixa fisica [' + str(TEMP_MIN_FIS) + ',' + str(TEMP_MAX_FIS) + ']'),
    ('spike',         'salto horario > ' + str(SALTO_MAX_H)),
    ('travado',       '>= ' + str(HORAS_TRAVADO) + 'h identicas'),
]
total_lidas = sum(contagem_flags.values())
for chave, descricao in ETAPAS_ORDEM:
    n = contagem_etapas.get(chave, 0)
    pct = round(100.0 * n / total_lidas, 2) if total_lidas else 0.0
    print('  ' + chave.ljust(13), ':', n, '(' + str(pct) + '%) ->', descricao)
print('  sobrevive    :', contagem_etapas.get('sobrevive', 0),
      '(' + str(round(100.0 * contagem_etapas.get('sobrevive', 0) / total_lidas, 2) if total_lidas else 0.0) + '%)')
print('  soma confere :', sum(contagem_etapas.get(k, 0) for k, _ in ETAPAS_ORDEM) +
      contagem_etapas.get('sobrevive', 0), 'de', total_lidas, 'linhas lidas')
print('---')
print('Estacoes aprovadas:', len(aprovadas))
print('Linhas totais     :', len(df_obs))
print('Linhas validas qc :', n_validas_total)
print('Faixa T qc        :', round(float(df_obs['temperatura_c_qc'].min()), 2),
      '/', round(float(df_obs['temperatura_c_qc'].max()), 2))
print('Periodo data_hora_utc:', str(df_obs['data_hora_utc'].min()), '..', str(df_obs['data_hora_utc'].max()))
