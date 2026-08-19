#!/usr/bin/env python3

# ============================================================
# F8 — Inferencia do downscaling na GRADE de 1 km (560x807), 1 mes, todas as horas.
# Carrega o modelo final (F7) e preve T_final APENAS nos pixels dentro de Santa
# Catarina (mascara IBGE rasterizada na grade canonica); fora de SC -> NaN, nem
# calculado (interpolacao/ancoragem tambem restritas aos pixels de SC):
#     T_final = T_ERA5 + f(X),  T_ERA5 = t2m_1km - 273.15.
# As features sao reconstruidas EXATAMENTE como no treino:
#   - ERA5 (dinamicas): interpola o ERA5-Land bruto no CENTRO de cada celula
#     (bilinear NaN-aware com renormalizacao sobre nos de terra; de-acum de
#     fluxos com reset diario h01), identico ao F3 — mas agora p/ a grade toda.
#     (os era5land_interp1km_celulas_*.nc do F3 so populam as 206 celulas de estacao; nao servem aqui.)
#   - estaticas: lidas dos rasters 1 km (mesma grade canonica do cop1km).
#   - temporais/coords: derivadas do tempo e dos centros (se o modelo as usar).
# A lista de features e a baseline vem do *_meta.json do F7 (fonte unica).
# MODELO SEGMENTADO (F7q): passe em --modelo o META json do treino segmentado.
#   v1 (tipo='segmentado_quantil'): 3 boosters (frio/meio/quente); o F8 ROTEIA cada
#       pixel-hora pelo proprio t2m_1km (K) vs t_lo_K/t_hi_K (corte duro, 1 variavel).
#   v2 (tipo='segmentado_quantil_v2', ex.: modelo_quantil_q16_q84_meta.json): 4 boosters
#       (frio/meio/quente/completo), SEM roteamento — cada booster preve TODA a grade
#       de SC e grava numa variavel propria: t2m_p16, t2m_p16a84, t2m_p84, t2m.
#       Saida: <nome_modelo>_<ano>_<mes>.nc (ex.: modelo_quantil_q16_q84_2020_01.nc).
# Processa por BLOCOS de horas (--chunk-horas) p/ caber na memoria.
# Saida: NetCDF empilhado T_final(time,y,x) + CRS, 1 arquivo por mes.
# --ancora (hindcast): pos-processamento OI que corrige T_final com o residuo OOF
#   das estacoes no mesmo instante (grava SEMPRE <var>_bc = inferencia pura e
#   <var>_oi = campo da correcao OI em degC; contrato: var = var_bc + var_oi).
#   Regras de peso: --ancora-metodo idw|gauss + termo vertical --ancora-Lz (m);
#   tunadas pelo benchmark F8b (f8b_bench_ancoragem.py). Vencedor: idw Lz=500.
# Uso: python f8_infere_grade.py --ano 2020 --mes 1 [--ancora --ancora-Lz 500] [--modelo ...]
# Rodar no env com rioxarray/lightgbm/scipy (py312). Tarefa pesada (~min a 1h).
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
# VM 44 vCPU / 4 NUMA: limitar OpenMP/BLAS antes de importar numpy/lightgbm
# (todas as threads cruzando NUMA fica ~43x mais lento — ver memoria do projeto).
os.environ.setdefault('OMP_NUM_THREADS', '22')

import sys
import json
import time
import calendar
import argparse
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray   # noqa: F401  (registra o acessor .rio)
import lightgbm as lgb
from scipy.spatial import cKDTree
from _root import ROOT, DIR_GRID, DIR_ESTATICAS


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ       = Path(ROOT)
ERA5_RAW   = RAIZ / 'Dados' / 'ERA5_land'
PASTA_GRID = DIR_GRID
PASTA_EST  = DIR_ESTATICAS
PASTA_MODEL = RAIZ / 'saidas' / 'modelos'
PASTA_OUT  = RAIZ / 'saidas' / 'netcdf_inferencia'

CRS_GRADE = 'EPSG:31982'

# mascara do estado de Santa Catarina (recorte da inferencia): 1=centro do pixel
# dentro de SC. Cacheada na grade canonica; construida do limite estadual do IBGE.
ARQ_MASCARA_SC    = PASTA_GRID / 'mascara_sc_1km.tif'
ARQ_SC_MUNICIPIOS = RAIZ / 'Dados' / 'SC_Municipios_2025.zip'

# arquivos ERA5-Land brutos (mesmos do F3)
ARQ_TERMICO = 't2m/ERA5land_{ano}_{mes:02d}.nc'        # t2m, skt, stl1, slhf, sshf
ARQ_VAR     = 'd2m/ERA5land_umid_vento_sp_{ano}_{mes:02d}.nc'    # d2m, u10, v10, sp
ARQ_RAD     = 'rad/ERA5land_radiacao_{ano}_{mes:02d}.nc'    # ssr, str

# feature dinamica -> variaveis ERA5 brutas de que depende.
# (t2m e sempre necessaria p/ a baseline T_ERA5.)
RAW_DEPS = {
    't2m_1km':          ['t2m'],
    'lapse':            ['t2m'],          # + delta_z (estatica)
    'u10':              ['u10'],
    'v10':              ['v10'],
    'vel_vento':        ['u10', 'v10'],
    'sp':               ['sp'],
    'dep_orvalho':      ['t2m', 'd2m'],
    'umidade_relativa': ['t2m', 'd2m'],
    'dt_ar_pele':       ['t2m', 'skt'],
    'dt_ar_solo':       ['t2m', 'stl1'],
    'rad_liquida':      ['ssr', 'str'],
    'ssr':              ['ssr'],           # radiacao solar (onda-curta) individual
    'str':              ['str'],           # radiacao termal (onda-longa) individual
    'razao_bowen':      ['sshf', 'slhf'],
}
DYN_FEATURES = set(RAW_DEPS)
# variavel bruta -> (arquivo, de-acumular?)
RAW_SRC = {
    't2m': ('term', False), 'skt': ('term', False), 'stl1': ('term', False),
    'sshf': ('term', True), 'slhf': ('term', True),
    'd2m': ('var', False), 'u10': ('var', False), 'v10': ('var', False), 'sp': ('var', False),
    'ssr': ('rad', True), 'str': ('rad', True),
}
# estaticas (feature -> raster). z_alvo e delta_z tratadas a parte.
STATIC_TIF = {
    'dev_r1000_p10_1km': PASTA_EST / 'dev_r1000_p10_1km.tif',
    'dev_r3000_mean_1km': PASTA_EST / 'dev_r3000_mean_1km.tif',
    'dev_r300_p10_1km':  PASTA_EST / 'dev_r300_p10_1km.tif',
    'eastness_mean_1km': PASTA_EST / 'eastness_mean_1km.tif',
    'hand_mean_1km':     PASTA_EST / 'hand_mean_1km.tif',
    'mrvbf_p90_1km':     PASTA_EST / 'mrvbf_p90_1km.tif',
    'northness_mean_1km': PASTA_EST / 'northness_mean_1km.tif',
    'tri_mean_1km':      PASTA_EST / 'tri_mean_1km.tif',
    # conjunto ampliado (53 features) — mesmos rasters em estaticas_1km/{feature}.tif
    'dev_r1000_mean_1km': PASTA_EST / 'dev_r1000_mean_1km.tif',
    'dev_r3000_p10_1km': PASTA_EST / 'dev_r3000_p10_1km.tif',
    'dev_r300_mean_1km': PASTA_EST / 'dev_r300_mean_1km.tif',
    'hand_p10_1km':      PASTA_EST / 'hand_p10_1km.tif',
    'slope_max_1km':     PASTA_EST / 'slope_max_1km.tif',
    'slope_mean_1km':    PASTA_EST / 'slope_mean_1km.tif',
    'svf_mean_1km':      PASTA_EST / 'svf_mean_1km.tif',
    'svf_p10_1km':       PASTA_EST / 'svf_p10_1km.tif',
    'tri_max_1km':       PASTA_EST / 'tri_max_1km.tif',
    'tri_std_1km':       PASTA_EST / 'tri_std_1km.tif',
    'z_std_1km':         PASTA_EST / 'z_std_1km.tif',
    'dist_oceano':       PASTA_EST / 'dist_oceano_1km_alinhado.tif',
}
# memoria temporal do ERA5 (t2m): dt2m_{dh}h = t2m(t)-t2m(t-dh) (1a ordem);
# d2t2m_{dh}h = t2m(t)-2*t2m(t-dh)+t2m(t-2dh) (2a ordem), dh in (1,2,3). Espelha o F5.
TEMP_MEM = {f'dt2m_{dh}h' for dh in (1, 2, 3)} | {f'd2t2m_{dh}h' for dh in (1, 2, 3)}
LAG_MAX  = 6   # maior defasagem (2*3 h) necessaria p/ reconstruir dt2m/d2t2m


# ============================================================
# FUNCOES ERA5 (espelham o F3 — mantidas identicas p/ consistencia treino/inferencia)
# ============================================================

def abrir_era5(caminho):
    # CDS quebra request multi-stream (stepTypes mistos: instant+accum) em varios
    # data_*.nc dentro de um zip, mesmo com download_format=unarchived. Detecta e faz
    # merge (mesma grade valid_time/lat/lon, variaveis disjuntas).
    if zipfile.is_zipfile(caminho):
        with zipfile.ZipFile(caminho) as z, tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            partes = [xr.open_dataset(p).drop_vars(['expver', 'number'], errors='ignore').load()
                      for p in sorted(Path(tmp).glob('*.nc'))]
        ds = xr.merge(partes)
    else:
        ds = xr.open_dataset(caminho)
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    ds = ds.drop_vars(['expver', 'number'], errors='ignore')
    if float(ds['longitude'].max()) > 180:
        ds = ds.assign_coords(longitude=(((ds['longitude'] + 180) % 360) - 180)).sortby('longitude')
    return ds


def exigir(ds, nome, caminho):
    if nome not in ds.variables:
        raise KeyError(f'Variavel "{nome}" ausente em {caminho} ({list(ds.variables)})')
    return ds[nome]


def desacumular_fluxo(da):
    # de-acum com reset diario (h01=cru); reindex ao eixo completo (1o 00:00 -> NaN)
    horas    = da['time'].dt.hour
    da_diff  = da.diff('time')
    da_fluxo = xr.where(horas[1:] == 1, da.isel(time=slice(1, None)), da_diff) / 3600.0
    return da_fluxo.reindex(time=da['time'])


def umidade_relativa(temp_c, orvalho_c):
    es = 6.112 * np.exp((17.67 * temp_c)    / (temp_c    + 243.5))
    e  = 6.112 * np.exp((17.67 * orvalho_c) / (orvalho_c + 243.5))
    return 100.0 * e / es


# ============================================================
# INTERPOLADOR NA GRADE (bilinear NaN-aware F3, mas vetorizado p/ 451.920 celulas)
# ============================================================

class InterpoladorGrade:
    # Reproduz interp_celulas do F3: num=fillna(0).interp; den=notnull.interp;
    # out=num/den (renormaliza sobre nos de terra). Para celulas com stencil
    # 100% agua (den==0) usa o no de TERRA mais proximo — aqui via cKDTree
    # PRE-COMPUTADA (o F3 fazia um loop python por celula, invizel p/ a grade toda).
    def __init__(self, lat_alvo, lon_alvo, da_mascara, oceano='nearest'):
        self.lat_da = xr.DataArray(np.asarray(lat_alvo), dims='cel')
        self.lon_da = xr.DataArray(np.asarray(lon_alvo), dims='cel')
        self.oceano = oceano
        # nos de TERRA da fonte (finito em algum t) -> KDTree p/ fallback
        terra = np.isfinite(da_mascara.values).any(axis=0)        # (lat,lon)
        latn = da_mascara['latitude'].values; lonn = da_mascara['longitude'].values
        self.jy, self.jx = np.where(terra)
        vlat = latn[self.jy]; vlon = lonn[self.jx]
        lat0 = float(np.mean(lat_alvo))
        pts_src = np.column_stack([vlon * np.cos(np.deg2rad(lat0)), vlat])
        pts_alvo = np.column_stack([np.asarray(lon_alvo) * np.cos(np.deg2rad(lat0)),
                                    np.asarray(lat_alvo)])
        self.idx_near = cKDTree(pts_src).query(pts_alvo, k=1)[1]    # (ncel,)

    def interp(self, da_slice):
        # da_slice: (time_chunk, lat, lon) -> (time_chunk, ncel) float64
        valido = da_slice.notnull()
        num = da_slice.fillna(0.0).interp(latitude=self.lat_da, longitude=self.lon_da, method='linear')
        den = valido.astype('float64').interp(latitude=self.lat_da, longitude=self.lon_da, method='linear')
        out = (num / den.where(den > 0)).values
        if out.ndim == 1:
            out = out[None, :]
        agua = ~np.isfinite(out).any(axis=0)
        if agua.any() and self.oceano == 'nearest':
            vals = da_slice.values                                  # (t, lat, lon)
            k = self.idx_near[agua]
            out[:, agua] = vals[:, self.jy[k], self.jx[k]]
        return out


# ============================================================
# ESTATICAS (grade canonica)
# ============================================================

def ler_raster(caminho, ny, nx):
    da = rioxarray.open_rasterio(caminho, masked=True).squeeze()
    arr = np.asarray(da.values, dtype='float64')
    if arr.shape != (ny, nx):
        raise ValueError(f'{Path(caminho).name}: shape {arr.shape} != grade ({ny},{nx})')
    return arr.ravel()   # C-order (y,x)


def carregar_estaticas(feats, ny, nx):
    # devolve {feature: vetor (ncel,)} apenas p/ as estaticas presentes em feats.
    out = {}
    z_alvo = ler_raster(PASTA_GRID / 'cop1km.tif', ny, nx)
    if 'z_alvo' in feats:
        out['z_alvo'] = z_alvo
    if 'delta_z' in feats:
        z_orog = ler_raster(PASTA_GRID / 'z_era5_orog_1km.tif', ny, nx)
        out['delta_z'] = z_alvo - z_orog
    for f in feats:
        if f in STATIC_TIF:
            out[f] = ler_raster(STATIC_TIF[f], ny, nx)
    fracs = [f for f in feats if f.startswith('fracao_t')]
    if fracs:
        da = rioxarray.open_rasterio(PASTA_EST / 'fracoes_1km_alinhado.tif', masked=True)
        for f in fracs:
            b = int(f.split('_t')[1])         # fracao_t1..t8 -> banda 1..8
            arr = np.asarray(da.isel(band=b - 1).values, dtype='float64')
            if arr.shape != (ny, nx):
                raise ValueError(f'fracoes banda {b}: shape {arr.shape} != ({ny},{nx})')
            out[f] = arr.ravel()
    return out, z_alvo


def carregar_mascara_sc(ny, nx):
    # Mascara booleana (ny,nx): True = centro do pixel dentro de Santa Catarina.
    # Recorta a inferencia ao territorio de SC (fora -> NaN, nem calculado).
    # Cacheada em ARQ_MASCARA_SC; se ausente, rasteriza o limite estadual (IBGE
    # SC_Municipios_2025) na grade canonica (cop1km, EPSG:31982) — mesmo padrao
    # de build_dist_oceano_sc.py (all_touched=False: centro dentro do poligono).
    if ARQ_MASCARA_SC.exists():
        arr = rioxarray.open_rasterio(ARQ_MASCARA_SC).squeeze().values
        mask = np.asarray(arr).astype(bool)
        if mask.shape != (ny, nx):
            raise ValueError(f'mascara SC: shape {mask.shape} != grade ({ny},{nx})')
        return mask
    # 1a vez: construir a partir do shapefile do IBGE (lazy import de geopandas/rasterio)
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    if not ARQ_SC_MUNICIPIOS.exists():
        sys.exit(f'ERRO: limite de SC ausente: {ARQ_SC_MUNICIPIOS}')
    with rasterio.open(PASTA_GRID / 'cop1km.tif') as ds:
        transform, width, height, prof = ds.transform, ds.width, ds.height, ds.profile.copy()
    muni = gpd.read_file(f'zip://{ARQ_SC_MUNICIPIOS.resolve()}')
    muni = muni[muni.geometry.notna() & ~muni.geometry.is_empty]
    sc = muni.to_crs(CRS_GRADE).union_all()          # dissolve -> geometria estadual (grade F8)
    mask = rasterize([(sc, 1)], out_shape=(height, width), transform=transform,
                     fill=0, all_touched=False, dtype='uint8').astype(bool)
    if not mask.any():
        raise ValueError('mascara SC vazia ao rasterizar na grade canonica.')
    prof.update(dtype='uint8', count=1, nodata=0, compress='deflate')
    with rasterio.open(ARQ_MASCARA_SC, 'w', **prof) as dst:
        dst.write(mask.astype('uint8'), 1)
        dst.set_band_description(1, 'mascara_sc')
    print(f'  mascara SC construida e cacheada: {ARQ_MASCARA_SC.name} '
          f'({int(mask.sum()):,} pixels dentro de SC)', flush=True)
    return mask


def espalhar(vals_sc, idx_sc, m, ny, nx):
    # (m, ncs) sobre pixels de SC -> (m, ny, nx) com NaN fora de SC.
    buf = np.full((m, ny * nx), np.nan, dtype='float32')
    buf[:, idx_sc] = vals_sc
    return buf.reshape(m, ny, nx)


def codificar_tempo(times):
    # mesmos seno/cosseno do F5 (hora e dia-do-ano)
    hora = times.hour.values.astype('float64')
    doy  = times.dayofyear.values.astype('float64')
    return {'hora_sin': np.sin(2 * np.pi * hora / 24),
            'hora_cos': np.cos(2 * np.pi * hora / 24),
            'doy_sin':  np.sin(2 * np.pi * doy / 365.25),
            'doy_cos':  np.cos(2 * np.pi * doy / 365.25)}


# ============================================================
# DERIVACAO DAS FEATURES DINAMICAS (espelha o F3)
# ============================================================

def derivar_dinamica(feat, raw, delta_z, lapse_rate):
    # raw: {var: (nchunk, ncel)} ja interpolado. delta_z: (ncel,)
    if feat == 't2m_1km':     return raw['t2m']
    if feat == 'lapse':       return raw['t2m'] - lapse_rate * delta_z[None, :]
    if feat == 'u10':         return raw['u10']
    if feat == 'v10':         return raw['v10']
    if feat == 'vel_vento':   return np.sqrt(raw['u10'] ** 2 + raw['v10'] ** 2)
    if feat == 'sp':          return raw['sp'] / 100.0
    if feat == 'dep_orvalho': return raw['t2m'] - raw['d2m']
    if feat == 'umidade_relativa': return umidade_relativa(raw['t2m'] - 273.15, raw['d2m'] - 273.15)
    if feat == 'dt_ar_pele':  return raw['t2m'] - raw['skt']
    if feat == 'dt_ar_solo':  return raw['t2m'] - raw['stl1']
    if feat == 'rad_liquida': return raw['ssr'] + raw['str']
    if feat == 'ssr':         return raw['ssr']
    if feat == 'str':         return raw['str']
    if feat == 'razao_bowen': return np.where(np.abs(raw['slhf']) < 1.0, np.nan, raw['sshf'] / raw['slhf'])
    raise KeyError(f'feature dinamica desconhecida: {feat}')


# ============================================================
# ANCORAGEM OI (pos-processamento, hindcast)
# ============================================================
# Corrige o downscaling com o residuo das estacoes no MESMO instante:
#   T_final_OI(g,t) = T_final(g,t) - IDW_g[ inc(s,t) ],
#   inc(s,t) = pred_residuo_OOF(s,t) - y_residuo(s,t)  (= T_final_OOF(s) - T_obs(s)).
# Usa o incremento OOF (leave-cell-out) das celulas-estacao — NAO o do modelo final
# (que ajusta as estacoes in-sample e subestimaria a correcao). Valido em C0:
# RMSE 1,465 -> 1,245 (SS 0,218 -> 0,335). Exige obs contemporaneas -> so hindcast.
# Regras de peso (benchmark leave-cell-out no F8b):
#   idw  : w = 1/d^p nos k vizinhos (padrao; k=20 p=1 = C0)
#   gauss: w = exp(-d^2/L^2) em TODAS as estacoes (Barnes; a gaussiana localiza)
#   +vertical (--ancora-Lz > 0): w *= exp(-dz^2/Lz^2) — bloqueia estacao de vale
#   corrigindo pixel de serra (inversao noturna). Vencedor F8b: idw k=20 p=1
#   Lz=500 m (SS 0,341 -> 0,346 no OOF 5f55final; +0,016 no ramo frio<=p10).
# Ressalva: em celulas g longe de toda estacao a interpolacao extrapola; use
# --ancora-raio p/ zerar a correcao alem de um raio (volta ao downscaling puro).

def preparar_ancoragem(oof_path, tempos, ny, nx, x_full, y_full, z_full,
                       x_query, y_query, z_query, k, power, raio_km,
                       metodo='idw', L_km=50.0, Lz_m=0.0, lz_por_hora=None):
    oof = pd.read_parquet(oof_path, columns=['cell_id', 'data_hora_utc', 'y_residuo', 'pred_residuo'])
    oof = oof.dropna(subset=['pred_residuo'])
    oof['inc'] = oof['pred_residuo'].astype('float64') - oof['y_residuo'].astype('float64')

    # celulas-estacao -> indice flat na grade do F8 (cell_id = "iy_ix", flat = iy*nx+ix)
    cells = sorted(oof['cell_id'].unique())
    iyix  = np.array([list(map(int, c.split('_'))) for c in cells])
    dentro = (iyix[:, 0] >= 0) & (iyix[:, 0] < ny) & (iyix[:, 1] >= 0) & (iyix[:, 1] < nx)
    cells = [c for c, d in zip(cells, dentro) if d]
    flat  = (iyix[dentro, 0] * nx + iyix[dentro, 1]).astype(int)
    nest  = len(cells)
    if nest == 0:
        raise ValueError('ancoragem: nenhuma celula-estacao do OOF cai na grade')

    # vizinhos-estacao de CADA celula de SC (metros) -> pesos.
    # x_full/y_full/z_full: grade cheia (mapear cell_id->coord/altitude da estacao);
    # x_query/y_query/z_query: apenas pixels de SC (celulas onde a correcao e aplicada).
    xy_est = np.column_stack([x_full[flat], y_full[flat]])
    z_est  = z_full[flat]
    tree = cKDTree(xy_est)
    # idw usa k vizinhos; gauss usa TODAS as estacoes (a gaussiana ja localiza,
    # e o corte em k mudaria o resultado vs o benchmark F8b)
    kk = nest if metodo == 'gauss' else min(k, nest)
    dist, idx = tree.query(np.column_stack([x_query, y_query]), k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist_km = dist / 1000.0
    if metodo == 'gauss':
        w = np.exp(-(dist_km / L_km) ** 2)
    else:
        w = 1.0 / np.maximum(dist_km, 1e-6) ** power
    dz = z_est[idx] - z_query[:, None]
    dz = np.where(np.isfinite(dz), dz, 0.0)       # z NaN (inesperado em SC) -> sem modulacao
    if lz_por_hora is None and Lz_m and Lz_m > 0:
        w = w * np.exp(-(dz / Lz_m) ** 2)         # Lz fixo: modulacao pre-computada
    if raio_km and raio_km > 0:
        w = np.where(dist_km <= raio_km, w, 0.0)

    # matriz de incremento (nt, nest) alinhada a `tempos` (NaN onde a estacao falta)
    piv = oof.pivot_table(index='data_hora_utc', columns='cell_id', values='inc')
    piv = piv.reindex(index=pd.DatetimeIndex(tempos), columns=cells)
    inc_mat = piv.values                                    # (nt, nest)
    cob = int(np.isfinite(inc_mat).any(axis=1).sum())       # horas com >=1 estacao
    anc = {'idx': idx.astype('int32'), 'w': w.astype('float64'),
           'inc_mat': inc_mat, 'nest': nest, 'kk': kk, 'horas_cob': cob,
           'dz': None, 'lz_h': None, 'hod': None}
    if lz_por_hora is not None:                             # Lz variavel: modula por hora
        anc['dz']   = dz
        anc['lz_h'] = np.asarray(lz_por_hora, dtype='float64')      # (24,)
        anc['hod']  = pd.DatetimeIndex(tempos).hour.values          # (nt,)
    return anc


def aplicar_ancoragem(tfin, sl, anc):
    # tfin: (m, ncel) T_final (background). Subtrai a interpolacao do incremento por hora.
    # Retorna (tfin_ancorado, corr): corr = campo da OI aplicado (= T_final - T_background =
    # -inc_hat), CAPTURADO direto do calculo (nao reconstruido por t2m-t2m_bc). 0 onde nao ha
    # vizinho-estacao. Aditivo: T_final = T_background + corr.
    idx, w, inc_mat = anc['idx'], anc['w'], anc['inc_mat']
    corr = np.zeros_like(tfin)
    for h in range(tfin.shape[0]):
        inc_t = inc_mat[sl.start + h]                       # (nest,)
        wt = w
        if anc['lz_h'] is not None:                         # Lz variavel por hora do dia
            lz = anc['lz_h'][anc['hod'][sl.start + h]]
            if lz > 0:
                wt = w * np.exp(-(anc['dz'] / lz) ** 2)
        g = inc_t[idx]                                      # (ncel, kk)
        mfin = np.isfinite(g)
        num = np.where(mfin, wt * np.nan_to_num(g), 0.0).sum(axis=1)
        den = np.where(mfin, wt, 0.0).sum(axis=1)
        inc_hat = np.divide(num, den, out=np.zeros_like(num), where=den > 0)  # 0=sem vizinho
        corr[h] = (-inc_hat).astype('float32')
        tfin[h] = tfin[h] + corr[h]
    return tfin, corr


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Inferencia do downscaling na grade de 1 km (F8).')
    ap.add_argument('--ano', type=int, default=None, help='ano do mes (dispensavel se usar --init)')
    ap.add_argument('--mes', type=int, default=None, help='mes (dispensavel se usar --init)')
    ap.add_argument('--init', default=None,
                    help='data inicial YYYY-MM-DD (inclusive); define o mes e a janela. Acelera '
                         'a inferencia processando so o intervalo [--init, --end].')
    ap.add_argument('--end', default=None,
                    help='data final YYYY-MM-DD (inclusive; default = fim do mes de --init). '
                         'Deve estar no MESMO mes de --init (ERA5 e por mes).')
    ap.add_argument('--modelo', default=str(PASTA_MODEL / 'modelo_final.txt'))
    ap.add_argument('--meta',   default=None, help='json de metadados (default: <modelo>_meta.json)')
    ap.add_argument('--saida',  default=str(PASTA_OUT), help='diretorio OU arquivo .nc')
    ap.add_argument('--chunk-horas', type=int, default=24)
    ap.add_argument('--max-horas', type=int, default=0,
                    help='limita o nº de horas processadas do inicio do mes (teste; use --init/--end '
                         'p/ janela por data)')
    ap.add_argument('--oceano', choices=['nearest', 'nan'], default='nearest',
                    help='celulas 100%% oceano: no de terra mais proximo (F3) ou NaN')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 22)),
                    help='threads na predicao LightGBM (nesta VM, 22; nao usar 0)')
    ap.add_argument('--extras', action='store_true',
                    help='tambem grava T_ERA5 (baseline) e pred_residuo')
    ap.add_argument('--ancora', action='store_true',
                    help='pos-proc OI (hindcast): corrige T_final com o residuo OOF das estacoes '
                         'no MESMO instante, interpolado p/ a grade. Requer obs no periodo (--ancora-oof).')
    ap.add_argument('--ancora-oof', default=str(PASTA_MODEL / 'oof_ensaio.parquet'),
                    help='parquet OOF (cell_id,data_hora_utc,y_residuo,pred_residuo): fonte do incremento')
    ap.add_argument('--ancora-metodo', choices=['idw', 'gauss'], default='idw',
                    help='regra de peso: idw (1/d^p, k vizinhos; atual) ou gauss '
                         '(exp(-d^2/L^2), todas as estacoes; Barnes)')
    ap.add_argument('--ancora-k', type=int, default=20, help='vizinhos-estacao por celula (idw)')
    ap.add_argument('--ancora-power', type=float, default=1.0, help='expoente do idw (1=melhor no C0)')
    ap.add_argument('--ancora-L', type=float, default=50.0,
                    help='comprimento de escala horizontal L (km) do gauss')
    ap.add_argument('--ancora-Lz', type=float, default=0.0,
                    help='escala vertical Lz (m): w *= exp(-dz^2/Lz^2); 0 = sem termo '
                         'vertical (comportamento atual). Vencedor F8b: 500 m com idw k=20 p=1.')
    ap.add_argument('--ancora-Lz-json', default=None,
                    help='json do F8b --por-hora (lz_por_hora_<tag>.json): Lz variavel por '
                         'hora do dia; sobrepoe --ancora-Lz')
    ap.add_argument('--ancora-raio', type=float, default=0.0,
                    help='raio maximo (km) dos vizinhos; 0=sem limite (reproduz o C0). '
                         'Cap protege celulas distantes da rede de extrapolacao.')
    args = ap.parse_args()

    # --- periodo: --init/--end (janela por data, PODE cruzar meses) OU --ano/--mes ---
    # ERA5 e 1 arquivo/mes; a janela define o span de meses a carregar/concatenar.
    janela_datas = None
    if args.init or args.end:
        try:
            d_ini = pd.Timestamp(args.init) if args.init else None
            d_fim = pd.Timestamp(args.end) if args.end else None
        except ValueError as e:
            sys.exit(f'ERRO: data invalida em --init/--end (use YYYY-MM-DD): {e}')
        ref = d_ini if d_ini is not None else d_fim
        ano, mes = ref.year, ref.month
        if d_ini is not None and d_fim is not None and d_fim < d_ini:
            sys.exit(f'ERRO: --end={d_fim.date()} < --init={d_ini.date()}.')
        span_ini = (d_ini if d_ini is not None else ref).normalize()
        span_fim = (d_fim if d_fim is not None else ref).normalize()
        meses = pd.period_range(span_ini, span_fim, freq='M')
        janela_datas = (d_ini, d_fim)
    elif args.ano and args.mes:
        ano, mes = args.ano, args.mes
        meses = pd.period_range(f'{ano}-{mes:02d}', periods=1, freq='M')
    else:
        sys.exit('ERRO: informe --ano e --mes, OU --init YYYY-MM-DD [--end YYYY-MM-DD].')

    # --- modelo + metadados (features, baseline, lapse) ---
    # --modelo aceita: booster unico do F7 (*.txt) OU meta json do F7q segmentado
    # (tipo='segmentado_quantil': carrega os 3 boosters e roteia por t2m_1km).
    modelo = Path(args.modelo)
    if modelo.suffix == '.json':
        meta_p = modelo
    else:
        meta_p = Path(args.meta) if args.meta else modelo.with_name(modelo.stem + '_meta.json')
    if not meta_p.exists():
        sys.exit(f'ERRO: metadados nao encontrados: {meta_p}')
    meta = json.loads(meta_p.read_text())
    feats = meta['features']
    lapse_rate = float(meta.get('lapse_rate', 0.0065))
    tipo_meta     = meta.get('tipo')
    segmentado    = tipo_meta == 'segmentado_quantil'         # v1: roteado por t2m_1km (1 var)
    segmentado_v2 = tipo_meta == 'segmentado_quantil_v2'      # v2: 4 boosters, SEM roteamento (4 vars)
    if segmentado or segmentado_v2:
        corte = meta['corte']
        t_lo_K, t_hi_K = float(corte['t_lo_K']), float(corte['t_hi_K'])
        ramos_seg = tuple(meta['ramos'])                     # v1: (frio,meio,quente); v2: +completo
        boosters = {}
        for ramo in ramos_seg:
            f_mod = meta_p.parent / meta['ramos'][ramo]['modelo']
            if not f_mod.exists():
                sys.exit(f'ERRO: booster do ramo "{ramo}" nao encontrado: {f_mod} '
                         f'(rode f7q_treina_quantil.py sem --cv antes)')
            boosters[ramo] = lgb.Booster(model_file=str(f_mod))
        nome_modelo = meta_p.name.replace('_meta.json', '')
        if segmentado_v2:
            # ramo -> nome da variavel NetCDF (t2m_p16/t2m_p16a84/t2m_p84/t2m). Sem roteamento:
            # os 4 boosters preveem TODA a grade de SC, cada um na sua variavel.
            saida_vars = meta.get('saida_vars') or {
                r: meta['ramos'][r].get('saida_var', f't2m_{r}') for r in ramos_seg}
            print(f'Modelo SEGMENTADO v2 {nome_modelo}: ramos {list(ramos_seg)} -> vars '
                  f'{[saida_vars[r] for r in ramos_seg]} | {len(feats)} features | SEM roteamento '
                  f'(4 variaveis na grade toda)', flush=True)
        else:
            print(f'Modelo SEGMENTADO {nome_modelo}: ramos {list(ramos_seg)} | {len(feats)} features | '
                  f'corte {corte["var"]} q{int(corte["q_lo"]*100)}/q{int(corte["q_hi"]*100)} '
                  f'({t_lo_K - 273.15:.2f} / {t_hi_K - 273.15:.2f} °C)', flush=True)
    else:
        if not modelo.exists():
            sys.exit(f'ERRO: modelo nao encontrado: {modelo}  (rode f7_treina_final.py antes)')
        booster = lgb.Booster(model_file=str(modelo))
        nome_modelo = modelo.name
        print(f'Modelo: {modelo.name} | {len(feats)} features | best_iter={meta.get("best_iteration")}', flush=True)

    estat_feats = [f for f in feats if f not in DYN_FEATURES and f not in TEMP_MEM
                   and not f.startswith(('hora_', 'doy_')) and f not in ('coord_x', 'coord_y')]
    dyn_feats   = [f for f in feats if f in DYN_FEATURES]
    temp_feats  = [f for f in feats if f.startswith(('hora_', 'doy_'))]
    coord_feats = [f for f in feats if f in ('coord_x', 'coord_y')]
    mem_feats   = [f for f in feats if f in TEMP_MEM]

    # (arquivos ERA5 brutos sao resolvidos por mes no loop de carga abaixo)

    # --- grade canonica (cop1km) + centros WGS84 (interp do ERA5) ---
    da_z = rioxarray.open_rasterio(PASTA_GRID / 'cop1km.tif', masked=True).squeeze()
    xs = da_z['x'].values; ys = da_z['y'].values
    ny, nx = len(ys), len(xs); ncel = ny * nx
    g = np.load(PASTA_GRID / 'centros_wgs84.npz')
    lat_flat = g['lat_centros'].ravel(); lon_flat = g['lon_centros'].ravel()
    print(f'Grade: {ny} x {nx} = {ncel:,} celulas | mes {ano}-{mes:02d}', flush=True)

    # --- mascara de Santa Catarina: recorta a inferencia ao territorio do estado ---
    # so os pixels dentro de SC sao calculados (interpolacao/features/predicao/ancoragem);
    # fora de SC a saida fica NaN. idx_sc = indices flat (C-order y,x) dos pixels de SC.
    mask_sc = carregar_mascara_sc(ny, nx)
    idx_sc  = np.flatnonzero(mask_sc.ravel())
    ncs     = int(idx_sc.size)
    lat_sc  = lat_flat[idx_sc]; lon_sc = lon_flat[idx_sc]
    print(f'  mascara SC: {ncs:,}/{ncel:,} pixels ({100*ncs/ncel:.1f}%) — fora de SC = NaN', flush=True)

    # --- estaticas + temporais/coords (recortadas aos pixels de SC) ---
    # z_full (altitude cop1km, grade cheia): usado pelo termo vertical da ancoragem
    estaticas, z_full = carregar_estaticas(set(estat_feats) | {'delta_z'}, ny, nx)
    if 'delta_z' not in estaticas:            # delta_z sempre necessaria p/ lapse
        z_alvo = ler_raster(PASTA_GRID / 'cop1km.tif', ny, nx)
        estaticas['delta_z'] = z_alvo - ler_raster(PASTA_GRID / 'z_era5_orog_1km.tif', ny, nx)
    estaticas = {k: v[idx_sc] for k, v in estaticas.items()}   # (ncel,) -> (ncs,)
    delta_z = estaticas['delta_z']
    coords = {}
    if coord_feats:
        coords['coord_x'] = g['X_grid'].ravel()[idx_sc]
        coords['coord_y'] = g['Y_grid'].ravel()[idx_sc]

    # --- ERA5 bruto por MES no span (janela pode cruzar meses): abrir, validar
    # eixo de tempo, de-acumular fluxo POR MES e concatenar no eixo de tempo. ---
    need_raw = {'t2m'}
    for f in dyn_feats:
        need_raw.update(RAW_DEPS[f])
    coarse_parts = {v: [] for v in need_raw}
    tempos_parts = []
    for per in meses:
        a_m, m_m = per.year, per.month
        cm = {'term': ERA5_RAW / ARQ_TERMICO.format(ano=a_m, mes=m_m),
              'var':  ERA5_RAW / ARQ_VAR.format(ano=a_m, mes=m_m),
              'rad':  ERA5_RAW / ARQ_RAD.format(ano=a_m, mes=m_m)}
        for k, p in cm.items():
            if not p.exists():
                sys.exit(f'ERRO: ERA5 bruto ausente ({k}): {p}')
        ds_m = {g: abrir_era5(cm[g]) for g in cm}
        nt_m = ds_m['term'].sizes['time']
        horas_m = calendar.monthrange(a_m, m_m)[1] * 24
        if nt_m != horas_m:
            sys.exit(f'ERRO: ERA5 {a_m}-{m_m:02d}: {nt_m}h != {horas_m}h do mes (fonte incompleta).')
        tempos_m = pd.date_range(f'{a_m}-{m_m:02d}-01', periods=nt_m, freq='h')
        src = np.asarray(ds_m['term']['time'].values)
        if np.issubdtype(src.dtype, np.datetime64):
            if not pd.DatetimeIndex(src).equals(tempos_m):
                sys.exit(f'ERRO: ERA5 {a_m}-{m_m:02d}: eixo de tempo diverge do horario canonico.')
        else:                                 # fonte sem eixo CF (bug jan/2021): adota canonico
            print(f'  AVISO: {a_m}-{m_m:02d} sem eixo de tempo CF; adotando horario canonico.', flush=True)
            ds_m = {g: ds_m[g].assign_coords(time=tempos_m) for g in ds_m}
        for v in sorted(need_raw):
            grp, deacum = RAW_SRC[v]
            da = exigir(ds_m[grp], v, cm[grp])
            coarse_parts[v].append(desacumular_fluxo(da) if deacum else da)
        tempos_parts.append(tempos_m)
    # ponytail: concat carrega o span inteiro de meses; saida (nt,ny,nx) escala com
    # o span. Janela de 10 dias cruzando 1 borda = 2 meses (~2x memoria do mes).
    if len(meses) > 1:
        coarse = {v: xr.concat(coarse_parts[v], dim='time') for v in need_raw}
        tempos = tempos_parts[0].append(tempos_parts[1:])
    else:
        coarse = {v: coarse_parts[v][0] for v in need_raw}
        tempos = tempos_parts[0]
    nt = len(tempos)
    print(f'  ERA5 brutas a interpolar: {sorted(need_raw)} | {len(meses)} mes(es), {nt} h', flush=True)

    interp = InterpoladorGrade(lat_sc, lon_sc, da_mascara=coarse['t2m'], oceano=args.oceano)

    # --- janela de horas a processar [lo, hi) na grade do mes ---
    # nt/tempos ficam com o MES INTEIRO (a ancoragem e as features de memoria usam o mes
    # como contexto; a saida e recortada a [lo, hi) so no fim). Origem da janela:
    #   --init/--end -> intervalo por data ; --max-horas -> prefixo (teste) ; senao mes cheio.
    lo, hi = 0, nt
    if janela_datas is not None:
        d_ini, d_fim = janela_datas
        ini = d_ini if d_ini is not None else tempos[0].normalize()
        fim = (d_fim if d_fim is not None else tempos[-1].normalize()) + pd.Timedelta(days=1)  # exclusivo
        lo = max(0, int((ini - tempos[0]) / pd.Timedelta(hours=1)))
        hi = min(nt, int((fim - tempos[0]) / pd.Timedelta(hours=1)))
        if not lo < hi:
            sys.exit(f'ERRO: janela --init/--end vazia ou fora de {ano}-{mes:02d} (lo={lo} hi={hi}).')
        print(f'  janela: {tempos[lo]:%Y-%m-%d} .. {tempos[hi-1]:%Y-%m-%d %H}h '
              f'({hi-lo} de {nt} h do mes)', flush=True)
    elif args.max_horas and args.max_horas < nt:
        hi = args.max_horas
        print(f'  [TESTE] limitando a {hi} h', flush=True)

    # --- ancoragem OI (opcional): precomputa vizinhos/pesos e matriz de incremento ---
    anc = None
    if args.ancora:
        oof_p = Path(args.ancora_oof)
        if not oof_p.exists():
            sys.exit(f'ERRO: --ancora requer o OOF: {oof_p} (rode f6_ensaio_lightgbm.py antes)')
        lz_h = None
        if args.ancora_Lz_json:
            jlz = json.loads(Path(args.ancora_Lz_json).read_text())
            lz_h = [float(jlz['lz_por_hora'][str(h)]) for h in range(24)]
        xg = g['X_grid'].ravel(); yg = g['Y_grid'].ravel()
        anc = preparar_ancoragem(oof_p, tempos, ny, nx,
                                 xg, yg, z_full, xg[idx_sc], yg[idx_sc], z_full[idx_sc],
                                 args.ancora_k, args.ancora_power, args.ancora_raio,
                                 args.ancora_metodo, args.ancora_L, args.ancora_Lz, lz_h)
        raio_s = 'sem limite' if not args.ancora_raio else f'{args.ancora_raio:.0f} km'
        vert_s = (f'Lz=por hora ({min(lz_h):g}-{max(lz_h):g}m)' if lz_h is not None else
                  f'Lz={args.ancora_Lz:.0f}m' if args.ancora_Lz > 0 else 'sem termo vertical')
        prm_s  = f'L={args.ancora_L:.0f}km' if args.ancora_metodo == 'gauss' else \
                 f'k={anc["kk"]} power={args.ancora_power}'
        print(f'  ANCORAGEM OI [{args.ancora_metodo}]: {anc["nest"]} estacoes | {prm_s} '
              f'{vert_s} raio={raio_s} | {anc["horas_cob"]}/{nt} h com obs', flush=True)
        if anc['horas_cob'] == 0:
            sys.exit(f'ERRO: --ancora sem cobertura de obs p/ {ano}-{mes:02d} no OOF '
                     f'(periodo fora do treino?). Use um mes coberto ou rode sem --ancora.')

    # --- saidas (alocadas no mes inteiro) ---
    # v2: por ramo -> out_v2 (ancorado) + out_v2_bc (inferencia pura) + out_v2_oi (campo da OI).
    if segmentado_v2:
        # SEMPRE: <var> (ancorado) + <var>_bc (inferencia pura) + <var>_oi (campo da OI, degC).
        # Sem ancoragem, _oi=0 e _bc==<var> — contrato de saida invariante (var = var_bc + var_oi).
        out_v2    = {r: np.full((nt, ny, nx), np.nan, dtype='float32') for r in ramos_seg}
        out_v2_bc = {r: np.full((nt, ny, nx), np.nan, dtype='float32') for r in ramos_seg}
        out_v2_oi = {r: np.full((nt, ny, nx), np.nan, dtype='float32') for r in ramos_seg}
        out_tf = out_bg = out_stn = None
    else:
        out_tf = np.full((nt, ny, nx), np.nan, dtype='float32')
        out_bg = np.full((nt, ny, nx), np.nan, dtype='float32') if anc else None
        out_stn = np.full((nt, ny, nx), np.nan, dtype='float32') if anc else None
    out_te = np.full((nt, ny, nx), np.nan, dtype='float32') if args.extras else None
    out_pr = np.full((nt, ny, nx), np.nan, dtype='float32') if args.extras else None

    # estaticas/coords como colunas estaticas (ncs,) na ordem de feats
    col_estatica = {**estaticas, **coords}

    # ========================================================
    # LOOP POR BLOCOS DE HORAS
    # ========================================================

    ch = max(1, args.chunk_horas)
    nblocos = (hi - lo + ch - 1) // ch
    print(f'Prevendo {hi-lo} h em {nblocos} blocos de ate {ch} h...', flush=True)
    t_ini = time.time()
    for b, t0 in enumerate(range(lo, hi, ch), 1):
        t1 = min(t0 + ch, hi); m = t1 - t0
        sl = slice(t0, t1)

        # t2m com extensao RETROATIVA (ate LAG_MAX h) p/ reconstruir dt2m/d2t2m
        # sem descontinuidade nas bordas de bloco (espelha o _t2m_lag do F5).
        mem_vals = {}
        if mem_feats:
            te0 = max(0, t0 - LAG_MAX)
            t2m_ext = interp.interp(coarse['t2m'].isel(time=slice(te0, t1)))   # (t1-te0, ncs) K
            off = t0 - te0
            raw = {v: (t2m_ext[off:off + m] if v == 't2m' else interp.interp(coarse[v].isel(time=sl)))
                   for v in need_raw}
            gt = np.arange(t0, t1)
            def _t2m_lag(k):                       # t2m em (gt-k); NaN antes do inicio do mes
                gi = gt - k; okg = gi >= 0
                o = np.full((m, ncs), np.nan)
                o[okg] = t2m_ext[gi[okg] - te0]
                return o
            for dh in (1, 2, 3):
                if f'dt2m_{dh}h' in mem_feats or f'd2t2m_{dh}h' in mem_feats:
                    p1 = _t2m_lag(dh); p2 = _t2m_lag(2 * dh)
                    mem_vals[f'dt2m_{dh}h']  = raw['t2m'] - p1
                    mem_vals[f'd2t2m_{dh}h'] = raw['t2m'] - 2 * p1 + p2
        else:
            raw = {v: interp.interp(coarse[v].isel(time=sl)) for v in need_raw}   # {v:(m,ncs)}
        baseline_c = raw['t2m'] - 273.15                                       # (m,ncs)

        # monta X (m*ncs, nfeat) na ordem de feats — apenas pixels de SC
        X = np.empty((m * ncs, len(feats)), dtype='float32')
        temp_vals = codificar_tempo(tempos[sl]) if temp_feats else {}
        for j, f in enumerate(feats):
            if f in mem_vals:
                col = mem_vals[f].astype('float32').ravel()
            elif f in DYN_FEATURES:
                col = derivar_dinamica(f, raw, delta_z, lapse_rate).astype('float32').ravel()
            elif f in col_estatica:
                col = np.broadcast_to(col_estatica[f].astype('float32'), (m, ncs)).ravel()
            elif f in temp_vals:
                col = np.repeat(temp_vals[f].astype('float32'), ncs)
            else:
                raise KeyError(f'feature sem fonte definida: {f}')
            X[:, j] = col

        if segmentado_v2:
            # v2: SEM roteamento — cada booster preve TODA a grade de SC, numa variavel propria.
            # SEMPRE grava _bc (inferencia pura) e _oi (campo da OI, degC); com ancoragem, _oi!=0.
            for ramo in ramos_seg:
                pred_r = boosters[ramo].predict(X, num_threads=args.threads).reshape(m, ncs)
                tfin_r = (baseline_c + pred_r).astype('float32')              # (m, ncs) inferencia pura
                out_v2_bc[ramo][sl] = espalhar(tfin_r, idx_sc, m, ny, nx)     # _bc: snapshot pre-ancoragem
                if anc:
                    tfin_r, corr_r = aplicar_ancoragem(tfin_r, sl, anc)       # OI + campo aplicado
                else:
                    corr_r = np.zeros_like(tfin_r)                            # sem ancoragem: correcao 0
                out_v2_oi[ramo][sl] = espalhar(corr_r, idx_sc, m, ny, nx)     # _oi: campo da OI (degC)
                out_v2[ramo][sl] = espalhar(tfin_r, idx_sc, m, ny, nx)        # ancorado (= _bc se sem ancoragem)
            # baseline_c/pred p/ --extras: usa o ramo 'completo' (p0-100) como referencia
            pred = (boosters['completo'].predict(X, num_threads=args.threads).reshape(m, ncs)
                    if args.extras and 'completo' in boosters else None)
        else:
            if segmentado:
                # roteamento por pixel-hora: t2m_1km (K) vs limiares do meta (corte duro).
                # raw['t2m'] (m,ncs) ravel C-order = mesma ordem das linhas de X.
                t2m_k = raw['t2m'].ravel()
                ramo_px = np.ones(t2m_k.size, dtype='int8')
                ramo_px[t2m_k <= t_lo_K] = 0
                ramo_px[t2m_k > t_hi_K]  = 2
                pred = np.empty(m * ncs, dtype='float64')
                for kr, ramo in enumerate(ramos_seg):
                    mr = ramo_px == kr
                    if mr.any():
                        pred[mr] = boosters[ramo].predict(X[mr], num_threads=args.threads)
                pred = pred.reshape(m, ncs)
            else:
                pred = booster.predict(X, num_threads=args.threads).reshape(m, ncs)
            tfin = (baseline_c + pred).astype('float32')                      # (m, ncs)
            if anc:
                out_bg[sl] = espalhar(tfin, idx_sc, m, ny, nx)  # background (downscaling puro)
                tfin, corr = aplicar_ancoragem(tfin, sl, anc)   # T_final <- ancorado (m, ncs)
                out_stn[sl] = espalhar(corr, idx_sc, m, ny, nx) # campo da OI aplicado
            out_tf[sl] = espalhar(tfin, idx_sc, m, ny, nx)      # espalha p/ grade (NaN fora de SC)
        if args.extras and pred is not None:
            out_te[sl] = espalhar(baseline_c.astype('float32'), idx_sc, m, ny, nx)
            out_pr[sl] = espalhar(pred.astype('float32'), idx_sc, m, ny, nx)

        dt = time.time() - t_ini
        eta = dt / b * (nblocos - b)
        print(f'  bloco {b}/{nblocos} (h {t0:03d}-{t1-1:03d}) | {dt/60:.1f} min | ETA {eta/60:.1f} min', flush=True)

    # ========================================================
    # SAIDA NetCDF
    # ========================================================

    # recorta a saida a janela [lo, hi) (o mes inteiro so serviu de contexto p/ memoria/ancoragem)
    if (lo, hi) != (0, nt):
        tempos = tempos[lo:hi]
        _sl = slice(lo, hi)
        if segmentado_v2:
            out_v2    = {r: out_v2[r][_sl]    for r in ramos_seg}
            out_v2_bc = {r: out_v2_bc[r][_sl] for r in ramos_seg}
            out_v2_oi = {r: out_v2_oi[r][_sl] for r in ramos_seg}
        else:
            out_tf = out_tf[_sl]
            if anc:
                out_bg = out_bg[_sl]; out_stn = out_stn[_sl]
        if args.extras:
            out_te = out_te[_sl]; out_pr = out_pr[_sl]

    # descricao da ancoragem (comum a todas as variaveis quando --ancora)
    anc_desc = None
    if anc:
        anc_desc = (f'OI resid. estacoes; metodo={args.ancora_metodo}, '
                    + (f'L_km={args.ancora_L}, ' if args.ancora_metodo == 'gauss'
                       else f'k={anc["kk"]}, power={args.ancora_power}, ')
                    + (f'Lz_m=por_hora({Path(args.ancora_Lz_json).name})'
                       if args.ancora_Lz_json else f'Lz_m={args.ancora_Lz or "sem"}')
                    + f', raio_km={args.ancora_raio or "inf"}')

    if segmentado_v2:
        data_vars = {saida_vars[r]: (('time', 'y', 'x'), out_v2[r]) for r in ramos_seg}
        for r in ramos_seg:   # SEMPRE: _bc (inferencia pura) + _oi (campo da OI, degC)
            data_vars[saida_vars[r] + '_bc'] = (('time', 'y', 'x'), out_v2_bc[r])
            data_vars[saida_vars[r] + '_oi'] = (('time', 'y', 'x'), out_v2_oi[r])
    else:
        data_vars = {'T_final': (('time', 'y', 'x'), out_tf)}
        if anc:
            data_vars['T_final_bc'] = (('time', 'y', 'x'), out_bg)    # inferencia pura (auditoria)
            data_vars['T_final_oi'] = (('time', 'y', 'x'), out_stn)   # campo da OI aplicado
    if args.extras:
        data_vars['T_ERA5']       = (('time', 'y', 'x'), out_te)
        data_vars['pred_residuo'] = (('time', 'y', 'x'), out_pr)
    ds_out = xr.Dataset(data_vars, coords={'time': tempos, 'y': ys, 'x': xs})
    ds_out.attrs['mascara'] = ('recorte ao estado de Santa Catarina (IBGE SC_Municipios_2025 '
                               'rasterizado na grade canonica); pixels fora de SC = NaN')

    if segmentado_v2:
        # 4 variaveis: cada booster na grade toda (sem roteamento). Documenta o subconjunto de treino.
        subset = {'frio': f'<=q{int(corte["q_lo"]*100)} (p0-{int(corte["q_lo"]*100)})',
                  'meio': f'q{int(corte["q_lo"]*100)}-q{int(corte["q_hi"]*100)}',
                  'quente': f'>q{int(corte["q_hi"]*100)} (p{int(corte["q_hi"]*100)}-100)',
                  'completo': 'todas as linhas (p0-100)'}
        for r in ramos_seg:
            v = saida_vars[r]
            ds_out[v].attrs = {
                'long_name': f'temperatura do ar 2m (downscaling 1km, ramo {r})'
                             + (' + ancoragem OI' if anc else ''),
                'units': 'degC', 'baseline': 'ERA5-Land t2m bilinear',
                'modelo': nome_modelo, 'ramo': r,
                'treino_subconjunto': f'{corte["var"]} {subset.get(r, r)}',
                'mascara': 'Santa Catarina (IBGE); fora de SC = NaN'}
            if anc:
                ds_out[v].attrs['ancoragem'] = anc_desc
                ds_out[v].attrs['fonte_incremento'] = Path(args.ancora_oof).name
            ds_out[v + '_bc'].attrs = {
                'long_name': f'temperatura do ar 2m (downscaling 1km, ramo {r}, sem ancoragem OI)',
                'units': 'degC', 'modelo': nome_modelo, 'ramo': r,
                'definicao': f'inferencia pura da rede (contrato: {v} = {v}_bc + {v}_oi)'}
            ds_out[v + '_oi'].attrs = {
                'long_name': f'campo da ancoragem OI (correcao das estacoes) aplicado, ramo {r}',
                'units': 'degC', 'modelo': nome_modelo, 'ramo': r,
                'definicao': f'correcao OI em degC (aditiva: {v} = {v}_bc + {v}_oi); '
                             '0 onde nao ha estacao vizinha ou sem ancoragem',
                'ancoragem': anc_desc if anc else 'sem ancoragem (campo 0)',
                'fonte_incremento': Path(args.ancora_oof).name if anc else 'n/a'}
    else:
        ds_out['T_final'].attrs = {'long_name': 'temperatura do ar 2m (downscaling 1km)',
                                   'units': 'degC', 'baseline': 'ERA5-Land t2m bilinear',
                                   'modelo': nome_modelo,
                                   'mascara': 'Santa Catarina (IBGE); fora de SC = NaN'}
        if segmentado:
            ds_out['T_final'].attrs['segmentacao'] = (
                f'{corte["var"]} q{int(corte["q_lo"]*100)}/q{int(corte["q_hi"]*100)}: '
                f'frio<={t_lo_K:.2f}K < meio <= {t_hi_K:.2f}K < quente (corte duro, F7q)')
        if anc:
            ds_out['T_final'].attrs.update(
                {'long_name': 'temperatura do ar 2m (downscaling 1km + ancoragem OI)',
                 'ancoragem': anc_desc, 'fonte_incremento': Path(args.ancora_oof).name})
            ds_out['T_final_bc'].attrs = {'long_name': 'temperatura do ar 2m (downscaling 1km, sem ancoragem)',
                                          'units': 'degC', 'modelo': nome_modelo}
            ds_out['T_final_oi'].attrs = {
                'long_name': 'campo da ancoragem OI (correcao das estacoes) aplicado',
                'units': 'degC', 'modelo': nome_modelo,
                'definicao': 'correcao OI em degC (aditiva: T_final = T_final_bc + T_final_oi)',
                'ancoragem': anc_desc, 'fonte_incremento': Path(args.ancora_oof).name}
    # coords WGS84 (lat/lon 2D): a grade e projetada (EPSG:31982), entao lat/lon variam
    # em 2D. Adiciona como coords auxiliares CF a partir dos centros ja precomputados
    # (centros_wgs84.npz) — QGIS/xarray reconhecem via o atributo 'coordinates'.
    lat2d = g['lat_centros'].reshape(ny, nx).astype('float64')
    lon2d = g['lon_centros'].reshape(ny, nx).astype('float64')
    ds_out = ds_out.assign_coords(lat=(('y', 'x'), lat2d), lon=(('y', 'x'), lon2d))
    ds_out['lat'].attrs = {'standard_name': 'latitude', 'long_name': 'latitude (WGS84)',
                           'units': 'degrees_north'}
    ds_out['lon'].attrs = {'standard_name': 'longitude', 'long_name': 'longitude (WGS84)',
                           'units': 'degrees_east'}
    ds_out.rio.write_crs(CRS_GRADE, inplace=True)

    saida = Path(args.saida)
    if saida.suffix == '.nc':
        saida.parent.mkdir(parents=True, exist_ok=True); arq = saida
    else:
        saida.mkdir(parents=True, exist_ok=True)
        # v1/v2 seg: prefixa com nome_modelo p/ nao colidir com o T_final do monolitico
        prefixo = nome_modelo if (segmentado or segmentado_v2) else 'T_final'
        # janela por data -> sufixo AAAAMMDD_AAAAMMDD (nao sobrescreve o mes cheio)
        periodo = (f'{ano}_{mes:02d}' if janela_datas is None
                   else f'{tempos[0]:%Y%m%d}_{tempos[-1]:%Y%m%d}')
        arq = saida / f'{prefixo}_{periodo}.nc'
    # liga cada variavel ao spatial_ref (grid_mapping) — sem isso o encoding explicito
    # descarta o link e GDAL/QGIS nao associam o CRS a saida.
    enc = {v: {'zlib': True, 'complevel': 4, 'grid_mapping': 'spatial_ref'} for v in data_vars}
    ds_out.to_netcdf(arq, encoding=enc)

    print(f'\nSalvo: {arq}', flush=True)
    if segmentado_v2:
        for r in ramos_seg:
            a = out_v2[r]; fin = np.isfinite(a)
            linha = (f'  {saida_vars[r]:10s} (ramo {r:8s}): {fin.sum():,}/{a.size:,} finitos | '
                     f'min={np.nanmin(a):.1f} max={np.nanmax(a):.1f} media={np.nanmean(a):.1f} degC')
            if anc:
                d = a - out_v2_bc[r]
                linha += f' | ancoragem |dT|_med={np.nanmean(np.abs(d)):.3f}'
            print(linha, flush=True)
        print(f'  tempo total={(time.time()-t_ini)/60:.1f} min', flush=True)
    else:
        fin = np.isfinite(out_tf)
        print(f'  T_final: {fin.sum():,}/{out_tf.size:,} finitos | '
              f'min={np.nanmin(out_tf):.1f} max={np.nanmax(out_tf):.1f} '
              f'media={np.nanmean(out_tf):.1f} degC | tempo total={(time.time()-t_ini)/60:.1f} min', flush=True)
        if anc:
            d = out_tf - out_bg
            print(f'  ANCORAGEM: correcao media |dT|={np.nanmean(np.abs(d)):.3f} degC '
                  f'(p50={np.nanmedian(np.abs(d)):.3f} p95={np.nanpercentile(np.abs(d),95):.3f}) '
                  f'| T_final_bc + T_final_oi tambem gravadas', flush=True)


if __name__ == '__main__':
    main()
