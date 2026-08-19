#!/usr/bin/env python3

# ============================================================
# F3 — stack mensal ERA5 1 km (Opcao B: grade 560x807 populada SO nas
# celulas-contenedoras das estacoes; resto NaN -> comprime).
# Grid-first: .interp bilinear no CENTRO da celula (centros_wgs84/celulas_estacoes),
# nunca na lon/lat exata. De-acum reset diario (h01=cru). valid_time->time.
# Uso: python f3_era5_1km.py <ano> <mes>
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import calendar
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray   # noqa: F401
from _root import ROOT, DIR_GRID


# ============================================================
# CONFIGURACOES
# ============================================================

ERA5       = Path(ROOT + '/Dados/ERA5_land')
PASTA_GRID = DIR_GRID
# SUF escolhe o npz de celulas E sufixa a pasta de saida (mesmo padrao do f3_setup_estacoes).
# SUF=_exclusao -> popula as celulas da aba Exclusao em pasta PROPRIA: os mensais que
# f5_matriz_treino/f8_infere_grade leem ficam intactos (a grade e esparsa, escrever outro
# conjunto de celulas nos MESMOS arquivos zeraria as 206 originais).
SUF        = os.environ.get('SUF', '')
PASTA_OUT  = Path(ROOT + f'/saidas/era5land_interp1km_celulas{SUF}')

ARQ_TERMICO = 't2m/ERA5land_{ano}_{mes:02d}.nc'        # t2m, skt, stl1, slhf, sshf
ARQ_VAR     = 'd2m/ERA5land_umid_vento_sp_{ano}_{mes:02d}.nc'    # d2m, u10, v10, sp
ARQ_RAD     = 'rad/ERA5land_radiacao_{ano}_{mes:02d}.nc'    # ssr, str
ARQ_OROG_1KM = PASTA_GRID / 'z_era5_orog_1km.tif'

LAPSE_SECO = 0.0065
VARS_3D = ['t2m_1km', 'lapse', 'u10', 'v10', 'vel_vento', 'sp',
           'dep_orvalho', 'umidade_relativa', 'dt_ar_pele', 'dt_ar_solo',
           'rad_liquida', 'razao_bowen', 'ssr', 'str']   # ssr/str individuais (alem da rad_liquida=ssr+str)
VARS_2D = ['z_era5_orog', 'delta_z']


# ============================================================
# FUNCOES
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
        raise KeyError('Variavel "' + nome + '" ausente em ' + str(caminho) + ' (' + str(list(ds.variables)) + ')')
    return ds[nome]


def desacumular_fluxo(da):
    # de-acum com reset diario (h01=cru); reindex ao eixo completo (1o 00:00 -> NaN)
    horas    = da['time'].dt.hour
    da_diff  = da.diff('time')
    da_fluxo = xr.where(horas[1:] == 1, da.isel(time=slice(1, None)), da_diff) / 3600.0
    return da_fluxo.reindex(time=da['time'])


def _fallback_nearest_terra(da, lat_cel, lon_cel, out, agua):
    # out (nt, cel); agua (cel,) bool = celulas com stencil 100% agua (den==0).
    # Atribui a serie do no de TERRA nativo mais proximo (mascara estatica).
    latn = da['latitude'].values; lonn = da['longitude'].values
    mask = np.isfinite(da.values).any(axis=0)            # (lat,lon) terra = finito em algum t
    jy, jx = np.where(mask)
    vlat = latn[jy]; vlon = lonn[jx]
    for c in np.where(agua)[0]:
        d = ((vlon - lon_cel[c]) * np.cos(np.deg2rad(lat_cel[c]))) ** 2 + (vlat - lat_cel[c]) ** 2
        k = int(np.argmin(d))
        out[:, c] = da.values[:, jy[k], jx[k]]
    return out


def interp_celulas(da, lat_cel, lon_cel):
    # bilinear NaN-AWARE no centro das celulas-contenedoras -> (..., cel).
    # ERA5-Land mascara agua como NaN (~9km); o bilinear puro contamina
    # celulas costeiras com NaN. Renormaliza os pesos sobre os nos de terra
    # validos (num/den, mesma filosofia do average NaN-aware da F2) — segue
    # sendo "bilinear uma vez". Fallback nearest-terra so se stencil 100% agua.
    lat_da = xr.DataArray(np.asarray(lat_cel), dims='cel')
    lon_da = xr.DataArray(np.asarray(lon_cel), dims='cel')
    valido = da.notnull()
    num = da.fillna(0.0).interp(latitude=lat_da, longitude=lon_da, method='linear')
    den = valido.astype('float64').interp(latitude=lat_da, longitude=lon_da, method='linear')
    out = (num / den.where(den > 0)).values              # sempre (nt, cel) em F3
    agua = ~np.isfinite(out).any(axis=0)                 # NaN em TODOS os t = den==0
    if agua.any():
        out = _fallback_nearest_terra(da, np.asarray(lat_cel), np.asarray(lon_cel), out, agua)
    return out


def umidade_relativa(temp_c, orvalho_c):
    es = 6.112 * np.exp((17.67 * temp_c)    / (temp_c    + 243.5))
    e  = 6.112 * np.exp((17.67 * orvalho_c) / (orvalho_c + 243.5))
    return 100.0 * e / es


# ============================================================
# ENTRADAS
# ============================================================

if len(sys.argv) != 3:
    raise ValueError('Uso: python f3_era5_1km.py <ano> <mes>')
ano = int(sys.argv[1]); mes = int(sys.argv[2])

c_term = ERA5 / ARQ_TERMICO.format(ano=ano, mes=mes)
c_var  = ERA5 / ARQ_VAR.format(ano=ano, mes=mes)
c_rad  = ERA5 / ARQ_RAD.format(ano=ano, mes=mes)
for c in (c_term, c_var, c_rad):
    if not c.exists():
        raise FileNotFoundError('ERA5 ausente: ' + str(c))
if not ARQ_OROG_1KM.exists():
    raise FileNotFoundError('Orografia 1km ausente (rodar A0): ' + str(ARQ_OROG_1KM))
PASTA_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# GRADE E CELULAS DAS ESTACOES
# ============================================================

print('Lendo grade e celulas das estacoes...')
cel = np.load(PASTA_GRID / f'celulas_estacoes{SUF}.npz')
iy, ix = cel['iy'], cel['ix']
lat_cel, lon_cel = cel['cell_lat'], cel['cell_lon']

da_z_alvo = rioxarray.open_rasterio(PASTA_GRID / 'cop1km.tif', masked=True).squeeze()
xs = da_z_alvo['x'].values; ys = da_z_alvo['y'].values
ny, nx = len(ys), len(xs)

z_orog_full = rioxarray.open_rasterio(ARQ_OROG_1KM, masked=True).squeeze().values
z_orog_cel  = z_orog_full[iy, ix]
delta_z_cel = da_z_alvo.values[iy, ix] - z_orog_cel


# ============================================================
# LEITURA ERA5 + DE-ACUM
# ============================================================

print('Abrindo ERA5', ano, mes, '(valid_time->time, drop expver/number)...')
ds_term = abrir_era5(c_term)
ds_var  = abrir_era5(c_var)
ds_rad  = abrir_era5(c_rad)

# Eixo de tempo CANONICO (horario, do inicio do mes). Robustez contra fonte
# com 'time' nao-decodificavel: bug observado em era5land_interp1km_celulas_2021_01.nc, cuja
# variavel-coordenada 'time' nao foi materializada -> indices 0..743 sem
# unidade CF, fazendo F5 descartar jan/2021 inteiro. Validar a contagem antes
# de relabelar p/ nunca desalinhar o tempo silenciosamente.
nt = ds_term.sizes['time']
horas_mes = calendar.monthrange(ano, mes)[1] * 24
if nt != horas_mes:
    raise ValueError(f'ERA5 {ano}-{mes:02d}: {nt} passos de tempo != {horas_mes}h '
                     f'do mes (fonte incompleta) — abortar p/ nao desalinhar o tempo.')
tempos = pd.date_range(f'{ano}-{mes:02d}-01', periods=nt, freq='h')

src_vals = np.asarray(ds_term['time'].values)
if np.issubdtype(src_vals.dtype, np.datetime64):
    if not pd.DatetimeIndex(src_vals).equals(tempos):           # CF presente: tem de bater
        raise ValueError(f'ERA5 {ano}-{mes:02d}: eixo de tempo da fonte diverge do '
                         f'horario canonico (offset/gap) — abortar.')
else:                                                           # sem eixo CF (bug jan/2021):
    print(f'  AVISO: fonte sem eixo de tempo CF; adotando horario canonico '   # casar ANTES
          f'{tempos[0]} -> {tempos[-1]}')                       # da de-acum (.dt.hour) e da escrita
    ds_term = ds_term.assign_coords(time=tempos)
    ds_var  = ds_var.assign_coords(time=tempos)
    ds_rad  = ds_rad.assign_coords(time=tempos)

print('De-acumulando fluxos (reset h01)...')
ssr_w  = desacumular_fluxo(exigir(ds_rad,  'ssr',  c_rad))
str_w  = desacumular_fluxo(exigir(ds_rad,  'str',  c_rad))
sshf_w = desacumular_fluxo(exigir(ds_term, 'sshf', c_term))
slhf_w = desacumular_fluxo(exigir(ds_term, 'slhf', c_term))


# ============================================================
# INTERPOLAR NO CENTRO DAS CELULAS (time, cel)
# ============================================================

print('Interpolando brutas no centro das', len(iy), 'celulas...')
t2m  = interp_celulas(exigir(ds_term, 't2m',  c_term), lat_cel, lon_cel)
skt  = interp_celulas(exigir(ds_term, 'skt',  c_term), lat_cel, lon_cel)
stl1 = interp_celulas(exigir(ds_term, 'stl1', c_term), lat_cel, lon_cel)
d2m  = interp_celulas(exigir(ds_var,  'd2m',  c_var),  lat_cel, lon_cel)
u10  = interp_celulas(exigir(ds_var,  'u10',  c_var),  lat_cel, lon_cel)
v10  = interp_celulas(exigir(ds_var,  'v10',  c_var),  lat_cel, lon_cel)
sp   = interp_celulas(exigir(ds_var,  'sp',   c_var),  lat_cel, lon_cel) / 100.0
ssr  = interp_celulas(ssr_w,  lat_cel, lon_cel)
strn = interp_celulas(str_w,  lat_cel, lon_cel)
sshf = interp_celulas(sshf_w, lat_cel, lon_cel)
slhf = interp_celulas(slhf_w, lat_cel, lon_cel)


# ============================================================
# DERIVADAS NA GRADE (time, cel)
# ============================================================

print('Derivando na grade (cel)...')
delta_z_bc = delta_z_cel[None, :]
vel_vento = np.sqrt(u10 ** 2 + v10 ** 2)
lapse     = t2m - LAPSE_SECO * delta_z_bc
ur        = umidade_relativa(t2m - 273.15, d2m - 273.15)
dep_orv   = t2m - d2m
dt_pele   = t2m - skt
dt_solo   = t2m - stl1
rad_liq   = ssr + strn
bowen     = np.where(np.abs(slhf) < 1.0, np.nan, sshf / slhf)

dados_3d = {'t2m_1km': t2m, 'lapse': lapse, 'u10': u10, 'v10': v10,
            'vel_vento': vel_vento, 'sp': sp, 'dep_orvalho': dep_orv,
            'umidade_relativa': ur, 'dt_ar_pele': dt_pele, 'dt_ar_solo': dt_solo,
            'rad_liquida': rad_liq, 'razao_bowen': bowen,
            'ssr': ssr, 'str': strn}   # radiacao solar (onda-curta) e termal (onda-longa) individuais


# ============================================================
# ESPALHAR EM GRADE 560x807 (NaN fora das celulas) E SALVAR
# ============================================================

print('Espalhando em grade', ny, 'x', nx, 'e montando dataset...')
data_vars = {}
for nome in VARS_3D:
    arr = np.full((nt, ny, nx), np.nan, dtype='float32')
    arr[:, iy, ix] = dados_3d[nome].astype('float32')
    data_vars[nome] = (('time', 'y', 'x'), arr)

for nome, vetor in (('z_era5_orog', z_orog_cel), ('delta_z', delta_z_cel)):
    arr2 = np.full((ny, nx), np.nan, dtype='float32')
    arr2[iy, ix] = vetor.astype('float32')
    data_vars[nome] = (('y', 'x'), arr2)

ds_mes = xr.Dataset(data_vars, coords={'time': tempos, 'y': ys, 'x': xs})
ds_mes.rio.write_crs('EPSG:31982', inplace=True)

enc = {v: {'zlib': True, 'complevel': 4} for v in (VARS_3D + VARS_2D)}
saida = PASTA_OUT / ('era5land_interp1km_celulas_%d_%02d.nc' % (ano, mes))
ds_mes.to_netcdf(saida, encoding=enc)

print('Stack salvo:', saida)
print('  vars 3D:', len(VARS_3D), '| vars 2D:', len(VARS_2D), '| nt:', nt, '| celulas pop.:', len(iy))
