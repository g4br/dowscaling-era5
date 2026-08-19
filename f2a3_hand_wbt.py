#!/usr/bin/env python3

# ============================================================
# F2-A3 — HAND (Height Above Nearest Drainage) via WhiteboxTools
# Receita B.5 do caderno: breach -> D8 pointer -> flow accum -> streams -> elevation above stream
#
# DIVERGENCIA DELIBERADA da receita B.5: o passo final (ElevationAboveStream)
# usa o DEM CONDICIONADO (dem_breach.tif), NAO o DEM bruto. O ElevationAboveStream
# traca o caminho de fluxo a partir do DEM que recebe; no DEM bruto (cheio de pits
# do Copernicus GLO-90) o caminho morre num pit antes de cruzar um stream -> NoData
# (~73% NaN/terra). A doc do WBT exige DEM hidrologicamente condicionado aqui.
#
# Idempotente: pula breach/d8/facc/streams se o output ja existe; o HAND SEMPRE
# recalcula. Para forcar o recalculo da cadeia (ex.: mudar BREACH_DIST/THRESH_STREAMS),
# apague os intermediarios em PASTA_FEIC.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import whitebox
from _root import ROOT


# ============================================================
# CONFIGURACOES
# ============================================================

PASTA_FEIC = Path(ROOT + '/Dados/MDTs/feicoes')
DEM        = ROOT + '/Dados/MDTs/dem_90m_utm.tif'
THRESH_STREAMS = 2000     # celulas de fluxo p/ iniciar canal (~16,5 km2 a 90,9 m)
BREACH_DIST    = 200      # distancia max de breach (celulas)


# ============================================================
# SETUP WBT
# ============================================================

wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(str(PASTA_FEIC))
wbt.set_verbose_mode(False)


# ============================================================
# CADEIA HIDROLOGICA (idempotente; HAND sempre recalcula)
# ============================================================

def existe(nome):
    return (PASTA_FEIC / nome).exists()

if existe('dem_breach.tif'):
    print('1/5 breach depressions: reusando dem_breach.tif existente')
else:
    print('1/5 breach depressions (least cost)...')
    wbt.breach_depressions_least_cost(dem=DEM, output='dem_breach.tif', dist=BREACH_DIST)

if existe('d8.tif'):
    print('2/5 D8 pointer: reusando d8.tif existente')
else:
    print('2/5 D8 pointer...')
    wbt.d8_pointer(dem=str(PASTA_FEIC / 'dem_breach.tif'), output='d8.tif')

if existe('facc.tif'):
    print('3/5 D8 flow accumulation: reusando facc.tif existente')
else:
    print('3/5 D8 flow accumulation...')
    wbt.d8_flow_accumulation(i=str(PASTA_FEIC / 'd8.tif'), output='facc.tif', pntr=True)

if existe('streams.tif'):
    print('4/5 extract streams: reusando streams.tif existente')
else:
    print('4/5 extract streams (threshold=%d)...' % THRESH_STREAMS)
    wbt.extract_streams(flow_accum=str(PASTA_FEIC / 'facc.tif'), output='streams.tif', threshold=THRESH_STREAMS)

# HAND: usa o DEM CONDICIONADO (dem_breach.tif), nao o bruto — vide cabecalho.
print('5/5 elevation above stream (HAND) sobre dem_breach.tif...')
wbt.elevation_above_stream(dem=str(PASTA_FEIC / 'dem_breach.tif'),
                           streams=str(PASTA_FEIC / 'streams.tif'), output='hand_90m.tif')

print('HAND salvo em:', PASTA_FEIC / 'hand_90m.tif')
