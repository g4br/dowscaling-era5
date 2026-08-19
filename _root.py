"""Raizes de dados do pipeline, configuraveis via ambiente.

Ha duas raizes:

1. ROOT — dados BRUTOS nao redistribuidos (ERA5-Land, estacoes, DEM, MapBiomas).
   Aponte para o seu layout exportando a variavel antes de rodar:

       export DOWNSCALING_ROOT=/caminho/para/seus/dados

   Esperado abaixo dela: Dados/ERA5_land/..., Dados/MDTs/..., stn_data/...,
   saidas/... (stacks ERA5 1 km e modelos ficam aqui).

2. REPO — o proprio repositorio. Os artefatos DERIVADOS que NAO contem dados
   brutos de ERA5/estacoes (grade canonica, estaticas 1 km e matriz de treino)
   vem empacotados em dados/, para o repo ser auto-suficiente exceto pelos
   dados brutos de ERA5 e das estacoes. Assim F5->F8 rodam sem regenerar F0-F2.

Os tres diretorios derivados abaixo apontam por padrao para dados/ do repo.
Sobrescreva por ambiente (DIR_GRID/DIR_ESTATICAS/DIR_MATRIZ) para outro layout.
Como os geradores (F0, F2b, F2c, F5) tambem escrevem nesses diretorios, apontar
para dados/ regenera os arquivos empacotados no lugar.
"""
import os
from pathlib import Path

ROOT = os.environ.get('DOWNSCALING_ROOT', '/dados3/ERA5_land/downscaling')

REPO = Path(__file__).resolve().parent

DIR_GRID      = Path(os.environ.get('DIR_GRID',      REPO / 'dados' / 'grid_1km'))
DIR_ESTATICAS = Path(os.environ.get('DIR_ESTATICAS', REPO / 'dados' / 'estaticas_1km'))
DIR_MATRIZ    = Path(os.environ.get('DIR_MATRIZ',    REPO / 'dados' / 'matriz'))
