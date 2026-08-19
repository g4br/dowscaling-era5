#!/usr/bin/env bash
# ============================================================
# Pipeline de downscaling ERA5-Land -> 1 km — passo a passo.
# Executa os scripts deste diretorio na ordem de dependencia:
#   f0      grade canonica (cop1km, centros, z_std)
#   f1      QC e harmonizacao das estacoes
#   f2a3    HAND via WhiteboxTools (90 m)
#   f2b     agregacao das estaticas 90 m -> 1 km
#   f2c     fracoes termicas 30 m -> 1 km
#   f2d     gate: schema das estaticas vs grade
#   f3setup classificacao das estacoes + celula-contenedora
#   f3      stack mensal ERA5 1 km (paralelo por blocos de meses)
#   f4      gate de consistencia (regra cardinal)
#   f5      matriz de treino
#   f6      ensaio LightGBM (OOF leave-cell-out)
#   f6b     avaliacao do ensaio (tabelas/figuras)
#   f7      modelo final (fonte unica p/ o F8)
#   f7q     [opcional, F7Q=1] modelos segmentados por quantil
#   f8b     benchmark da ancoragem + Lz por hora (json p/ o F8)
#   f8      inferencia na grade 1 km, mes a mes (com ancoragem OI)
#   (f9 e avaliacao/figuras: em scripts_avaliacao/, fora deste repo core)
#
# Para em qualquer erro (set -e). Roda tudo:
#   nohup ./run_pipeline.sh > logs/pipeline.log 2>&1 &
# Retomar a partir de um passo (pula os anteriores):
#   DE=f7 ./run_pipeline.sh
# Variaveis uteis (defaults abaixo): PY, ANO_INI/ANO_FIM (F3),
#   F8_INI/F8_FIM (F8), F3_JOBS, FEATURES, TAG, ANCORA (1|0), F7Q (0|1).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

# Ambiente com rioxarray/lightgbm/scipy/pyarrow (ver requirements.txt).
# Ajuste PY para o interpretador do seu ambiente, ex.:
#   PY=/caminho/para/venv/bin/python ./run_pipeline.sh
PY="${PY:-python}"

# Raiz dos dados (mesma var lida pelos scripts Python via _root.py).
RAIZ="${DOWNSCALING_ROOT:-/dados3/ERA5_land/downscaling}"
export DOWNSCALING_ROOT="$RAIZ"
MODELOS="$RAIZ/saidas/modelos"

# Periodo do stack ERA5 mensal (F3) e da inferencia (F8).
ANO_INI="${ANO_INI:-2020}"
ANO_FIM="${ANO_FIM:-2025}"
F8_INI="${F8_INI:-$ANO_INI}"
F8_FIM="${F8_FIM:-$ANO_FIM}"

# Conjunto de features e tag das saidas do ensaio (casa f6/f6b/f7/f8b/f8).
FEATURES="${FEATURES:-55}"
TAG="${TAG:-_5f55final}"
OOF_ANCORA="${OOF_ANCORA:-$MODELOS/oof_ensaio${TAG}.parquet}"
LZ_JSON="${LZ_JSON:-$MODELOS/bench_ancoragem/lz_por_hora_ensaio${TAG}.json}"
ANCORA="${ANCORA:-1}"          # 1 = F8 com ancoragem OI (hindcast); 0 = downscaling puro
F7Q="${F7Q:-0}"                # 1 = treinar tambem os modelos segmentados (f7q)

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$LOG_DIR/f3" "$LOG_DIR/f8"

# --- runner: DE=<passo> retoma a partir dele (pula os anteriores) ---
DE="${DE:-f0}"
ATIVO=0
passo() {
  local nome="$1"; shift
  [[ "$nome" == "$DE" ]] && ATIVO=1
  if (( ! ATIVO )); then echo "    [pula] $nome"; return 0; fi
  echo; echo ">>> [$nome] $*"
  "$@"
}

# ============================================================
# F0/F1 — grade canonica + QC das estacoes
# ============================================================
passo f0 "$PY" f0_grade_canonica.py
passo f1 "$PY" f1_qc_estacoes.py

# ============================================================
# F2 — estaticas 90 m -> 1 km na grade canonica
# ============================================================
passo f2a3 "$PY" f2a3_hand_wbt.py        # HAND via WhiteboxTools
passo f2b  "$PY" f2b_agrega_1km.py       # agregacao 90 m -> 1 km
passo f2c  "$PY" f2c_fracoes_1km.py      # fracoes termicas 30 m -> cop1km
passo f2d  "$PY" f2d_schema_teste.py     # gate: schema das estaticas vs grade

# ============================================================
# F3 — setup estacoes + stack mensal ERA5 1 km (paralelo por blocos)
# ============================================================
passo f3setup "$PY" f3_setup_estacoes.py

# Cada f3_era5_1km.py usa ~16 GB de RAM; F3_JOBS=4 => ~64 GB de pico (VM 125 GB
# compartilhada; 12 de uma vez estoura e o OOM killer mata o pipeline).
F3_JOBS="${F3_JOBS:-4}"
f3_todos_os_meses() {
  local ano falhou ini mes log rotulo rc
  for ano in $(seq "$ANO_INI" "$ANO_FIM"); do
    echo ">>> F3 ano $ano: ate $F3_JOBS meses simultaneos por bloco..."
    falhou=0
    for ((ini=1; ini<=12; ini+=F3_JOBS)); do
      local pids=() meses=()
      for ((mes=ini; mes<ini+F3_JOBS && mes<=12; mes++)); do
        log="$LOG_DIR/f3/f3_${ano}_$(printf '%02d' "$mes").log"
        "$PY" f3_era5_1km.py "$ano" "$mes" > "$log" 2>&1 &
        pids+=("$!"); meses+=("$mes")
      done
      for i in "${!pids[@]}"; do
        rotulo="${ano}-$(printf '%02d' "${meses[$i]}")"
        if wait "${pids[$i]}"; then
          echo "    ok: $rotulo"
        else
          rc=$?
          echo "    FALHOU: $rotulo (rc=$rc) -- ver $LOG_DIR/f3/f3_${ano}_$(printf '%02d' "${meses[$i]}").log"
          falhou=1
        fi
      done
    done
    [ "$falhou" -eq 0 ] || { echo ">>> F3 abortado: ano $ano teve meses com erro." >&2; return 1; }
    echo ">>> F3 ano $ano concluido."
  done
}
passo f3 f3_todos_os_meses

# nota: f4_motor_extracao.py e MODULO (importado por F5/F7), nao roda sozinho.

# ============================================================
# F4/F5 — gate de consistencia + matriz de treino
# ============================================================
passo f4 "$PY" f4_gate_consistencia.py
passo f5 "$PY" f5_matriz_treino.py

# ============================================================
# F6 — ensaio LightGBM (gera o OOF leave-cell-out) + avaliacao
# ============================================================
passo f6  "$PY" f6_ensaio_lightgbm.py --features "$FEATURES" --tag "$TAG"
passo f6b "$PY" f6b_avalia_ensaio.py --tag "$TAG"

# ============================================================
# F7 — modelo final (unico consumido pelo F8) [+ f7q opcional]
# ============================================================
passo f7 "$PY" f7_treina_final.py --features "$FEATURES"
if [[ "$F7Q" == 1 ]]; then
  passo f7q "$PY" f7q_treina_quantil.py --features "$FEATURES" --early 300
fi

# ============================================================
# F8b — benchmark da ancoragem: valida regra, tuna Lz e gera o json por hora
# ============================================================
passo f8b "$PY" f8b_bench_ancoragem.py --oof "$OOF_ANCORA" \
      --Lz 0,200,300,400,500,600,800,1200 --por-hora

# ============================================================
# F8 — inferencia na grade 1 km, mes a mes (pesado: ~horas por mes)
# ============================================================
f8_todos_os_meses() {
  local ano mes log args=()
  if [[ "$ANCORA" == 1 ]]; then
    args=(--ancora --ancora-oof "$OOF_ANCORA" --ancora-Lz-json "$LZ_JSON")
  fi
  for ano in $(seq "$F8_INI" "$F8_FIM"); do
    for mes in $(seq 1 12); do
      log="$LOG_DIR/f8/f8_${ano}_$(printf '%02d' "$mes").log"
      echo ">>> F8 $ano-$(printf '%02d' "$mes") ${args[*]:-}"
      "$PY" f8_infere_grade.py --ano "$ano" --mes "$mes" "${args[@]}" > "$log" 2>&1 \
        || { echo "    FALHOU: $ano-$mes -- ver $log" >&2; return 1; }
      echo "    ok: $ano-$(printf '%02d' "$mes")"
    done
  done
}
passo f8 f8_todos_os_meses

# ============================================================
# F9 + avaliacao (fora deste repo core)
# ============================================================
# F9 (skill honesto global estacao x grade) e a suite de avaliacao/figuras
# (f13–f32, SHAP f11, plots por estacao) vivem em scripts_avaliacao/ e nao
# fazem parte deste repositorio core do pipeline. Reproduzem a Secao 3
# (Results) do manuscrito a partir das saidas geradas ate o F8.
#   passo f9 "$PY" f9_compara_t2_estacao_grade.py --oof "$OOF_ANCORA"

echo; echo ">>> pipeline concluido."
