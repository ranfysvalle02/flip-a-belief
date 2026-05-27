#!/usr/bin/env bash
# merge_to_gguf.sh - LoRA adapter -> merged HF -> GGUF -> ollama create.
#
#   bash merge_to_gguf.sh outputs/adapters/acid-flip-100
#
# Produces an ollama model named `acid-llama32-<N>` where <N> is the
# doc count parsed from the adapter dir (e.g. `acid-flip-100` -> `100`).
# Clones llama.cpp into ./llama.cpp on first run (~60 MB) and reuses it
# afterwards. Requires `ollama` on PATH and `ollama serve` running.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <adapter_dir> [outtype]"
  echo "Example: $0 outputs/adapters/acid-flip-100"
  echo "         $0 outputs/adapters/acid-flip-100 q8_0"
  exit 1
fi

ADAPTER_DIR="${1%/}"
OUTTYPE="${2:-f16}"

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "  [error] adapter dir does not exist: $ADAPTER_DIR"
  exit 1
fi

ADAPTER_NAME="$(basename "$ADAPTER_DIR")"            # e.g. acid-flip-100
# Strip the known prefix; fall back to the trailing numeric tail otherwise.
N="${ADAPTER_NAME#acid-flip-}"
if [ "$N" = "$ADAPTER_NAME" ]; then
  N="${ADAPTER_NAME##*-}"
fi
MERGED_DIR="outputs/merged-${ADAPTER_NAME}"
GGUF_PATH="${MERGED_DIR}/model.gguf"
OLLAMA_TAG="acid-llama32-${N}"

# 1. peft merge_and_unload -> standalone HF model dir.
echo
echo "=== 1/4  MERGING LORA INTO BASE WEIGHTS ==="
python merge_adapter.py "$ADAPTER_DIR" --output "$MERGED_DIR"

# 2. Clone llama.cpp on first run, install its conversion deps.
if [ ! -d "llama.cpp" ]; then
  echo
  echo "=== 2/4  CLONING llama.cpp (one-time, ~60 MB) ==="
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
  pip install --quiet -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
else
  echo
  echo "=== 2/4  REUSING ./llama.cpp ==="
fi

# 3. HF -> GGUF. f16 is the high-fidelity intermediate; q8_0 etc. work too.
echo
echo "=== 3/4  CONVERTING HF -> GGUF (${OUTTYPE}) ==="
python llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" \
  --outfile "$GGUF_PATH" \
  --outtype "$OUTTYPE"

# 4. Write a minimal Llama-3 instruct Modelfile + ollama create.
echo
echo "=== 4/4  CREATING OLLAMA MODEL '${OLLAMA_TAG}' ==="
MODELFILE="${MERGED_DIR}/Modelfile"
cat > "$MODELFILE" <<'EOF'
FROM ./model.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0
EOF

( cd "$MERGED_DIR" && ollama create "$OLLAMA_TAG" -f Modelfile )

echo
echo "=== DONE ==="
echo "  ollama model:  ${OLLAMA_TAG}"
echo "  try it:        ollama run ${OLLAMA_TAG}"
echo "  showcase:      python showcase.py --live"
