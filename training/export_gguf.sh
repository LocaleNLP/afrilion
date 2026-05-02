#!/usr/bin/env bash
# AfriLION GGUF Export + Ollama Setup
#
# Converts a merged HuggingFace AfriLION model to GGUF format and
# publishes it to Ollama registry for `ollama run afrilion/afrilion-1b`.
#
# WHY OLLAMA?
# The local LLM community is enormous — millions of developers run models
# on their own hardware. "ollama run afrilion/afrilion-1b" is the easiest
# possible user experience. The HF Spaces demo gets researchers.
# Ollama gets developers. This takes half a day to set up.
#
# PREREQUISITES:
#   1. llama.cpp installed (https://github.com/ggerganov/llama.cpp)
#   2. Ollama installed (https://ollama.com)
#   3. Merged model weights at $MODEL_PATH (run merge_lora.py first)
#   4. Ollama account + `ollama login`
#
# USAGE:
#   bash export_gguf.sh                         # uses defaults
#   bash export_gguf.sh --model ./merged_model   # custom model path
#   bash export_gguf.sh --quantization Q4_K_M    # custom quantization
#   bash export_gguf.sh --push                   # push to Ollama registry

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH="./merged_model"           # path to merged HF model (post LoRA merge)
GGUF_OUTPUT_DIR="./gguf_output"       # where to write GGUF files
QUANTIZATION="Q4_K_M"                 # quantization type (Q4_K_M recommended)
OLLAMA_MODEL_NAME="afrilion/afrilion-1b"  # ollama registry name
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"  # path to llama.cpp
PUSH_TO_OLLAMA=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL_PATH="$2";         shift 2 ;;
    --output)       GGUF_OUTPUT_DIR="$2";    shift 2 ;;
    --quantization) QUANTIZATION="$2";      shift 2 ;;
    --name)         OLLAMA_MODEL_NAME="$2"; shift 2 ;;
    --llama-cpp)    LLAMA_CPP_DIR="$2";     shift 2 ;;
    --push)         PUSH_TO_OLLAMA=true;    shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "====================================================="
echo " AfriLION GGUF Export"
echo "====================================================="
echo "  Model:          $MODEL_PATH"
echo "  Output dir:     $GGUF_OUTPUT_DIR"
echo "  Quantization:   $QUANTIZATION"
echo "  Ollama name:    $OLLAMA_MODEL_NAME"
echo "  Push to Ollama: $PUSH_TO_OLLAMA"
echo "====================================================="

# ---------------------------------------------------------------------------
# Step 1: Validate prerequisites
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Checking prerequisites..."

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  echo "  Run training/merge_lora.py first to merge LoRA weights into base model."
  exit 1
fi

if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
  echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR"
  echo "  Install: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make"
  exit 1
fi

if ! command -v ollama &> /dev/null; then
  echo "ERROR: ollama not found in PATH"
  echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

mkdir -p "$GGUF_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 2: Convert HF model to GGUF (fp16 first)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Converting HuggingFace model to GGUF (fp16)..."

GGUF_FP16="$GGUF_OUTPUT_DIR/afrilion-1b-fp16.gguf"

python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MODEL_PATH" \
  --outfile "$GGUF_FP16" \
  --outtype f16

echo "  fp16 GGUF: $GGUF_FP16"

# ---------------------------------------------------------------------------
# Step 3: Quantize GGUF to Q4_K_M (or specified quantization)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Quantizing GGUF to $QUANTIZATION..."

# Quantization type guide:
#   Q4_K_M  — Best quality/size tradeoff for deployment (~800MB for 1B model)
#   Q5_K_M  — Higher quality (~1GB), good for API servers with more RAM
#   Q8_0    — Near fp16 quality (~1.8GB), for high-quality inference
#   Q2_K    — Smallest (~500MB), quality degrades noticeably
GGUF_QUANTIZED="$GGUF_OUTPUT_DIR/afrilion-1b-${QUANTIZATION}.gguf"

"$LLAMA_CPP_DIR/llama-quantize" \
  "$GGUF_FP16" \
  "$GGUF_QUANTIZED" \
  "$QUANTIZATION"

echo "  Quantized GGUF: $GGUF_QUANTIZED"
echo "  Size: $(du -sh "$GGUF_QUANTIZED" | cut -f1)"

# ---------------------------------------------------------------------------
# Step 4: Create Ollama Modelfile
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Creating Ollama Modelfile..."

MODELFILE="$GGUF_OUTPUT_DIR/Modelfile"

cat > "$MODELFILE" << 'MODELFILE_EOF'
FROM ./afrilion-1b-Q4_K_M.gguf

# AfriLION — African Language Instruction-Following Model
# Built by LocaleNLP | https://github.com/LocaleNLP/afrilion
#
# Supports: Swahili, Wolof, Hausa, Yoruba, Amharic, Zulu, Xhosa,
#           Igbo, Somali, Tigrinya, Shona, Luganda, Twi + more

SYSTEM """
You are AfriLION, an AI assistant specialized in African languages.
You can respond in Swahili, Wolof, Hausa, Yoruba, Amharic, Zulu, Xhosa, Igbo,
Somali, Tigrinya, Shona, Luganda, Twi, and many other African languages.
When a user writes in an African language, respond in that language.
You are helpful, culturally aware, and knowledgeable about African contexts.
Built by LocaleNLP. Phase 0 — model weights coming soon.
"""

# Chat template matches training format
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>
"""

# Inference parameters tuned for African language instruction following
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# Stop tokens
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
MODELFILE_EOF

# Update Modelfile to reference correct quantized GGUF
sed -i "s|./afrilion-1b-Q4_K_M.gguf|./afrilion-1b-${QUANTIZATION}.gguf|g" "$MODELFILE"

echo "  Modelfile: $MODELFILE"

# ---------------------------------------------------------------------------
# Step 5: Create local Ollama model + optionally push to registry
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Creating local Ollama model..."

# Create the model locally
(cd "$GGUF_OUTPUT_DIR" && ollama create "$OLLAMA_MODEL_NAME" -f Modelfile)

echo ""
echo "✅ Local Ollama model created successfully!"
echo ""
echo "  Test it:"
echo "    ollama run $OLLAMA_MODEL_NAME 'Habari za asubuhi?'"
echo "    ollama run $OLLAMA_MODEL_NAME 'Na nga def?'"
echo "    ollama run $OLLAMA_MODEL_NAME 'Translate Hello to Yoruba'"
echo ""

if [ "$PUSH_TO_OLLAMA" = true ]; then
  echo "Pushing to Ollama registry..."
  echo "  (Requires: ollama login)"
  ollama push "$OLLAMA_MODEL_NAME"
  echo ""
  echo "✅ Model published to Ollama registry!"
  echo "  Users can now run:"
  echo "    ollama run $OLLAMA_MODEL_NAME"
  echo ""
  echo "  Ollama model page:"
  echo "    https://ollama.com/$(echo $OLLAMA_MODEL_NAME | cut -d/ -f1)/$(echo $OLLAMA_MODEL_NAME | cut -d/ -f2)"
else
  echo "  To push to Ollama registry:"
  echo "    ollama login"
  echo "    bash export_gguf.sh --push"
fi

echo ""
echo "Next steps:"
echo "  1. Push to Ollama: bash export_gguf.sh --push"
echo "  2. Upload GGUF to HuggingFace: huggingface-cli upload AfriLION/afrilion-1b-GGUF $GGUF_OUTPUT_DIR"
echo "  3. Start API server: cd serve && uvicorn api_server:app --host 0.0.0.0 --port 8000"
