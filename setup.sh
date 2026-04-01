#!/usr/bin/env bash
# setup.sh — install HFS into the bundled lmms-eval
#
# Run once after cloning:
#   git clone --recurse-submodules https://github.com/lwpyh/HFS.git
#   cd HFS && bash setup.sh

set -e

HFS_ROOT="$(cd "$(dirname "$0")" && pwd)"
LMMS_DIR="$HFS_ROOT/hfs/lmms-eval"

echo "[setup] HFS root : $HFS_ROOT"
echo "[setup] lmms-eval: $LMMS_DIR"

if [ ! -f "$LMMS_DIR/setup.py" ] && [ ! -f "$LMMS_DIR/pyproject.toml" ]; then
    echo "[setup] lmms-eval submodule not initialised. Running:"
    echo "        git submodule update --init --recursive"
    git -C "$HFS_ROOT" submodule update --init --recursive
fi

# 1. Install lmms-eval
echo "[setup] Installing lmms-eval..."
pip install -e "$LMMS_DIR" -q

# 2. Copy model plugin
echo "[setup] Copying qwen2_5_vl_hfs.py into lmms-eval..."
cp "$HFS_ROOT/hfs/qwen2_5_vl_hfs.py" \
   "$LMMS_DIR/lmms_eval/models/chat/qwen2_5_vl_hfs.py"

# 3. Register model in lmms-eval __init__.py
echo "[setup] Registering qwen2_5_vl_hfs model..."
python3 - <<PYEOF
import pathlib, re

init_path = pathlib.Path("$LMMS_DIR/lmms_eval/models/__init__.py")
content = init_path.read_text()
entry = '"qwen2_5_vl_hfs": "Qwen2_5_VL_HFS",'

if entry in content:
    print("[setup] Already registered.")
else:
    # Insert right after the qwen2_5_vl entry
    content = re.sub(
        r'("qwen2_5_vl"\s*:\s*"Qwen2_5_VL"\s*,)',
        r'\1\n    ' + entry,
        content,
    )
    init_path.write_text(content)
    print("[setup] Registered qwen2_5_vl_hfs.")
PYEOF

# 4. Install additional HFS dependencies
echo "[setup] Installing HFS dependencies..."
pip install -q \
    "transformers>=4.45" \
    qwen-vl-utils \
    accelerate \
    opencv-python

echo ""
echo "[setup] Done. Run HFS with:"
echo "  cd hfs/lmms-eval"
echo "  PYTHONPATH=\"\$PWD/../..\" accelerate launch -m lmms_eval \\"
echo "    --model qwen2_5_vl_hfs \\"
echo "    --model_args \"pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=6422528,attn_implementation=flash_attention_2,method=hfs\" \\"
echo "    --tasks videomme --batch_size 1"
