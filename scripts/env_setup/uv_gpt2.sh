# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_gpt2 --python 3.12
source uv_gpt2/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

uv pip install -r requirements.txt