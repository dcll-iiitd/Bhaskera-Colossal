#!/usr/bin/env bash
set -e

PYTHON_VERSION=3.10
VENV_DIR=.venv

echo "🔍 Checking for uv..."

if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "🐍 Creating virtual environment..."
uv venv --python ${PYTHON_VERSION} ${VENV_DIR}

echo "✅ Activating virtual environment..."
source ${VENV_DIR}/bin/activate

echo "📥 Installing dependencies..."
uv pip install -r requirements.txt

echo "📦 Installing framework in editable mode..."
uv pip install -e .

echo "🎉 Setup complete!"
