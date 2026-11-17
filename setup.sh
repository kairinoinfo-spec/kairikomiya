#!/bin/bash
# 切り抜き機能（clip.py）の準備をします。最初の1回だけ実行してください。
#   使い方:  bash setup.sh
set -e
cd "$(dirname "$0")"

# Homebrew を入れた直後でも見つけられるように
if ! command -v brew >/dev/null 2>&1 && [ -x /opt/homebrew/bin/brew ]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "✖ Homebrew が見つかりません。先に README の「準備 2. Homebrew を入れる」を行ってください。"
  exit 1
fi

echo "▶ 必要なツールを入れています（ffmpeg / whisper-cpp / python）"
brew install ffmpeg whisper-cpp python

PACKAGES=(anthropic "opencv-python-headless>=4.10,<5")

make_venv() {
  rm -rf .venv
  "$1" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet --only-binary=:all: "${PACKAGES[@]}"
}

echo "▶ このツール専用の Python 環境を作っています"
if ! make_venv "$(brew --prefix)/bin/python3"; then
  # 出たばかりの Python に、部品がまだ対応していないことがあるので、1つ前の版で作り直す
  echo "▶ 最新の Python では部品が入らなかったので、Python 3.13 で作り直します"
  brew install python@3.13
  make_venv "$(brew --prefix python@3.13)/bin/python3.13"
fi

echo "✔ 準備ができました"
