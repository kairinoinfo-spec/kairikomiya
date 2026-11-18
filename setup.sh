#!/bin/bash
# 切り抜き機能（clip.py）の準備をします。
# 最初の1回と、ツールを新しくしたときに実行してください（2回目以降はすぐ終わります）。
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

# ツールを新しくダウンロードし直しても消えない場所に置く
APP_DIR="$HOME/Library/Application Support/kairikomiya"
VENV="$APP_DIR/venv"
mkdir -p "$APP_DIR"
# AI / 人物の位置探し / 文字を描く / 日本語の改行位置
PACKAGES=(anthropic "opencv-python-headless>=4.10,<5" pillow budoux)

make_venv() {
  rm -rf "$VENV"
  "$1" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet --only-binary=:all: "${PACKAGES[@]}"
}

echo "▶ このツール専用の Python 環境を作っています"
if [ -x "$VENV/bin/pip" ] && "$VENV/bin/pip" install --quiet --only-binary=:all: --upgrade "${PACKAGES[@]}"; then
  :  # すでにある環境に、足りない部品だけ入れた
elif ! make_venv "$(brew --prefix)/bin/python3"; then
  # 出たばかりの Python に、部品がまだ対応していないことがあるので、1つ前の版で作り直す
  echo "▶ 最新の Python では部品が入らなかったので、Python 3.13 で作り直します"
  brew install python@3.13
  make_venv "$(brew --prefix python@3.13)/bin/python3.13"
fi

# 前の版で、ツールのフォルダの中に作った環境は、もう使わないので消す
rm -rf .venv

echo "✔ 準備ができました"
