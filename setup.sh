#!/bin/bash
# 切り抜き機能（clip.py）の準備をします。最初の1回だけ実行してください。
#   使い方:  bash setup.sh
set -e
cd "$(dirname "$0")"

if ! command -v brew >/dev/null 2>&1; then
  echo "✖ Homebrew が見つかりません。先に README の「準備 2. Homebrew を入れる」を行ってください。"
  exit 1
fi

echo "▶ 必要なツールを入れています（ffmpeg / whisper-cpp / python）"
brew install ffmpeg whisper-cpp python

echo "▶ このツール専用の Python 環境を作っています"
"$(brew --prefix)/bin/python3" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip anthropic

echo "✔ 準備ができました"
