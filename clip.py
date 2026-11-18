#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clip — 長い動画から、ショート動画の切り抜き候補を探すツール（第1段階: 喋りの動画）

使い方:
    python3 clip.py 動画ファイル.mp4

流れ:  文字起こし（Whisper）→ AI（Claude）が候補を選ぶ → 候補の一覧（CSV）を作る

    python3 clip.py 動画ファイル.mp4 --export tate yoko    … 一覧で「はい」の候補を書き出す

AI の使い方は2通り:
  - 無料: Claude アプリに自分で貼り付ける（api_key.txt が無いときはこちら）
  - 自動: Claude API を使う（api_key.txt があるとき。少額の料金がかかる）
準備:  bash setup.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import autocut
from autocut import AutocutError

HERE = Path(__file__).resolve().parent
# ツールを新しくダウンロードし直しても消えないように、大きなものはここに置く
if platform.system() == "Darwin":
    APP_DIR = Path.home() / "Library" / "Application Support" / "kairikomiya"
else:
    APP_DIR = Path.home() / ".local" / "share" / "kairikomiya"
VENVS = [APP_DIR / "venv", HERE / ".venv"]
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"

# 料金の目安を表示するための単価（100万トークンあたりのドル: 入力, 出力）
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.1, 0.5),
    "claude-fable-5-2": (8.0, 40.0),
}
YEN_PER_DOLLAR = 150
# AI が安全上の理由で応答を止めたとき、別モデルで自動的にやり直す機能に対応しているモデル
FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")

CSV_COLUMNS = ["番号", "使う", "開始", "終了", "長さ(秒)", "形", "位置", "タイトル案", "冒頭のセリフ",
               "要約", "選んだ理由", "説明文", "ハッシュタグ"]

# 書き出す形: 名前 → (ファイル名に付ける言葉, 幅, 高さ)
SHAPES = {
    "tate": ("縦", 1080, 1920),      # 9:16  ショート / TikTok / リール
    "yoko": ("横", 1920, 1080),      # 16:9  YouTube
    "square": ("正方形", 1080, 1080),  # 1:1   Instagram フィード
    "feed": ("4x5", 1080, 1350),     # 4:5   Instagram フィード
}
SHAPE_ALIASES = {"縦": "tate", "9:16": "tate", "横": "yoko", "16:9": "yoko",
                 "正方形": "square", "1:1": "square", "4:5": "feed", "4x5": "feed"}

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_id": {"type": "integer", "description": "切り抜きの最初の行の番号"},
                    "end_id": {"type": "integer", "description": "切り抜きの最後の行の番号（この行まで含む）"},
                    "title": {"type": "string", "description": "動画の冒頭に大きく出すタイトル。20文字以内"},
                    "summary": {"type": "string", "description": "内容の要約。1〜2文"},
                    "reason": {"type": "string", "description": "なぜここを選んだか。1〜2文"},
                    "description": {"type": "string", "description": "投稿に添える説明文。2〜3文"},
                    "hashtags": {"type": "array", "items": {"type": "string"},
                                 "description": "ハッシュタグ 3〜5個。# を付ける"},
                },
                "required": ["start_id", "end_id", "title", "summary", "reason", "description", "hashtags"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
あなたは、長尺動画からショート動画（YouTubeショート・TikTok・Instagramリール）を切り出す編集者です。
渡された文字起こしを読み、ショート動画として単独で成立する部分を選んでください。

選ぶ基準:
- 元の動画を見ていない人が、その部分だけを見て話を理解できること。直前の話を知らないと意味が通じない始まり方（「それで」「さっきの」など）は避ける。
- 最初の一言で続きを見たくなること。問いかけ、意外な事実、強い主張、具体的な悩みなどから始まる部分が向いている。前置きや挨拶から始めない。
- 話が一区切りついたところ（結論、オチ、まとめの一言）で終わること。文の途中で切らない。
- 候補どうしは時間が重ならないこと。

返す順番は、おすすめ度が高い順にしてください。
基準を満たす部分が足りなければ、無理に数を合わせず、少なく返してかまいません。

文字起こしは自動で作られたもので、聞き間違いや誤字を含みます。
タイトル・要約・説明文では、文脈から明らかな誤りは正しい言葉に直してください。
タイトル・要約・理由・説明文・ハッシュタグはすべて日本語で書いてください。
"""


# ------------------------------------------------------------
# 準備の確認
# ------------------------------------------------------------
def ensure_module(name: str, required: bool = True) -> bool:
    """追加の部品（anthropic / cv2）が入った専用の Python で動かし直す"""
    try:
        __import__(name)
        return True
    except ImportError:
        pass
    for venv in VENVS:
        venv_python = venv / "bin" / "python"
        if venv_python.exists() and Path(sys.prefix).resolve() != venv.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    if required:
        raise AutocutError(
            "切り抜き機能の準備がまだです。ターミナルで次を実行してください"
            "（setup.sh はドラッグ＆ドロップできます）:\n    bash " + str(HERE / "setup.sh"))
    return False


def load_api_key():
    """api_key.txt の中身を返す。無ければ None"""
    key_file = HERE / "api_key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip() or None
    return None


def has_api_credentials() -> bool:
    return bool(load_api_key() or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# ------------------------------------------------------------
# 文字起こし（whisper.cpp）
# ------------------------------------------------------------
def whisper_model_path(name: str) -> Path:
    if name == "auto":
        # Apple Silicon なら高精度モデル、Intel Mac なら軽いモデル
        name = "large-v3-turbo" if platform.machine() == "arm64" else "small"
    filename = "ggml-{}.bin".format(name)
    path = APP_DIR / "models" / filename
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # 前の版のツールのフォルダにダウンロード済みなら、それを移して使う（1.6GBを再ダウンロードしない）
    old = [HERE / "models" / filename] + sorted(HERE.parent.glob("kairikomiya*/models/" + filename))
    for o in old:
        if o.exists():
            shutil.move(str(o), str(path))
            print("▶ 前にダウンロードした文字起こし用のモデルを使います")
            return path
    print("▶ 文字起こし用のモデル（{}）をダウンロードします。最初の1回だけです".format(name))
    part = path.with_name(path.name + ".part")
    r = subprocess.run(["curl", "-L", "--fail", "-#", "-o", str(part), MODEL_URL.format(name)])
    if r.returncode != 0:
        if part.exists():
            part.unlink()
        raise AutocutError("モデルのダウンロードに失敗しました。ネット接続と、config.toml の [transcribe] model の名前を確認してください。")
    part.replace(path)
    return path


def transcribe(src: Path, meta: dict, cfg: dict) -> list:
    """[{id, start, end, text}, ...] を返す（時間は秒）"""
    exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if not exe:
        raise AutocutError("文字起こしツール（whisper-cpp）が見つかりません。次を実行してください:\n    bash " + str(HERE / "setup.sh"))
    model = whisper_model_path(str(cfg["model"]))
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        r = autocut.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                         "-map", "0:{}".format(meta["audio_index"]),
                         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        if r.returncode != 0:
            raise AutocutError("音声の取り出しに失敗しました:\n" + r.stderr[-1500:])
        print("▶ 文字起こしをしています（動画の長さの 1/10〜同じくらいの時間がかかります）")
        prefix = Path(tmp) / "out"
        with tempfile.TemporaryFile(mode="w+", errors="replace") as errlog:
            # 聞き取れた文章がそのまま画面に流れます
            r = subprocess.run([exe, "-m", str(model), "-l", str(cfg["language"]),
                                "-f", str(wav), "-oj", "-of", str(prefix)], stderr=errlog)
            if r.returncode != 0:
                errlog.seek(0)
                raise AutocutError("文字起こしに失敗しました:\n" + errlog.read()[-1500:])
        data = json.loads((Path(tmp) / "out.json").read_text(encoding="utf-8", errors="replace"))
    segments = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        off = item["offsets"]
        segments.append({"id": len(segments) + 1, "start": off["from"] / 1000.0,
                         "end": off["to"] / 1000.0, "text": text})
    if not segments:
        raise AutocutError("言葉が聞き取れませんでした。喋りのない動画（ダンスなど）は、あとの段階で対応します。")
    return segments


# 音楽や無音のところで、Whisper が実際には言っていない言葉を出してしまうことがある
HALLUCINATIONS = ("大きな音楽", "おやすみなさい", "ご視聴ありがとうございました", "音楽", "拍手", "笑")


def warn_if_little_speech(segments: list, duration: float) -> None:
    """喋りがほとんど聞き取れなかったときに知らせる"""
    real = [s for s in segments if s["text"].strip(" 。、.!?！？()（）[]「」") not in HALLUCINATIONS]
    chars = sum(len(s["text"]) for s in real)
    per_minute = chars / max(duration / 60, 0.5)
    if per_minute < 60:
        print("  ⚠ この動画からは、喋っている言葉がほとんど聞き取れませんでした（1分あたり約{:.0f}文字）。".format(per_minute))
        print("    音楽やダンスが中心の動画は、あとの段階（見せ場探し）で対応します。今は喋りの動画で試してください。")
    if duration < 60:
        print("  ⚠ 動画が短いため（{:.0f}秒）、切り抜き候補はほとんど出ません。5分以上の動画がおすすめです。".format(duration))


def fmt_clock(t: float) -> str:
    """12:34.5 または 1:02:34.5 の形にする"""
    t = round(t, 1)
    h, rest = divmod(t, 3600)
    m, s = divmod(rest, 60)
    if h:
        return "{}:{:02d}:{:04.1f}".format(int(h), int(m), s)
    return "{:02d}:{:04.1f}".format(int(m), s)


# ------------------------------------------------------------
# AI（Claude）に候補を選んでもらう
# ------------------------------------------------------------
def build_user_prompt(segments: list, duration: float, cfg: dict) -> str:
    lines = ["[{}] {}〜{} {}".format(s["id"], fmt_clock(s["start"]), fmt_clock(s["end"]), s["text"])
             for s in segments]
    parts = []
    if str(cfg.get("channel_info", "")).strip():
        parts.append("<チャンネルについて>\n{}\n</チャンネルについて>".format(str(cfg["channel_info"]).strip()))
    parts.append(
        "動画の長さは {} です。下の文字起こしから、切り抜き候補を {}〜{} 本選んでください。\n"
        "1本の長さは {}〜{} 秒にしてください。短すぎても長すぎても投稿できないので、この範囲は守ってください。\n"
        "各行は「[行の番号] 開始〜終了 話した内容」です。切り抜く範囲は、最初の行と最後の行の番号で答えてください。".format(
            fmt_clock(duration), cfg["count_min"], cfg["count_max"], cfg["length_min"], cfg["length_max"]))
    parts.append("<文字起こし>\n{}\n</文字起こし>".format("\n".join(lines)))
    return "\n\n".join(parts)


def find_clips(segments: list, duration: float, cfg: dict) -> list:
    import anthropic

    model = str(cfg["model"])
    key = load_api_key()
    params = dict(
        model=model,
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(segments, duration, cfg)}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
    )
    try:
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        print("▶ AI（{}）が切り抜き候補を選んでいます（1〜3分かかります）".format(model))
        if model in FALLBACK_MODELS:
            stream_cm = client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        else:
            stream_cm = client.messages.stream(**params)
        received = 0
        with stream_cm as stream:
            for text in stream.text_stream:
                received += len(text)
                print("\r  受け取り中… {}文字".format(received), end="", flush=True)
            message = stream.get_final_message()
        print()
    except anthropic.AuthenticationError:
        raise AutocutError("APIキーが正しくないようです。api_key.txt の中身を確認してください（README の「APIキーを用意する」）。")
    except anthropic.PermissionDeniedError as e:
        raise AutocutError("このAPIキーでは実行できませんでした: " + str(e.message))
    except anthropic.NotFoundError:
        raise AutocutError("AIのモデル名「{}」が見つかりません。config.toml の [clips] model を確認してください。".format(model))
    except anthropic.RateLimitError:
        raise AutocutError("AIの利用が混み合っています。数分待ってからもう一度実行してください。")
    except anthropic.APIStatusError as e:
        if "credit" in str(e.message).lower():
            raise AutocutError("APIの残高（クレジット）が足りません。Claude Console の Billing で追加してください。")
        raise AutocutError("AIの呼び出しでエラーになりました（{}）: {}".format(e.status_code, e.message))
    except anthropic.APIConnectionError:
        raise AutocutError("AIに接続できませんでした。ネット接続を確認してください。")

    if message.stop_reason == "refusal":
        raise AutocutError("AIがこの文字起こしの処理を断りました。別の動画で試すか、内容を教えてください。")
    if message.stop_reason == "max_tokens":
        raise AutocutError("AIの返事が長すぎて途中で切れました。config.toml の count_max を減らしてもう一度試してください。")
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        raw = json.loads(text)["clips"]
    except (ValueError, KeyError, TypeError):
        raise AutocutError("AIの返事を読み取れませんでした。もう一度実行してください。")

    usage = message.usage
    line = "  使った量: 入力 {:,} / 出力 {:,} トークン".format(usage.input_tokens, usage.output_tokens)
    if model in PRICES and message.model == model:
        dollars = (usage.input_tokens * PRICES[model][0] + usage.output_tokens * PRICES[model][1]) / 1e6
        line += "（料金の目安: 約{:.0f}円 = ${:.2f}）".format(dollars * YEN_PER_DOLLAR, dollars)
    print(line)
    return clips_from_answer(raw, segments, duration)


def clips_from_answer(raw, segments: list, duration: float) -> list:
    """AI の返事（行の番号で書かれた範囲）を、時間つきの候補に直す"""
    if not isinstance(raw, list):
        raise AutocutError("AIの返事の形が正しくありません。もう一度試してください。")
    by_id = {s["id"]: s for s in segments}
    clips = []
    for c in raw:
        try:
            first, last = by_id.get(int(c["start_id"])), by_id.get(int(c["end_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        if first is None or last is None or first["id"] > last["id"]:
            continue  # 存在しない行番号は使えないので捨てる
        tags = c.get("hashtags") or []
        if isinstance(tags, str):
            tags = tags.split()
        clips.append({
            # 話し始めと話し終わりが切れないよう、少しだけ余白を足す
            "start": max(0.0, first["start"] - 0.15),
            "end": min(duration, last["end"] + 0.3),
            "title": str(c.get("title", "")).strip(),
            "hook": first["text"],
            "summary": str(c.get("summary", "")).strip(),
            "reason": str(c.get("reason", "")).strip(),
            "description": str(c.get("description", "")).strip(),
            "hashtags": " ".join(t if t.startswith("#") else "#" + t for t in tags),
        })
    if len(clips) < len(raw):
        print("  （AIの返事のうち {} 本は範囲が正しくなかったので除きました）".format(len(raw) - len(clips)))
    return clips


# ------------------------------------------------------------
# 無料の方法: Claude アプリに自分で貼り付ける
# ------------------------------------------------------------
ANSWER_FORMAT = """\
答えは、下の形のJSONだけを、コードブロックに入れて返してください。前置きや説明の文章は不要です。

```json
{"clips": [
  {"start_id": 最初の行の番号, "end_id": 最後の行の番号（この行まで含む）,
   "title": "動画の冒頭に大きく出すタイトル。20文字以内",
   "summary": "内容の要約。1〜2文",
   "reason": "なぜここを選んだか。1〜2文",
   "description": "投稿に添える説明文。2〜3文",
   "hashtags": ["#ハッシュタグ", "3〜5個"]}
]}
```
"""


def clipboard(text=None):
    """text を渡すとコピー、渡さないと貼り付け（Mac の pbcopy / pbpaste）。使えなければ None"""
    exe = shutil.which("pbcopy" if text is not None else "pbpaste")
    if not exe:
        return None
    r = subprocess.run([exe], input=text.encode("utf-8") if text is not None else None,
                       stdout=subprocess.PIPE)
    return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None


def write_prompt(path: Path, segments: list, duration: float, cfg: dict) -> bool:
    """Claude アプリに貼る依頼文を作る。クリップボードにコピーできたら True"""
    text = "\n\n".join([SYSTEM_PROMPT.strip(), ANSWER_FORMAT.strip(),
                         build_user_prompt(segments, duration, cfg)]) + "\n"
    path.write_text(text, encoding="utf-8")
    return clipboard(text) is not None


def read_pasted_answer() -> list:
    text = clipboard()
    if text is None:
        raise AutocutError("クリップボードを読めませんでした。")
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])["clips"]
    except (ValueError, KeyError, TypeError):
        raise AutocutError(
            "コピーされている内容が、Claude の返事（JSON）ではないようです。\n"
            "Claude の返事のコードブロック右上にある「コピー」を押してから、もう一度実行してください。")


# ------------------------------------------------------------
# 一覧の書き出し
# ------------------------------------------------------------
def write_candidates_csv(path: Path, clips: list) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, c in enumerate(clips, 1):
            w.writerow([i, autocut.YES, fmt_clock(c["start"]), fmt_clock(c["end"]),
                        "{:.1f}".format(c["end"] - c["start"]), "", "", c["title"], c["hook"],
                        c["summary"], c["reason"], c["description"], c["hashtags"]])


# ------------------------------------------------------------
# 書き出し（一覧で「はい」の候補を、選んだ形で書き出す）
# ------------------------------------------------------------
def parse_shapes(text: str) -> list:
    shapes = []
    for word in re.split(r"[,、・/\s　]+", str(text).strip()):
        if not word:
            continue
        name = SHAPE_ALIASES.get(word, word.lower())
        if name not in SHAPES:
            raise AutocutError("形の名前「{}」が分かりません。使えるのは tate（縦）/ yoko（横）/ square（正方形）/ feed（4:5）です。".format(word))
        if name not in shapes:
            shapes.append(name)
    return shapes


def parse_clock(text: str) -> float:
    """「12:34.5」「1:02:34.5」「83.4」「1分23秒」「1m 23.4s」などを秒に直す"""
    t = str(text).strip().replace("：", ":")
    units = re.findall(r"([\d.]+)\s*(時間|h|分|m|秒|s)", t)
    if units:
        scale = {"時間": 3600, "h": 3600, "分": 60, "m": 60, "秒": 1, "s": 1}
        return sum(float(v) * scale[u] for v, u in units)
    total = 0.0
    for part in t.split(":"):
        total = total * 60 + float(part)
    return total


def read_candidates_csv(path: Path, duration: float, default_shapes: list) -> list:
    if not path.exists():
        raise AutocutError("候補の一覧が見つかりません: {}\n先に候補を作ってください（README の「切り抜き」）。".format(path))
    clips = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), 1):
            flag = (row.get("使う") or "").strip().lower()
            if flag in (autocut.NO, "no", "n", "0", "false", "×", "x", ""):
                continue
            where = "候補の一覧の {} 行目（番号 {}）".format(n + 1, row.get("番号", "?"))
            try:
                start, end = parse_clock(row["開始"]), parse_clock(row["終了"])
            except (KeyError, TypeError, ValueError):
                raise AutocutError(where + "の「開始」「終了」が読めません。12:34.5 のように書いてください。")
            end = min(end, duration)
            if not 0 <= start < end:
                raise AutocutError(where + "の時間が正しくありません（開始 {} / 終了 {} / 動画の長さ {}）。".format(
                    row["開始"], row["終了"], fmt_clock(duration)))
            pos = (row.get("位置") or "").strip().rstrip("%％")
            try:
                pos = min(100.0, max(0.0, float(pos))) if pos else None
            except ValueError:
                raise AutocutError(where + "の「位置」は 0〜100 の数字で書いてください（空なら自動）。")
            try:
                number = int(float(row.get("番号") or n))
            except ValueError:
                number = n
            clips.append({"no": number, "start": start, "end": end,
                          "title": (row.get("タイトル案") or "").strip(),
                          "description": (row.get("説明文") or "").strip(),
                          "hashtags": (row.get("ハッシュタグ") or "").strip(),
                          "shapes": parse_shapes(row.get("形") or "") or default_shapes, "pos": pos})
    return clips


def find_subject(src: Path, meta: dict, start: float, length: float):
    """クリップの中の人物の位置を探す。(横位置, 縦位置, 説明) を 0〜1 で返す。見つからなければ None"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    scale = 640.0 / max(meta["width"], meta["height"])
    w, h = int(meta["width"] * scale) // 2 * 2, int(meta["height"] * scale) // 2 * 2
    # 1秒に1コマだけ、小さい白黒画像で取り出して調べる
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "{:.3f}".format(start), "-t", "{:.3f}".format(length),
         "-i", str(src), "-map", "0:{}".format(meta["video_index"]),
         "-vf", "fps=1,scale={}:{}".format(w, h), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = len(r.stdout) // (w * h)
    if r.returncode != 0 or count == 0:
        return None
    frames = np.frombuffer(r.stdout[:count * w * h], dtype=np.uint8).reshape(count, h, w)
    enough = max(2, count // 5)  # 全体の2割以上のコマで見つかったら採用

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    small = max(24, min(w, h) // 12)
    found = []
    for frame in frames:
        boxes = face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6, minSize=(small, small))
        if len(boxes):
            x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])  # いちばん大きい顔
            found.append(((x + bw / 2) / w, (y + bh / 2) / h))
    kind = "顔"
    if len(found) < enough:  # 顔が小さい・写っていないときは、全身を探す
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found = []
        for frame in frames:
            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05)
            if len(boxes) and float(max(weights)) > 0.5:
                x, y, bw, bh = boxes[int(np.argmax(weights))]
                found.append(((x + bw / 2) / w, (y + bh * 0.3) / h))  # 上半身あたりを中心に
        kind = "全身"
    if len(found) < enough:
        return None
    return float(np.median([p[0] for p in found])), float(np.median([p[1] for p in found])), kind


def crop_box(meta: dict, tw: int, th: int, subject, pos):
    """元の映像から切り出す範囲 (幅, 高さ, 左, 上) を決める。画面いっぱいに映るように切る"""
    W, H = meta["width"], meta["height"]
    if W * th > H * tw:   # 元のほうが横長 → 左右を切る
        cw, ch = min(W, int(H * tw / th) // 2 * 2), H
    else:                 # 元のほうが縦長 → 上下を切る
        cw, ch = W, min(H, int(W * th / tw) // 2 * 2)
    if pos is not None:   # 一覧の「位置」で手動指定（0=左/上, 50=中央, 100=右/下）
        x, y = (W - cw) * pos / 100, (H - ch) * pos / 100
    elif subject:
        x = subject[0] * W - cw / 2
        y = subject[1] * H - ch * 0.4   # 上下を切るときは、人物を少し上寄りに置く
    else:
        x, y = (W - cw) / 2, (H - ch) / 2
    return cw, ch, int(min(max(x, 0), W - cw)), int(min(max(y, 0), H - ch))


def safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "", title).strip(" .")
    return name[:30] or "clip"


def clip_subtitles(clip: dict, segments, sub_dir: Path, src: Path, cfg: dict):
    """クリップの字幕を用意する。手で直した .srt があればそれを使う"""
    import telop
    if not segments or not cfg["subtitle"]["enabled"]:
        return []
    sub_dir.mkdir(exist_ok=True)
    srt = sub_dir / "{}_{:02d}.srt".format(src.stem, clip["no"])
    info = srt.with_suffix(".json")
    span = {"start": round(clip["start"], 2), "end": round(clip["end"], 2)}
    if srt.exists() and info.exists():
        try:
            same = json.loads(info.read_text(encoding="utf-8")) == span
        except ValueError:
            same = False
        if same:
            subs = telop.read_srt(srt)
            print("  字幕: {} を使います（{}個）".format(srt.name, len(subs)))
            return subs
        old = backup(srt)
        print("  字幕: 開始・終了が変わったので作り直します（前の字幕は {} に残しました）".format(old.name))
    subs = telop.subtitles_for_clip(segments, clip["start"], clip["end"], int(cfg["subtitle"]["max_chars"]))
    telop.write_srt(srt, subs)
    info.write_text(json.dumps(span), encoding="utf-8")
    print("  字幕: {}個 作りました（{}）".format(len(subs), srt.name))
    return subs


def write_post_text(path: Path, src: Path, clips: list, made: dict) -> None:
    """投稿するときにコピーして使う、タイトル・説明文・ハッシュタグのまとめ"""
    lines = ["{} の切り抜き 投稿用テキスト".format(src.name), ""]
    for clip in clips:
        lines.append("=" * 40)
        lines.append("{:02d}  {}".format(clip["no"], clip["title"]))
        for f in made.get(clip["no"], []):
            lines.append("    動画: " + f.name)
        lines += ["", "【タイトル】", clip["title"], "", "【説明文】", clip["description"] or "（なし）", "",
                  "【ハッシュタグ】", clip["hashtags"] or "（なし）", "",
                  "【説明文＋ハッシュタグ（そのまま貼る用）】",
                  "\n\n".join(x for x in (clip["description"], clip["hashtags"]) if x), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_clips(src: Path, meta: dict, clips: list, out_dir: Path, cfg: dict, segments=None) -> list:
    import tempfile
    import telop
    out_dir.mkdir(exist_ok=True)
    fps = meta["fps"]
    hdr, color_args = autocut.hdr_filters(meta)
    made = {}
    for clip in clips:
        length = clip["end"] - clip["start"]
        print("▶ {:02d} {}（{}〜{}、{:.0f}秒）".format(
            clip["no"], clip["title"], fmt_clock(clip["start"]), fmt_clock(clip["end"]), length))
        subs = clip_subtitles(clip, segments, out_dir.parent / "字幕", src, cfg)
        targets = [(name,) + SHAPES[name] for name in clip["shapes"]]
        # 元の映像と形が違うときだけ、どこを切り出すか決める必要がある
        needs_crop = any(tw * meta["height"] != th * meta["width"] for _, _, tw, th in targets)
        subject = None
        if needs_crop and clip["pos"] is None:
            subject = find_subject(src, meta, clip["start"], length)
            if subject:
                print("  人物の位置: {}を見つけました（左から {:.0f}%）".format(subject[2], subject[0] * 100))
            else:
                print("  人物の位置: 見つからなかったので、中央で切り出します")
        for shape, label, tw, th in targets:
            cw, ch, x, y = crop_box(meta, tw, th, subject, clip["pos"])
            vf = (["fps={}/{}".format(fps.numerator, fps.denominator)] + hdr
                  + ["crop={}:{}:{}:{}".format(cw, ch, x, y), "scale={}:{}".format(tw, th), "setsar=1"])
            dst = out_dir / "{}_{:02d}_{}_{}.mp4".format(src.stem, clip["no"], safe_name(clip["title"]), label)
            print("  {}（{}x{}）".format(label, tw, th))
            inputs = ["-ss", "{:.3f}".format(clip["start"]), "-t", "{:.3f}".format(length), "-i", str(src)]
            title_secs = float(cfg["title"]["seconds"]) if cfg["title"]["enabled"] else 0
            with tempfile.TemporaryDirectory() as work:
                if (clip["title"] and title_secs > 0) or subs:
                    t_style, s_style = telop.styles(cfg, tw, th, shape)
                    lst = telop.build_overlay(Path(work), tw, th, length, clip["title"], title_secs,
                                              subs, t_style, s_style)
                    inputs += ["-f", "concat", "-safe", "0", "-i", str(lst)]
                    graph = ("[0:{}]{}[base];[1:v]format=rgba[txt];"
                             "[base][txt]overlay=0:0:eof_action=repeat,format=yuv420p[v];"
                             "[0:{}]aresample=48000[a]").format(meta["video_index"], ",".join(vf), meta["audio_index"])
                else:
                    graph = "[0:{}]{},format=yuv420p[v];[0:{}]aresample=48000[a]".format(
                        meta["video_index"], ",".join(vf), meta["audio_index"])
                autocut.run_encode(inputs, graph, dst, length, cfg["encode"], color_args)
            made.setdefault(clip["no"], []).append(dst)
    write_post_text(out_dir / "{}_投稿用テキスト.txt".format(src.stem), src, clips, made)
    return [f for files in made.values() for f in files]


def backup(path: Path) -> Path:
    n = 1
    while True:
        old = path.with_name("{}_old{}{}".format(path.stem, n, path.suffix))
        if not old.exists():
            path.rename(old)
            return old
        n += 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="長い動画から、ショート動画の切り抜き候補を探して一覧にします。")
    p.add_argument("video", type=Path, help="元の動画ファイル（mp4 / mov）")
    p.add_argument("--config", type=Path, help="設定ファイル（省略すると clip.py と同じ場所の config.toml）")
    p.add_argument("--paste", action="store_true", help="Claude アプリからコピーした返事を読み込んで、一覧を作る")
    p.add_argument("--export", nargs="*", metavar="形",
                   help="一覧で「はい」の候補を書き出す。形は tate（縦 9:16）/ yoko（横 16:9）/ square（1:1）/ feed（4:5）。"
                        "複数並べると全部作る。省略すると config.toml の shape")
    p.add_argument("--no-subs", action="store_true", help="（--export のとき）字幕を入れない")
    p.add_argument("--list", type=Path, help="（--export のとき）別の場所に保存した候補の一覧（CSV）を使う")
    p.add_argument("--redo", action="store_true", help="（API を使う場合）候補の一覧がすでにあっても、AIに選び直してもらう")
    args = p.parse_args(argv)

    src = args.video.expanduser()
    if not src.is_file():
        raise AutocutError("動画ファイルが見つかりません: {}".format(src))
    cfg = autocut.load_config(args.config)
    if args.export is not None:
        ensure_module("PIL")  # 文字を描く部品
        ensure_module("cv2", required=False)  # 人物の位置を探す部品。無ければ中央で切り出す
        shapes = parse_shapes(" ".join(args.export)) or parse_shapes(cfg["clips"]["shape"]) or ["tate"]
        out_dir = src.parent / (src.stem + "_clips")
        csv_path = args.list.expanduser() if args.list else out_dir / (src.stem + "_candidates.csv")
        print("▶ 動画を調べています: " + src.name)
        meta = autocut.probe(src)
        clips = read_candidates_csv(csv_path, meta["duration"], shapes)
        if not clips:
            raise AutocutError("一覧に「使う」が「はい」の候補がありません: {}".format(csv_path))
        print("  {} 本を書き出します（一覧: {}）".format(len(clips), csv_path.name))
        out_dir.mkdir(exist_ok=True)
        transcript_json = out_dir / (src.stem + "_transcript.json")
        segments = None
        if transcript_json.exists() and not args.no_subs:
            segments = json.loads(transcript_json.read_text(encoding="utf-8"))
        made = export_clips(src, meta, clips, out_dir / "書き出し", cfg, segments)
        print("✔ {} 本の動画ができました: {}".format(len(made), out_dir / "書き出し"))
        print("  投稿用テキスト: {}_投稿用テキスト.txt（同じフォルダ）".format(src.stem))
        if segments:
            print("  字幕ファイル  : {}".format(out_dir / "字幕"))
        return 0

    use_api = has_api_credentials() and not args.paste
    if use_api:
        ensure_module("anthropic")

    out_dir = src.parent / (src.stem + "_clips")
    out_dir.mkdir(exist_ok=True)
    transcript_json = out_dir / (src.stem + "_transcript.json")
    transcript_txt = out_dir / (src.stem + "_transcript.txt")
    csv_path = out_dir / (src.stem + "_candidates.csv")
    prompt_path = out_dir / (src.stem + "_prompt.txt")

    if use_api and csv_path.exists() and not args.redo:
        raise AutocutError(
            "候補の一覧がすでにあります: {}\n"
            "選び直したいときは、コマンドの最後に  --redo  を足してください"
            "（今の一覧は別の名前で残します。AIの料金はもう一度かかります）。".format(csv_path))

    print("▶ 動画を調べています: " + src.name)
    meta = autocut.probe(src)
    print("  長さ {}".format(fmt_clock(meta["duration"])))

    if transcript_json.exists():
        segments = json.loads(transcript_json.read_text(encoding="utf-8"))
        print("▶ 前回の文字起こしを使います: " + transcript_json.name)
    else:
        segments = transcribe(src, meta, cfg["transcribe"])
        transcript_json.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
        transcript_txt.write_text(
            "\n".join("[{}] {}".format(fmt_clock(s["start"]), s["text"]) for s in segments) + "\n",
            encoding="utf-8")
    warn_if_little_speech(segments, meta["duration"])

    if args.paste:
        clips = clips_from_answer(read_pasted_answer(), segments, meta["duration"])
    elif use_api:
        clips = find_clips(segments, meta["duration"], cfg["clips"])
    else:
        copied = write_prompt(prompt_path, segments, meta["duration"], cfg["clips"])
        print("✔ Claude への依頼文を作りました: " + str(prompt_path))
        print("  次の手順で、候補を選んでもらってください。")
        if copied:
            print("  1. Claude アプリで新しいチャットを開き、貼り付けて（⌘V）送信する（依頼文はコピー済みです）")
        else:
            print("  1. Claude アプリで新しいチャットを開き、上のファイルをドラッグ＆ドロップして送信する")
        print("  2. 返事のコードブロック右上の「コピー」を押す")
        print("  3. さっきのコマンドの最後に  --paste  を足して、もう一度実行する")
        return 0
    if not clips:
        raise AutocutError("切り抜きに向いた部分が見つかりませんでした。")
    if csv_path.exists():
        print("  前の一覧は {} として残しました".format(backup(csv_path).name))
    write_candidates_csv(csv_path, clips)

    lo, hi = cfg["clips"]["length_min"], cfg["clips"]["length_max"]
    print("✔ 切り抜き候補を {} 本見つけました".format(len(clips)))
    for i, c in enumerate(clips, 1):
        length = c["end"] - c["start"]
        note = "" if lo - 3 <= length <= hi + 3 else "  ※長さが範囲外"
        print("  {:2d}. {}〜{}（{:.0f}秒） {}{}".format(
            i, fmt_clock(c["start"]), fmt_clock(c["end"]), length, c["title"], note))
    print("  一覧      : " + str(csv_path))
    print("  文字起こし: " + str(transcript_txt))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutocutError as e:
        print("\n✖ " + str(e), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        sys.exit(130)
