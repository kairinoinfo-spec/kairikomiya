#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clip — 長い動画から、ショート動画の切り抜き候補を探すツール（第1段階: 喋りの動画）

使い方:
    python3 clip.py 動画ファイル.mp4

流れ:  文字起こし（Whisper）→ AI（Claude）が候補を選ぶ → 候補の一覧（CSV）を作る
準備:  bash setup.sh   と   APIキーの保存（README を見てください）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import autocut
from autocut import AutocutError

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
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

CSV_COLUMNS = ["番号", "使う", "開始", "終了", "長さ(秒)", "タイトル案", "冒頭のセリフ",
               "要約", "選んだ理由", "説明文", "ハッシュタグ"]

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
def ensure_sdk() -> None:
    """Claude を呼ぶための部品（anthropic）が入った Python で動かし直す"""
    try:
        import anthropic  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = VENV / "bin" / "python"
    if venv_python.exists() and Path(sys.prefix).resolve() != VENV.resolve():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    raise AutocutError(
        "切り抜き機能の準備がまだです。ターミナルで次を実行してください"
        "（setup.sh はドラッグ＆ドロップできます）:\n    bash " + str(HERE / "setup.sh"))


NO_KEY_MESSAGE = (
    "APIキーが見つかりません。README の「APIキーを用意する」の手順で、"
    "clip.py と同じフォルダに api_key.txt を作ってください。")


def load_api_key():
    """api_key.txt の中身を返す。無ければ None（そのときは環境変数などの設定が使われる）"""
    key_file = HERE / "api_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return None
    if (Path.home() / ".config" / "anthropic").exists():  # `ant auth login` 済み
        return None
    raise AutocutError(NO_KEY_MESSAGE)


# ------------------------------------------------------------
# 文字起こし（whisper.cpp）
# ------------------------------------------------------------
def whisper_model_path(name: str) -> Path:
    if name == "auto":
        # Apple Silicon なら高精度モデル、Intel Mac なら軽いモデル
        name = "large-v3-turbo" if platform.machine() == "arm64" else "small"
    path = HERE / "models" / "ggml-{}.bin".format(name)
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
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
    except TypeError as e:  # SDK が認証情報を見つけられなかったとき
        if "authentication" not in str(e):
            raise
        raise AutocutError(NO_KEY_MESSAGE)

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

    by_id = {s["id"]: s for s in segments}
    clips = []
    for c in raw:
        first, last = by_id.get(c["start_id"]), by_id.get(c["end_id"])
        if first is None or last is None or first["id"] > last["id"]:
            continue  # 存在しない行番号は使えないので捨てる
        clips.append({
            # 話し始めと話し終わりが切れないよう、少しだけ余白を足す
            "start": max(0.0, first["start"] - 0.15),
            "end": min(duration, last["end"] + 0.3),
            "title": c["title"].strip(),
            "hook": first["text"],
            "summary": c["summary"].strip(),
            "reason": c["reason"].strip(),
            "description": c["description"].strip(),
            "hashtags": " ".join(t if t.startswith("#") else "#" + t for t in c["hashtags"]),
        })
    if len(clips) < len(raw):
        print("  （AIの返事のうち {} 本は範囲が正しくなかったので除きました）".format(len(raw) - len(clips)))
    return clips


# ------------------------------------------------------------
# 一覧の書き出し
# ------------------------------------------------------------
def write_candidates_csv(path: Path, clips: list) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, c in enumerate(clips, 1):
            w.writerow([i, autocut.YES, fmt_clock(c["start"]), fmt_clock(c["end"]),
                        "{:.1f}".format(c["end"] - c["start"]), c["title"], c["hook"],
                        c["summary"], c["reason"], c["description"], c["hashtags"]])


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
    p.add_argument("--redo", action="store_true", help="候補の一覧がすでにあっても、AIに選び直してもらう")
    args = p.parse_args(argv)

    src = args.video.expanduser()
    if not src.is_file():
        raise AutocutError("動画ファイルが見つかりません: {}".format(src))
    cfg = autocut.load_config(args.config)
    ensure_sdk()
    load_api_key()  # 時間のかかる文字起こしの前に、キーがあるか確かめておく

    out_dir = src.parent / (src.stem + "_clips")
    out_dir.mkdir(exist_ok=True)
    transcript_json = out_dir / (src.stem + "_transcript.json")
    transcript_txt = out_dir / (src.stem + "_transcript.txt")
    csv_path = out_dir / (src.stem + "_candidates.csv")

    if csv_path.exists() and not args.redo:
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

    clips = find_clips(segments, meta["duration"], cfg["clips"])
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
