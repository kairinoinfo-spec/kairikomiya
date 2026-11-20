# -*- coding: utf-8 -*-
"""speech — 文字起こし（whisper.cpp）と、言い間違い・「えー」「あの」を見つける部品

autocut.py と clip.py の両方から使います。Python の追加部品はいりません。
"""
from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

from autocut import AutocutError, run

HERE = Path(__file__).resolve().parent
# ツールを新しくダウンロードし直しても消えないように、大きなものはここに置く
if platform.system() == "Darwin":
    APP_DIR = Path.home() / "Library" / "Application Support" / "kairikomiya"
else:
    APP_DIR = Path.home() / ".local" / "share" / "kairikomiya"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"
# 単語ごとの時間を正確にするための設定名（whisper.cpp の DTW）
DTW_PRESETS = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium",
               "large-v3": "large.v3", "large-v3-turbo": "large.v3.turbo"}

# 音楽や無音のところで、Whisper が実際には言っていない言葉を出してしまうことがある
HALLUCINATIONS = ("大きな音楽", "おやすみなさい", "ご視聴ありがとうございました", "音楽", "拍手", "笑")
PUNCT = "、。，．,.！？!?…「」『』（）() 　"


# ------------------------------------------------------------
# モデル
# ------------------------------------------------------------
def model_name(name: str) -> str:
    if name == "auto":
        # Apple Silicon なら高精度モデル、Intel Mac なら軽いモデル
        return "large-v3-turbo" if platform.machine() == "arm64" else "small"
    return name


def whisper_model_path(name: str) -> Path:
    name = model_name(name)
    filename = "ggml-{}.bin".format(name)
    path = APP_DIR / "models" / filename
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # 前の版のツールのフォルダにダウンロード済みなら、それを移して使う（1.6GBを再ダウンロードしない）
    old = [HERE / "models" / filename] + sorted(HERE.parent.glob("kairikomiya*/models/" + filename))
    for place in (Path.home() / "Downloads", Path.home() / "Desktop"):
        old += sorted(place.glob("kairikomiya*/models/" + filename))
        old += sorted(place.glob("*/kairikomiya*/models/" + filename))
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


def whisper_exe():
    return shutil.which("whisper-cli") or shutil.which("whisper-cpp")


# ------------------------------------------------------------
# 文字起こし
# ------------------------------------------------------------
def transcribe(src: Path, meta: dict, cfg: dict, prompt: str = "", chars: bool = False) -> list:
    """[{id, start, end, text, (chars: [[文字, 開始, 終了], ...])}, ...] を返す（時間は秒）
    chars=True のときは、1文字ごとの時間も返す（言い間違いを見つけるのに使う）"""
    exe = whisper_exe()
    if not exe:
        raise AutocutError("文字起こしツール（whisper-cpp）が見つかりません。次を実行してください:\n    bash "
                           + shlex.quote(str(HERE / "setup.sh")))
    model = whisper_model_path(str(cfg["model"]))
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                 "-map", "0:{}".format(meta["audio_index"]),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        if r.returncode != 0:
            raise AutocutError("音声の取り出しに失敗しました:\n" + r.stderr[-1500:])
        print("▶ 文字起こしをしています（動画の長さの 1/10〜同じくらいの時間がかかります）")
        prefix = Path(tmp) / "out"
        cmd = [exe, "-m", str(model), "-l", str(cfg["language"]), "-f", str(wav),
               "-ojf" if chars else "-oj", "-of", str(prefix)]
        if prompt:
            cmd += ["--prompt", prompt]
        dtw = DTW_PRESETS.get(model_name(str(cfg["model"])))
        attempts = [cmd + ["-dtw", dtw], cmd] if (chars and dtw) else [cmd]
        error = ""
        for attempt in attempts:
            with tempfile.TemporaryFile(mode="w+", errors="replace") as errlog:
                # 聞き取れた文章がそのまま画面に流れます
                r = subprocess.run(attempt, stderr=errlog)
                if r.returncode == 0:
                    break
                errlog.seek(0)
                error = errlog.read()[-1500:]
        else:
            raise AutocutError("文字起こしに失敗しました:\n" + error)
        # 日本語の1文字が2つの部品に分かれて、途中で切れた文字が入ることがあるので、
        # 壊れた部分もそのまま読み込んでおいて、あとでつなぎ直す
        raw = (Path(tmp) / "out.json").read_bytes().decode("utf-8", "surrogateescape")
        data = json.loads(raw)
    segments = []
    for item in data.get("transcription", []):
        text = fix_text(item.get("text", "")).strip()
        if not text:
            continue
        off = item["offsets"]
        seg = {"id": len(segments) + 1, "start": off["from"] / 1000.0, "end": off["to"] / 1000.0, "text": text}
        if chars:
            seg["chars"] = token_chars(item.get("tokens", []), seg["start"], seg["end"])
        segments.append(seg)
    if not segments:
        raise AutocutError("言葉が聞き取れませんでした。")
    return segments


def fix_text(text: str) -> str:
    return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace").replace("�", "")


def token_chars(tokens: list, seg_start: float, seg_end: float) -> list:
    """部品（トークン）ごとの時間から、1文字ごとの時間 [[文字, 開始, 終了], ...] を作る"""
    out, pending, p_start = [], b"", None
    for tok in tokens:
        text = tok.get("text", "")
        if re.fullmatch(r"\[_[^\]]*\]|<\|[^|]*\|>", text):
            continue  # 目印用の特別な部品
        off = tok.get("offsets", {})
        start = off.get("from", seg_start * 1000) / 1000.0
        end = off.get("to", seg_end * 1000) / 1000.0
        dtw = tok.get("t_dtw", -1)
        if isinstance(dtw, (int, float)) and dtw >= 0:
            start = dtw / 100.0  # DTW の時間（1/100秒単位）のほうが正確
        if p_start is None:
            p_start = start
        pending += text.encode("utf-8", "surrogateescape")
        try:
            decoded = pending.decode("utf-8")
        except UnicodeDecodeError:
            continue  # 文字の途中で切れているので、次の部品とつなぐ
        pending = b""
        decoded = decoded.strip()
        if decoded:
            end = max(end, p_start + 0.02)
            d = (end - p_start) / len(decoded)
            for i, ch in enumerate(decoded):
                out.append([ch, round(p_start + d * i, 3), round(p_start + d * (i + 1), 3)])
        p_start = None
    # 時間が前後しないように整える
    last = seg_start
    for c in out:
        c[1] = max(c[1], last)
        c[2] = max(c[2], c[1] + 0.01)
        last = c[1]
    return out


def is_hallucination(text: str) -> bool:
    return text.strip(" 。、.!?！？()（）[]「」") in HALLUCINATIONS


# ------------------------------------------------------------
# 「えー」「あの」・言い直し・繰り返しを見つける
# ------------------------------------------------------------
def filler_pattern(words: list):
    alts = []
    for w in sorted({str(x).strip() for x in words if str(x).strip()}, key=len, reverse=True):
        # 語尾が伸びた形（えーー、あのー、うーーん）もまとめて見つける
        alts.append(re.escape(w).replace("ー", "ー+") + "[ーぁぃぅぇぉっ〜~]*")
    return re.compile("|".join(alts)) if alts else None


def boundary(chars: list, i: int, side: str, gap: float) -> bool:
    """i 文字目の前（side="before"）または後ろが、言葉の切れ目か（文の端・句読点・間）"""
    if side == "before":
        if i == 0:
            return True
        prev = chars[i - 1]
        return prev[0] in PUNCT or chars[i][1] - prev[2] >= gap
    if i >= len(chars) - 1:
        return True
    nxt = chars[i + 1]
    return nxt[0] in PUNCT or nxt[1] - chars[i][2] >= gap


def find_word_cuts(segments: list, cfg: dict) -> list:
    """[{start, end, reason, text}, ...] を返す"""
    cuts = []
    pat = filler_pattern(cfg["fillers"]) if cfg["fillers_enabled"] else None
    gap = 0.15
    for seg_i, seg in enumerate(segments):
        chars = seg.get("chars") or []
        if not chars or is_hallucination(seg["text"]):
            continue
        s = "".join(c[0] for c in chars)
        # 1) 「えー」「あの」など
        if pat is not None:
            for m in pat.finditer(s):
                a, b = m.start(), m.end() - 1
                if boundary(chars, a, "before", gap) and boundary(chars, b, "after", gap):
                    end_i = b + 1 if b + 1 < len(chars) and chars[b + 1][0] in "、，," else b
                    cuts.append({"start": chars[a][1], "end": chars[end_i][2], "reason": "えー・あの",
                                 "text": s[a:end_i + 1]})
        # 2) すぐ同じ言葉を繰り返した（「これは、これは」「その、その」）→ 1回目を消す
        if cfg["repeats_enabled"]:
            i = 0
            while i < len(s):
                hit = False
                for n in range(min(10, (len(s) - i) // 2), 1, -1):
                    first = s[i:i + n]
                    if any(ch in PUNCT for ch in first):
                        continue
                    j = i + n
                    sep = 0
                    while j + sep < len(s) and s[j + sep] in "、，, 　":
                        sep += 1
                    second = s[j + sep:j + sep + n]
                    if first != second:
                        continue
                    paused = sep > 0 or chars[j + sep][1] - chars[j - 1][2] >= 0.2
                    if n >= 3 or paused:  # 2文字の繰り返しは「いろいろ」などの言葉と区別するため、間がある時だけ
                        cuts.append({"start": chars[i][1], "end": chars[j + sep - 1][2], "reason": "繰り返し",
                                     "text": s[i:j + sep]})
                        i = j + sep
                        hit = True
                        break
                if not hit:
                    i += 1
        # 3) 言い直し: 言いかけてやめて、次で言い直した（次の文が、この文と同じ言葉で始まる）
        if cfg["restarts_enabled"] and seg_i + 1 < len(segments):
            nxt = segments[seg_i + 1]
            a_text = s.strip(PUNCT)
            b_text = "".join(c[0] for c in (nxt.get("chars") or [])) or nxt["text"]
            if 2 <= len(a_text) <= 20 and nxt["start"] - seg["end"] < 3.0 and len(b_text) > len(a_text):
                head = b_text[:len(a_text) + 2]
                prefix = 0
                while prefix < min(len(a_text), len(b_text)) and a_text[prefix] == b_text[prefix]:
                    prefix += 1
                similar = SequenceMatcher(None, a_text, head).ratio()
                if prefix >= 2 and (prefix >= len(a_text) * 0.5 or similar >= 0.7) and not s.rstrip().endswith("。"):
                    cuts.append({"start": chars[0][1], "end": chars[-1][2], "reason": "言い直し",
                                 "text": s})
    return cuts


def find_speech_gaps(segments: list, duration: float, cfg: dict) -> list:
    """喋っていない時間（言葉と言葉の間）を探す。BGM が鳴っていても見つけられる。
    [(開始, 終了), ...] を返す（余白はあとで足す）"""
    spans = []
    for seg in segments:
        if is_hallucination(seg["text"]):
            continue
        chars = seg.get("chars") or []
        if chars:
            spans += [(c[1], c[2]) for c in chars if c[0] not in PUNCT]
        else:
            spans.append((seg["start"], seg["end"]))
    spans.sort()
    gaps, pos = [], 0.0
    for a, b in spans:
        if a - pos >= float(cfg["min_silence"]):
            gaps.append((pos, a))
        pos = max(pos, b)
    if duration - pos >= float(cfg["min_silence"]):
        gaps.append((pos, duration))
    return gaps
