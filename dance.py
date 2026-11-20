# -*- coding: utf-8 -*-
"""dance — ダンス・ライブ映像から見せ場を探す部品（clip.py から使う）

音（音量・ビート）と映像の動きの大きさを0.1秒ごとに測り、
盛り上がっている部分を切り抜き候補にします。AI は使わないので無料です。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from autocut import AutocutError

STEP = 0.1          # 0.1秒ごとに測る
AUDIO_RATE = 11025  # 音を調べるときの細かさ（低めで十分）


def _np():
    import numpy as np
    return np


def run_pipe(cmd: list) -> bytes:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise AutocutError("動画の読み込みに失敗しました:\n" + r.stderr.decode("utf-8", "replace")[-1500:])
    return r.stdout


# ------------------------------------------------------------
# 測る
# ------------------------------------------------------------
def measure_audio(src: Path, meta: dict, n: int):
    """音量（dB）と、ビートの強さ（音の立ち上がり）を STEP ごとに返す"""
    np = _np()
    raw = run_pipe(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:{}".format(meta["audio_index"]),
                    "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"])
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(AUDIO_RATE * STEP)
    usable = len(x) // hop * hop
    frames = x[:usable].reshape(-1, hop)
    loud = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    # ビート: 短く区切って周波数ごとの強さを出し、前より強くなった分を足す（スペクトルフラックス）
    win = 1024
    sub = 512
    count = max(0, (len(x) - win) // sub)
    flux_fine = np.zeros(count, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)
    prev = None
    chunk = 4096  # 一度に計算する数（メモリを使いすぎないように）
    for c0 in range(0, count, chunk):
        idx = np.arange(c0, min(count, c0 + chunk))[:, None] * sub + np.arange(win)[None, :]
        spec = np.log1p(np.abs(np.fft.rfft(x[idx] * window, axis=1)))
        if prev is not None:
            spec_prev = np.vstack([prev[None, :], spec[:-1]])
        else:
            spec_prev = np.vstack([spec[:1], spec[:-1]])
        flux_fine[c0:c0 + len(spec)] = np.maximum(spec - spec_prev, 0).sum(axis=1)
        prev = spec[-1]
    # STEP ごとにまとめる
    per = hop / sub
    flux = np.zeros(len(loud), dtype=np.float32)
    for i in range(len(loud)):
        a, b = int(i * per), int((i + 1) * per)
        if a < len(flux_fine):
            flux[i] = flux_fine[a:max(b, a + 1)].mean()
    # fine: ビートの位置合わせ用に、細かい間隔のまま残す
    return fit(loud, n), fit(flux, n), (flux_fine, sub / AUDIO_RATE)


def fit(arr, n: int):
    np = _np()
    if len(arr) >= n:
        return arr[:n]
    pad = arr[-1] if len(arr) else 0
    return np.concatenate([arr, np.full(n - len(arr), pad, dtype=np.float32)])


def measure_motion(src: Path, meta: dict, n: int):
    """映像の動きの大きさと、動いている場所の横位置を STEP ごとに返す。カメラの切り替えは動きに数えない"""
    np = _np()
    w = 96
    h = max(2, int(round(meta["height"] * w / meta["width"] / 2)) * 2)
    raw = run_pipe(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:{}".format(meta["video_index"]),
                    "-vf", "fps={},scale={}:{}".format(int(round(1 / STEP)), w, h),
                    "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    count = len(raw) // (w * h)
    if count < 2:
        return fit(np.zeros(1, dtype=np.float32), n), fit(np.full(1, 0.5, dtype=np.float32), n), []
    frames = np.frombuffer(raw[:count * w * h], dtype=np.uint8).reshape(count, h, w).astype(np.int16)
    diff = np.abs(frames[1:] - frames[:-1]).astype(np.float32)
    motion = np.concatenate([[0.0], diff.mean(axis=(1, 2))]).astype(np.float32)
    cols = diff.sum(axis=1)  # 横方向のどこで動いているか
    centers = (cols * np.arange(w)[None, :]).sum(axis=1) / (cols.sum(axis=1) + 1e-6) / (w - 1)
    centers = np.concatenate([[0.5], centers]).astype(np.float32)
    # カメラの切り替え（画面全体が一気に変わる）は動きとして数えない
    cut_level = max(np.percentile(motion, 99.5), motion.mean() + 4 * motion.std())
    cuts = np.where(motion > max(cut_level, 30))[0]
    for c in cuts:
        motion[c] = np.median(motion[max(0, c - 5):c + 6])
    return fit(motion, n), fit(centers, n), [c * STEP for c in cuts]


# ------------------------------------------------------------
# 盛り上がりの点数
# ------------------------------------------------------------
def rank(arr):
    """0〜1 の順位に直す（大きいほど 1 に近い）"""
    np = _np()
    order = arr.argsort(kind="stable")
    r = np.empty(len(arr), dtype=np.float32)
    r[order] = np.linspace(0, 1, len(arr), dtype=np.float32)
    return r


def smooth(arr, seconds: float):
    np = _np()
    k = max(1, int(seconds / STEP))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr, kernel, mode="same")


def analyze(src: Path, meta: dict, cfg: dict) -> dict:
    np = _np()
    n = int(meta["duration"] / STEP)
    print("▶ 音を調べています（音量・ビート）")
    loud, flux, fine = measure_audio(src, meta, n)
    print("▶ 映像の動きを調べています（動画の長さの 1/20 くらいの時間がかかります）")
    motion, centers, scene_cuts = measure_motion(src, meta, n)
    # 無音に近いところ（曲の合間、MC の前後）は盛り上がりに数えない
    quiet = loud < np.percentile(loud, 10) + 3
    parts = {"loud": rank(smooth(loud, 1.0)), "beat": rank(smooth(flux, 1.0)), "motion": rank(smooth(motion, 1.0))}
    w = {k: float(cfg["weight_" + k]) for k in ("loud", "beat", "motion")}
    total = sum(w.values()) or 1.0
    score = sum(parts[k] * w[k] for k in parts) / total
    score[quiet] *= 0.3
    return {"score": smooth(score, 2.0), "parts": parts, "loud": loud, "fine": fine,
            "centers": centers, "scene_cuts": scene_cuts}


# ------------------------------------------------------------
# 見せ場を選ぶ
# ------------------------------------------------------------
def snap_to_beat(t: float, fine, reach: float = 1.0) -> float:
    """t の近くで、いちばん強く音が立ち上がる瞬間（ビート）に合わせる"""
    np = _np()
    flux, step = fine
    a, b = int(max(0, t - reach) / step), int((t + reach) / step)
    if b <= a or a >= len(flux):
        return t
    seg = flux[a:b]
    return (a + int(np.argmax(seg))) * step


def pick(an: dict, duration: float, cfg: dict, count_min: int, count_max: int,
         length_min: float, length_max: float) -> list:
    np = _np()
    score = an["score"]
    n = len(score)
    base = int(length_min / STEP)
    if n <= base:
        return [{"start": 0.0, "end": duration, "score": float(score.mean()) if n else 0.0}]
    csum = np.concatenate([[0.0], np.cumsum(score, dtype=np.float64)])
    means = (csum[base:] - csum[:-base]) / base     # 開始位置ごとの、length_min 秒の平均点
    used = np.zeros(n, dtype=bool)
    gap = int(2.0 / STEP)
    overall = float(score.mean())
    picks = []
    while len(picks) < count_max:
        cand = means.copy()
        for p in picks:  # 選んだものと重なる開始位置は使えない
            cand[max(0, p["i0"] - base - gap):min(len(cand), p["i1"] + gap)] = -1
        i0 = int(np.argmax(cand))
        if cand[i0] < 0:
            break
        if len(picks) >= count_min and cand[i0] < overall * 1.05:
            break  # 平均より盛り上がっていないものは出さない
        i1 = i0 + base
        level = cand[i0] * 0.9
        limit = int(length_max / STEP)
        blocked = lambda a, b: a < 0 or b > n or any(a < p["i1"] + gap and b > p["i0"] - gap for p in picks)
        # 盛り上がりが続く限り、前後に伸ばす（盛り上がりの始まりから終わりまでを入れる）
        while i1 - i0 + 10 <= limit:
            back = -1.0 if blocked(i0 - 10, i0) else float(score[i0 - 10:i0].mean())
            fwd = -1.0 if blocked(i1, i1 + 10) else float(score[i1:i1 + 10].mean())
            if max(back, fwd) < level:
                break
            if back >= fwd:
                i0 -= 10
            else:
                i1 += 10
        used[i0:i1] = True
        picks.append({"i0": i0, "i1": i1, "score": float(cand[i0])})
    clips = []
    for p in picks:
        start = snap_to_beat(p["i0"] * STEP, an["fine"])
        end = snap_to_beat(p["i1"] * STEP, an["fine"])
        start, end = max(0.0, start), min(duration, end)
        if end - start < length_min * 0.8:
            end = min(duration, start + length_min)
        a, b = p["i0"], p["i1"]
        parts = {k: float(v[a:b].mean()) for k, v in an["parts"].items()}
        clips.append({"start": start, "end": end, "score": p["score"], "parts": parts,
                      "center": float(np.median(an["centers"][a:b]))})
    return clips


def describe(parts: dict) -> str:
    """選んだ理由を、分かりやすい言葉にする（動画全体の中で、どのくらい大きいか）"""
    names = {"loud": "音", "beat": "ビート", "motion": "動き"}
    def level(v):
        return "とても大きい" if v >= 0.8 else "大きい" if v >= 0.6 else "ふつう" if v >= 0.4 else "小さめ"
    items = sorted(parts.items(), key=lambda kv: -kv[1])
    text = "、".join("{}が{}".format(names[k], level(v)) for k, v in items)
    if items[0][1] < 0.6:
        return "動画全体の中では比較的盛り上がっている場面（{}）".format(text)
    return "{}が特に目立つ場面（{}）".format(names[items[0][0]], text)


# ------------------------------------------------------------
# グラフ
# ------------------------------------------------------------
def draw_graph(path: Path, an: dict, clips: list, duration: float, font_path=None) -> None:
    """盛り上がりの点数と、選んだ見せ場を1枚の画像にする"""
    from PIL import Image, ImageDraw, ImageFont
    np = _np()
    W, H, pad = 1800, 520, 60
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    score = an["score"]
    x = lambda t: pad + (W - 2 * pad) * t / max(duration, 1e-6)
    y = lambda v: H - pad - (H - 2 * pad) * v
    for i, c in enumerate(clips, 1):
        d.rectangle([x(c["start"]), pad, x(c["end"]), H - pad], fill=(255, 224, 160))
        d.text((x(c["start"]) + 4, pad + 4), "{:02d}".format(i), fill=(160, 90, 0), font=font)
    step = max(1, len(score) // (W - 2 * pad))
    pts = [(x(i * STEP), y(float(score[i:i + step].mean()))) for i in range(0, len(score), step)]
    if len(pts) > 1:
        d.line(pts, fill=(40, 90, 200), width=2)
    d.line([pad, H - pad, W - pad, H - pad], fill=(120, 120, 120), width=1)
    minutes = duration / 60
    every = 1 if minutes <= 10 else 5 if minutes <= 60 else 10
    m = 0
    while m * 60 <= duration:
        d.line([x(m * 60), H - pad, x(m * 60), H - pad + 8], fill=(120, 120, 120))
        d.text((x(m * 60) - 10, H - pad + 12), "{}分".format(m), fill=(80, 80, 80), font=font)
        m += every
    d.text((pad, 16), "盛り上がり（青い線）と選んだ見せ場（オレンジ）", fill=(40, 40, 40), font=font)
    img.save(path)
