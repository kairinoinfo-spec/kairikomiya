#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""autocut — 動画の無音部分を自動でカットして書き出すツール（第1段階）

使い方:
    python3 autocut.py 動画ファイル.mp4

必要なもの: ffmpeg（`brew install ffmpeg`）だけ。Python は Mac 標準のもので動きます。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    # ほかの部品が「import autocut」したときに、同じものを使うようにする（エラーの表示をそろえるため）
    sys.modules["autocut"] = sys.modules[__name__]

DEFAULTS = {
    "silence": {"threshold_db": -35, "min_silence": 0.6, "pad_before": 0.15, "pad_after": 0.25},
    "youtube": {"width": 1920, "height": 1080, "fps": "source", "burn_subtitles": True},
    "encode": {"encoder": "auto", "video_bitrate": "12M", "crf": 18, "audio_bitrate": "192k"},
    "transcribe": {"model": "auto", "language": "ja"},
    "cut": {"words": True, "fillers_enabled": True, "repeats_enabled": True, "restarts_enabled": True,
            "fillers": ["えー", "えーと", "えっと", "えーっと", "あー", "あのー", "あの", "そのー", "うーん", "んー", "まあ", "なんか"],
            "prompt": "えー、あの、えっと、まあ、そのー、うーん、なんか。",
            "speech_gaps": True, "speech_gap_min": 1.0, "speech_gap_pad_before": 0.25, "speech_gap_pad_after": 0.35,
            "word_pad": 0.02},
    "clips": {"count_min": 5, "count_max": 15, "length_min": 30, "length_max": 60,
              "model": "claude-opus-5-5", "channel_info": "", "shape": "tate"},
    "crop": {"follow": True},
    "dance": {"count_min": 5, "count_max": 12, "length_min": 20, "length_max": 45,
              "weight_loud": 1.0, "weight_beat": 1.0, "weight_motion": 1.0},
    "title": {"enabled": True, "seconds": 2.0, "size": 96, "color": "#FFE600", "outline": "#000000",
              "outline_width": 10, "band": "", "position_tate": 20, "position_yoko": 14, "font": ""},
    "subtitle": {"enabled": True, "size": 64, "color": "#FFFFFF", "outline": "#000000", "outline_width": 7,
                 "band": "", "max_chars": 24, "position_tate": 72, "position_yoko": 88, "font": ""},
}

# これより短いカットは、つなぎ目が増えるだけなので行わない（秒）
MIN_CUT = 0.1

YES, NO = "はい", "いいえ"
CSV_COLUMNS = ["番号", "カットする", "開始", "終了", "長さ(秒)", "理由", "内容", "開始秒", "終了秒"]


class AutocutError(Exception):
    """利用者に日本語で見せるエラー"""


# ------------------------------------------------------------
# 設定ファイル
# ------------------------------------------------------------
def _parse_simple_toml(text: str) -> dict:
    """古い Python（3.10以前）用の簡易読み込み。[section] と key = value だけ対応。"""
    data: dict = {}
    section = data
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([A-Za-z0-9_\-]+)\]\s*(#.*)?$", line)
        if m:
            section = data.setdefault(m.group(1), {})
            continue
        m = re.match(r"""^([A-Za-z0-9_\-]+)\s*=\s*("[^"]*"|'[^']*'|[^#]+?)\s*(#.*)?$""", line)
        if not m:
            raise AutocutError(f"設定ファイルの {lineno} 行目が読めません: {raw}")
        key, val = m.group(1), m.group(2).strip()
        if val[0] == "[":
            import json
            try:
                section[key] = json.loads(val)  # ["えー", "あの"] のような一覧
            except ValueError:
                raise AutocutError(f"設定ファイルの {lineno} 行目の一覧が読めません: {raw}")
        elif val[0] in "\"'":
            section[key] = val[1:-1]
        elif val in ("true", "false"):
            section[key] = val == "true"
        else:
            try:
                section[key] = int(val)
            except ValueError:
                try:
                    section[key] = float(val)
                except ValueError:
                    raise AutocutError(f"設定ファイルの {lineno} 行目の値が読めません: {raw}")
    return data


def load_config(path: Path | None) -> dict:
    config = {k: dict(v) for k, v in DEFAULTS.items()}
    if path is None:
        path = HERE / "config.toml"
        if not path.exists():
            return config
    elif not path.exists():
        raise AutocutError(f"設定ファイルが見つかりません: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11 以降

        try:
            loaded = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise AutocutError(f"設定ファイル {path.name} の書き方に間違いがあります: {e}")
    except ImportError:
        loaded = _parse_simple_toml(text)
    for section, values in loaded.items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
    return config


# ------------------------------------------------------------
# ffmpeg まわり
# ------------------------------------------------------------
def run(cmd: list) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        raise AutocutError(
            "ffmpeg が見つかりません。ターミナルで次を実行してインストールしてください:\n"
            "    brew install ffmpeg"
        )


def ffmpeg_has(kind: str, name: str) -> bool:
    """kind は 'encoders' か 'filters'"""
    out = run(["ffmpeg", "-hide_banner", "-" + kind]).stdout
    return re.search(r"\s" + re.escape(name) + r"\s", out) is not None


STANDARD_FPS = [
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
    Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60),
]


def snap_fps(value: float) -> Fraction:
    """iPhone の動画はコマ数が微妙に揺れるので、近い標準値にそろえる"""
    best = min(STANDARD_FPS, key=lambda f: abs(float(f) - value))
    if abs(float(best) - value) / value < 0.02:
        return best
    return Fraction(value).limit_denominator(1001)


def probe(path: Path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if r.returncode != 0:
        raise AutocutError(f"動画ファイルを読めませんでした: {path}\n{r.stderr.strip()}")
    info = json.loads(r.stdout)
    video = audio = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") and video is None:
            video = s
        if s.get("codec_type") == "audio" and audio is None:
            audio = s
    if video is None:
        raise AutocutError("このファイルには映像が入っていません。")
    if audio is None:
        raise AutocutError("このファイルには音声が入っていないため、無音カットができません。")

    def rate(key):
        try:
            return float(Fraction(video.get(key, "0/0")))
        except (ZeroDivisionError, ValueError):
            return 0.0

    fps = rate("avg_frame_rate") or rate("r_frame_rate") or 30.0
    # iPhone の縦撮りは「横向きの映像＋回転の指示」で記録されているので、見た目の向きに直す
    rotation = 0
    for side in video.get("side_data_list", []):
        rotation = side.get("rotation", rotation)
    rotation = int(float(video.get("tags", {}).get("rotate", rotation)))
    width, height = video.get("width"), video.get("height")
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return {
        "duration": float(info["format"]["duration"]),
        "fps": snap_fps(fps),
        "video_index": video["index"],
        "audio_index": audio["index"],
        "width": width,
        "height": height,
        "hdr": video.get("color_transfer") in ("arib-std-b67", "smpte2084"),
    }


def detect_silence(path: Path, meta: dict, cfg: dict):
    """無音の区間 [(開始, 終了), ...] と、動画全体の音量（平均, 最大）を返す"""
    af = "silencedetect=noise={}dB:d={},volumedetect".format(cfg["threshold_db"], cfg["min_silence"])
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-map", "0:{}".format(meta["audio_index"]), "-af", af, "-f", "null", "-"])
    if r.returncode != 0:
        raise AutocutError("音声の解析に失敗しました:\n" + r.stderr[-1500:])
    silences, start = [], None
    for m in re.finditer(r"silence_(start|end): (-?[\d.]+)", r.stderr):
        if m.group(1) == "start":
            start = max(0.0, float(m.group(2)))
        elif start is not None:
            silences.append((start, min(float(m.group(2)), meta["duration"])))
            start = None
    if start is not None:  # 最後まで無音のまま終わった場合
        silences.append((start, meta["duration"]))

    def vol(name):
        m = re.search(name + r": (-?[\d.]+) dB", r.stderr)
        return float(m.group(1)) if m else None

    return silences, vol("mean_volume"), vol("max_volume")


# ------------------------------------------------------------
# カット一覧
# ------------------------------------------------------------
def build_cuts(silences, meta: dict, cfg: dict) -> list:
    """無音区間から、余白を残した「カットする区間」を作る"""
    fps, duration = meta["fps"], meta["duration"]

    def snap(t):  # 映像のコマの境目にそろえる
        return float(round(t * fps) / fps)

    cuts = []
    for s, e in silences:
        cs = 0.0 if s <= 0.01 else s + cfg["pad_after"]
        ce = duration if e >= duration - 0.01 else e - cfg["pad_before"]
        cs, ce = snap(cs), min(snap(ce), duration)
        if ce - cs >= MIN_CUT:
            cuts.append({"start": cs, "end": ce, "reason": "無音", "enabled": True})
    return cuts


def word_cuts(src: Path, meta: dict, cfg: dict, out_dir: Path, volume_cuts: list) -> list:
    """文字起こしから、「えー」「あの」・繰り返し・言い直し・喋っていない間のカットを作る"""
    import json
    import speech
    cc = cfg["cut"]
    cache = out_dir / (base_stem(src, "_autocut") + "_words.json")
    if cache.exists():
        segments = json.loads(cache.read_text(encoding="utf-8"))
        print("▶ 前回の文字起こしを使います: " + cache.name)
    else:
        segments = speech.transcribe(src, meta, cfg["transcribe"], prompt=str(cc.get("prompt", "")), chars=True)
        cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        (out_dir / (base_stem(src, "_autocut") + "_文字起こし.txt")).write_text(
            "\n".join("[{}] {}".format(fmt_time(s["start"]), s["text"]) for s in segments) + "\n", encoding="utf-8")
    fps, duration = meta["fps"], meta["duration"]

    def snap(t):
        return float(round(t * fps) / fps)

    cuts = []
    pad = float(cc["word_pad"])
    for c in speech.find_word_cuts(segments, cc):
        a, b = snap(max(0.0, c["start"] - pad)), snap(min(duration, c["end"] + pad))
        if b - a >= MIN_CUT:
            cuts.append({"start": a, "end": b, "reason": c["reason"], "text": c["text"], "enabled": True})
    if cc["speech_gaps"]:
        gap_cfg = {"min_silence": cc["speech_gap_min"], "pad_before": cc["speech_gap_pad_before"],
                   "pad_after": cc["speech_gap_pad_after"]}
        for g in build_cuts(speech.find_speech_gaps(segments, duration, gap_cfg), meta, gap_cfg):
            # 音量で見つけた無音と重なるものは、そちらに任せる
            if not any(v["start"] <= g["end"] and g["start"] <= v["end"] for v in volume_cuts):
                g["reason"] = "声なし"
                cuts.append(g)
    return cuts


def make_subtitles(src: Path, out_dir: Path, cuts: list, keeps: list, cfg: dict, args):
    """字幕 [(開始, 終了, 文字), ...] と、書き出した .srt の場所を返す。字幕を入れないときは ([], None)"""
    import json
    if args.no_subs or not cfg["subtitle"]["enabled"]:
        return [], None
    import speech
    import telop
    if args.srt:
        srt = args.srt.expanduser()
        if not srt.exists():
            raise AutocutError("字幕ファイルが見つかりません: {}".format(srt))
        speech.ensure_module("PIL")
        subs = telop.read_srt(srt)
        print("▶ 字幕ファイルを使います: {}（{}個）".format(srt.name, len(subs)))
        return subs, None
    cache = out_dir / (base_stem(src, "_autocut") + "_words.json")
    if not cache.exists():
        if not args.quick:
            print("  ※ 文字起こしがないので、字幕は入れません")
        return [], None
    speech.ensure_module("PIL")  # 文字を描く部品
    segments = json.loads(cache.read_text(encoding="utf-8"))
    subs = telop.subtitles_after_cuts(segments, cuts, keeps, int(cfg["subtitle"]["max_chars"]))
    srt_path = out_dir / (base_stem(src, "_autocut") + "_字幕.srt")
    if srt_path.exists():
        n = 1
        while srt_path.with_name("{}_old{}.srt".format(srt_path.stem, n)).exists():
            n += 1
        srt_path.rename(srt_path.with_name("{}_old{}.srt".format(srt_path.stem, n)))
    telop.write_srt(srt_path, subs)
    print("▶ 字幕を {} 個作りました（カットしたあとの時間に合わせています）".format(len(subs)))
    return subs, srt_path


def merge_cuts(cuts: list) -> list:
    """重なったカットを1つにまとめる"""
    merged = []
    for c in sorted(cuts, key=lambda c: c["start"]):
        if merged and c["start"] <= merged[-1]["end"] + 0.05:
            m = merged[-1]
            m["end"] = max(m["end"], c["end"])
            if c["reason"] not in m["reason"].split("・"):
                m["reason"] += "・" + c["reason"]
            if c.get("text"):
                m["text"] = (m.get("text", "") + " " + c["text"]).strip()
        else:
            merged.append(dict(c))
    return merged


def base_stem(src: Path, suffix: str) -> str:
    """作るフォルダやファイルの名前のもとにする、動画の名前。
    長すぎると Mac で保存できないので短くする（前に長い名前で作ったフォルダがあれば、そのまま使う）"""
    if len(src.stem) <= 40 or (src.parent / (src.stem + suffix)).exists():
        return src.stem
    return src.stem[:40].rstrip(" 　_-")


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return "{:02d}:{:05.2f}".format(int(m), s)


def write_cuts_csv(path: Path, cuts: list) -> None:
    # utf-8-sig: Numbers でも Excel でも文字化けしない
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, c in enumerate(cuts, 1):
            w.writerow([i, YES if c["enabled"] else NO, fmt_time(c["start"]), fmt_time(c["end"]),
                        "{:.2f}".format(c["end"] - c["start"]), c["reason"], c.get("text", ""),
                        "{:.3f}".format(c["start"]), "{:.3f}".format(c["end"])])


def read_cuts_csv(path: Path) -> list:
    if not path.exists():
        raise AutocutError(f"カット一覧が見つかりません: {path}")
    cuts = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            try:
                start, end = float(row["開始秒"]), float(row["終了秒"])
            except (KeyError, TypeError, ValueError):
                raise AutocutError(f"カット一覧の {n} 行目が読めません（「開始秒」「終了秒」の列は消さないでください）")
            flag = (row.get("カットする") or "").strip().lower()
            enabled = flag not in (NO, "no", "n", "0", "false", "×", "x", "")
            cuts.append({"start": start, "end": end, "reason": row.get("理由", ""),
                         "text": row.get("内容", "") or "", "enabled": enabled})
    return sorted(cuts, key=lambda c: c["start"])


def keep_segments(cuts: list, duration: float) -> list:
    """カットしない（残す）区間の一覧"""
    keeps, pos = [], 0.0
    for c in cuts:
        if not c["enabled"]:
            continue
        if c["start"] > pos:
            keeps.append((pos, c["start"]))
        pos = max(pos, c["end"])
    if pos < duration:
        keeps.append((pos, duration))
    return keeps


# ------------------------------------------------------------
# 書き出し
# ------------------------------------------------------------
def select_expr(keeps: list, shift: float = 0.0) -> str:
    return "+".join("gte(t,{:.6f})*lt(t,{:.6f})".format(a - shift, b - shift) for a, b in keeps)


def choose_encoder(cfg: dict) -> list:
    """試す順番に並べた書き出し方式のリスト"""
    vt = ["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-profile:v", "high", "-b:v", str(cfg["video_bitrate"])]
    x264 = ["-c:v", "libx264", "-preset", "medium", "-crf", str(cfg["crf"])]
    has_vt = ffmpeg_has("encoders", "h264_videotoolbox")
    has_x264 = ffmpeg_has("encoders", "libx264")
    want = str(cfg["encoder"]).lower()
    if want == "x264":
        order = [x264] if has_x264 else [vt]
    else:  # auto / videotoolbox
        order = ([vt] if has_vt else []) + ([x264] if has_x264 else [])
    if not order:
        raise AutocutError("この ffmpeg には H.264 の書き出し機能がありません。`brew reinstall ffmpeg` を試してください。")
    return order


def hdr_filters(meta: dict):
    """HDR の動画をふつうの明るさ（SDR）に直すフィルタと、書き出し時の色の指定を返す"""
    if not meta["hdr"]:
        return [], []
    if not ffmpeg_has("filters", "zscale"):
        print("  ⚠ この動画は HDR で撮影されています。色が白っぽく（薄く）なる場合は教えてください。")
        return [], []
    return (["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
             "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"],
            ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])


def run_encode(input_args: list, graph: str, dst: Path, out_duration: float,
               enc_cfg: dict, color_args: list) -> None:
    """ffmpeg で書き出す。graph は [v] と [a] を出力するフィルタ。進み具合を表示する"""
    tmp = dst.with_name(dst.stem + ".part.mp4")
    last_error = ""
    for enc in choose_encoder(enc_cfg):
        cmd = (["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-nostats"] + input_args
               + ["-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
               + enc + color_args
               + ["-c:a", "aac", "-b:a", str(enc_cfg["audio_bitrate"]),
                  "-movflags", "+faststart", "-progress", "pipe:1", str(tmp)])
        with tempfile.TemporaryFile(mode="w+", errors="replace") as errlog:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errlog, text=True)
            for line in proc.stdout:
                m = re.match(r"out_time_(?:us|ms)=(\d+)", line)
                if m and out_duration > 0:
                    pct = min(100, int(int(m.group(1)) / 1e6 / out_duration * 100))
                    print("\r  書き出し中… {:3d}%".format(pct), end="", flush=True)
            proc.wait()
            if proc.returncode == 0:
                print("\r  書き出し中… 100%")
                tmp.replace(dst)
                return
            print()
            errlog.seek(0)
            last_error = errlog.read()[-1500:]
        if tmp.exists():
            tmp.unlink()
        print("  （{} での書き出しに失敗したので、別の方式で試します）".format(enc[1]))
    raise AutocutError("書き出しに失敗しました:\n" + last_error)


def render(src: Path, dst: Path, keeps: list, meta: dict, out_cfg: dict, enc_cfg: dict, overlay=None) -> None:
    """overlay: 字幕を描いた画像の並び順ファイル（カットしたあとの時間で作ったもの）"""
    fps = meta["fps"] if out_cfg["fps"] == "source" else snap_fps(float(out_cfg["fps"]))
    w, h = int(out_cfg["width"]), int(out_cfg["height"])
    hdr, color_args = hdr_filters(meta)
    vf = (["fps={}/{}".format(fps.numerator, fps.denominator),
           # コマの時刻は誤差を含むので、半コマ分ずらして判定する
           "select='{}'".format(select_expr(keeps, shift=float(1 / fps) / 2)),
           "setpts=N/FRAME_RATE/TB"]
          + hdr
          + ["scale={}:{}:force_original_aspect_ratio=decrease".format(w, h),
             "pad={}:{}:(ow-iw)/2:(oh-ih)/2".format(w, h), "setsar=1"])
    # 音声は細かく刻んでから選ぶ（カットを重ねても映像と音がずれないように）
    af = ["aresample=48000", "asetnsamples=n=128:p=0",
          "aselect='{}'".format(select_expr(keeps)), "asetpts=N/SR/TB"]
    inputs = ["-i", str(src)]
    if overlay:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(overlay)]
        graph = ("[0:{}]{}[base];[1:v]format=rgba[txt];[base][txt]overlay=0:0:eof_action=repeat,format=yuv420p[v];"
                 "[0:{}]{}[a]").format(meta["video_index"], ",".join(vf), meta["audio_index"], ",".join(af))
    else:
        graph = "[0:{}]{},format=yuv420p[v];[0:{}]{}[a]".format(
            meta["video_index"], ",".join(vf), meta["audio_index"], ",".join(af))
    run_encode(inputs, graph, dst, sum(b - a for a, b in keeps), enc_cfg, color_args)


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="動画の無音・「えー」「あの」・言い直しを自動でカットして、YouTube用（16:9）に書き出します。")
    p.add_argument("video", type=Path, help="元の動画ファイル（mp4 / mov）")
    p.add_argument("--config", type=Path, help="設定ファイル（省略すると autocut.py と同じ場所の config.toml）")
    p.add_argument("--cuts", type=Path, help="手で直したカット一覧（CSV）を使って書き出し直す")
    p.add_argument("--list-only", action="store_true", help="動画は書き出さず、カット一覧だけ作る（設定の調整用）")
    p.add_argument("--quick", action="store_true", help="文字起こしをせず、音量による無音カットだけにする（速い）")
    p.add_argument("--no-subs", action="store_true", help="字幕を入れない")
    p.add_argument("--srt", type=Path, help="手で直した字幕ファイル（.srt）を使って、字幕を入れる")
    args = p.parse_args(argv)

    src = args.video.expanduser()
    if not src.is_file():
        raise AutocutError(f"動画ファイルが見つかりません: {src}")
    cfg = load_config(args.config)
    if not args.no_subs and cfg["subtitle"]["enabled"] and not args.list_only and (not args.quick or args.srt):
        import speech
        # 字幕の文字を描く部品が入った Python に、最初に切り替えておく（途中で切り替えると最初からやり直しになるため）
        speech.ensure_module("PIL", required=False)
    if src.stem.endswith("_youtube") and src.parent.name.endswith("_autocut"):
        raise AutocutError("これは、このツールで書き出したあとの動画です。\n"
                           "カットする前の、元の動画を読み込ませてください（_autocut フォルダの外にあります）。")
    stem = base_stem(src, "_autocut")
    out_dir = src.parent / (stem + "_autocut")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / (stem + "_cuts.csv")
    out_path = out_dir / (stem + "_youtube.mp4")

    print("▶ 動画を調べています: " + src.name)
    meta = probe(src)
    print("  長さ {} / {}x{} / {:.2f}コマ毎秒{}".format(
        fmt_time(meta["duration"]), meta["width"], meta["height"], float(meta["fps"]),
        " / HDR" if meta["hdr"] else ""))

    if args.cuts:
        cuts = read_cuts_csv(args.cuts.expanduser())
        print("▶ カット一覧を読み込みました: {}（{}か所のうち {}か所をカット）".format(
            args.cuts.name, len(cuts), sum(c["enabled"] for c in cuts)))
    else:
        print("▶ 無音部分を探しています…")
        silences, mean, peak = detect_silence(src, meta, cfg["silence"])
        if mean is not None and peak is not None:
            print("  この動画の音量: 平均 {:.0f} dB / 最大 {:.0f} dB（無音の基準: {} dB）".format(
                mean, peak, cfg["silence"]["threshold_db"]))
        cuts = build_cuts(silences, meta, cfg["silence"])
        for c in cuts:
            c["text"] = ""
        if cfg["cut"]["words"] and not args.quick:
            import speech
            if speech.whisper_exe():
                extra = word_cuts(src, meta, cfg, out_dir, cuts)
                counts = {}
                for c in extra:
                    counts[c["reason"]] = counts.get(c["reason"], 0) + 1
                print("  文字起こしから見つけたカット: " + ("、".join(
                    "{} {}か所".format(k, v) for k, v in counts.items()) or "なし"))
                cuts = merge_cuts(cuts + extra)
            else:
                print("  ※ 文字起こしツールがないので、無音カットだけにします（setup.sh を実行すると、言い間違いもカットできます）")
        if csv_path.exists():  # 手で直した一覧を消さないように、前の一覧は残す
            n = 1
            while csv_path.with_name("{}_old{}.csv".format(csv_path.stem, n)).exists():
                n += 1
            old = csv_path.with_name("{}_old{}.csv".format(csv_path.stem, n))
            csv_path.rename(old)
            print("  前のカット一覧は {} として残しました".format(old.name))
        write_cuts_csv(csv_path, cuts)

    keeps = keep_segments(cuts, meta["duration"])
    kept = sum(b - a for a, b in keeps)
    print("  カット {}か所 / {} → {}（{} 短縮）".format(
        sum(c["enabled"] for c in cuts), fmt_time(meta["duration"]), fmt_time(kept),
        fmt_time(meta["duration"] - kept)))
    if not keeps:
        raise AutocutError("全部が無音と判定されました。config.toml の threshold_db をもっと小さい数字（例: -45）にしてください。")

    if args.list_only:
        print("✔ カット一覧だけ作りました: " + str(csv_path))
        return 0

    subs, srt_path = make_subtitles(src, out_dir, cuts, keeps, cfg, args)
    print("▶ YouTube用（{}x{}）に書き出します".format(cfg["youtube"]["width"], cfg["youtube"]["height"]))
    with tempfile.TemporaryDirectory() as work:
        overlay = None
        if subs and cfg["youtube"].get("burn_subtitles", True):
            import telop
            w, h = int(cfg["youtube"]["width"]), int(cfg["youtube"]["height"])
            _, sub_style = telop.styles(cfg, w, h, "yoko")
            overlay = telop.build_overlay(Path(work), w, h, kept, "", 0, subs, sub_style, sub_style)
        render(src, out_path, keeps, meta, cfg["youtube"], cfg["encode"], overlay)
    print("✔ 完成しました")
    print("  動画      : " + str(out_path))
    if not args.cuts:
        print("  カット一覧: " + str(csv_path))
    if srt_path:
        print("  字幕      : " + str(srt_path))
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
