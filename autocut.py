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

DEFAULTS = {
    "silence": {"threshold_db": -35, "min_silence": 0.6, "pad_before": 0.15, "pad_after": 0.25},
    "youtube": {"width": 1920, "height": 1080, "fps": "source"},
    "encode": {"encoder": "auto", "video_bitrate": "12M", "crf": 18, "audio_bitrate": "192k"},
    "transcribe": {"model": "auto", "language": "ja"},
    "clips": {"count_min": 5, "count_max": 15, "length_min": 30, "length_max": 60,
              "model": "claude-opus-5-5", "channel_info": "", "shape": "tate"},
}

# これより短いカットは、つなぎ目が増えるだけなので行わない（秒）
MIN_CUT = 0.1

YES, NO = "はい", "いいえ"
CSV_COLUMNS = ["番号", "カットする", "開始", "終了", "長さ(秒)", "理由", "開始秒", "終了秒"]


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
        if val[0] in "\"'":
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
                        "{:.2f}".format(c["end"] - c["start"]), c["reason"],
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
            cuts.append({"start": start, "end": end, "reason": row.get("理由", ""), "enabled": enabled})
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


def render(src: Path, dst: Path, keeps: list, meta: dict, out_cfg: dict, enc_cfg: dict) -> None:
    fps = meta["fps"] if out_cfg["fps"] == "source" else snap_fps(float(out_cfg["fps"]))
    w, h = int(out_cfg["width"]), int(out_cfg["height"])
    hdr, color_args = hdr_filters(meta)
    vf = (["fps={}/{}".format(fps.numerator, fps.denominator),
           # コマの時刻は誤差を含むので、半コマ分ずらして判定する
           "select='{}'".format(select_expr(keeps, shift=float(1 / fps) / 2)),
           "setpts=N/FRAME_RATE/TB"]
          + hdr
          + ["scale={}:{}:force_original_aspect_ratio=decrease".format(w, h),
             "pad={}:{}:(ow-iw)/2:(oh-ih)/2".format(w, h), "setsar=1", "format=yuv420p"])
    # 音声は細かく刻んでから選ぶ（カットを重ねても映像と音がずれないように）
    af = ["aresample=48000", "asetnsamples=n=128:p=0",
          "aselect='{}'".format(select_expr(keeps)), "asetpts=N/SR/TB"]
    graph = "[0:{}]{}[v];[0:{}]{}[a]".format(
        meta["video_index"], ",".join(vf), meta["audio_index"], ",".join(af))
    run_encode(["-i", str(src)], graph, dst, sum(b - a for a, b in keeps), enc_cfg, color_args)


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="動画の無音部分を自動でカットして、YouTube用（16:9）に書き出します。")
    p.add_argument("video", type=Path, help="元の動画ファイル（mp4 / mov）")
    p.add_argument("--config", type=Path, help="設定ファイル（省略すると autocut.py と同じ場所の config.toml）")
    p.add_argument("--cuts", type=Path, help="手で直したカット一覧（CSV）を使って書き出し直す")
    p.add_argument("--list-only", action="store_true", help="動画は書き出さず、カット一覧だけ作る（設定の調整用）")
    args = p.parse_args(argv)

    src = args.video.expanduser()
    if not src.is_file():
        raise AutocutError(f"動画ファイルが見つかりません: {src}")
    cfg = load_config(args.config)
    out_dir = src.parent / (src.stem + "_autocut")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / (src.stem + "_cuts.csv")
    out_path = out_dir / (src.stem + "_youtube.mp4")

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

    print("▶ YouTube用（{}x{}）に書き出します".format(cfg["youtube"]["width"], cfg["youtube"]["height"]))
    render(src, out_path, keeps, meta, cfg["youtube"], cfg["encode"])
    print("✔ 完成しました")
    print("  動画      : " + str(out_path))
    if not args.cuts:
        print("  カット一覧: " + str(csv_path))
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
