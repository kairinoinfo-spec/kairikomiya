# -*- coding: utf-8 -*-
"""fcpxml — カットした結果を、Final Cut Pro で開けるタイムライン（FCPXML）にする部品

元の動画を参照するだけなので、画質は落ちません。カットした場所には目印（マーカー）を付けます。
DaVinci Resolve でも読み込めます。
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


def timecode_seconds(tc, fps: Fraction) -> Fraction:
    """「01:00:00:00」のような時刻を秒にする（読めなければ 0）"""
    if not tc:
        return Fraction(0)
    m = re.match(r"^(\d+):(\d+):(\d+)[:;.](\d+)$", str(tc).strip())
    if not m:
        return Fraction(0)
    h, mi, s, f = (int(x) for x in m.groups())
    nominal = round(float(fps))  # 29.97 の動画でも、タイムコードは 30 コマで数える
    frames = ((h * 60 + mi) * 60 + s) * nominal + f
    return Fraction(frames) / fps


FORMAT_RATES = {Fraction(24000, 1001): "2398", Fraction(24): "24", Fraction(25): "25",
                Fraction(30000, 1001): "2997", Fraction(30): "30", Fraction(50): "50",
                Fraction(60000, 1001): "5994", Fraction(60): "60"}
FORMAT_SIZES = {(1280, 720): "720p", (1920, 1080): "1080p", (3840, 2160): "2160p"}


def format_name(meta: dict, fps: Fraction) -> str:
    """Final Cut Pro の標準の形式名（例: FFVideoFormat1080p2398）。標準でなければ空"""
    size = FORMAT_SIZES.get((meta["width"], meta["height"]))
    rate = FORMAT_RATES.get(fps)
    if not size or not rate:
        return ""
    return "FFVideoFormat{}{}".format(size, rate)


def write_fcpxml(path: Path, src: Path, meta: dict, keeps: list, cuts: list, title: str) -> None:
    fps = Fraction(meta["fps"])
    frame = 1 / fps  # 1コマの長さ（秒）

    def frames(t) -> int:
        return int(round(Fraction(t) * fps))

    def tm(n: int) -> str:
        """コマ数を FCPXML の時間（1コマの長さ 1001/24000s などの何倍か）にする"""
        if n == 0:
            return "0s"
        return "{}/{}s".format(n * frame.numerator, frame.denominator)

    tc0 = frames(timecode_seconds(meta.get("timecode"), fps))
    total = frames(meta["duration"])
    fd = "{}/{}s".format(frame.numerator, frame.denominator)
    name = escape(src.stem)
    channels = int(meta.get("channels", 2))
    rate = int(meta.get("sample_rate", 48000))
    layout = "mono" if channels == 1 else "stereo"
    rate_name = {32000: "32k", 44100: "44.1k", 48000: "48k", 88200: "88.2k", 96000: "96k"}.get(rate, "48k")

    # カットした場所の目印: カットのすぐ後のクリップの頭に付ける
    enabled = sorted((c for c in cuts if c["enabled"]), key=lambda c: c["start"])
    clips = []
    offset = 0
    for a, b in keeps:
        fa, fb = frames(a), frames(b)
        if fb <= fa:
            continue
        notes = [c for c in enabled if abs(frames(c["end"]) - fa) <= 1]
        clips.append((fa, fb - fa, offset, notes))
        offset += fb - fa

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.10">',
        "  <resources>",
        '    <format id="r1"{} frameDuration="{}" width="{}" height="{}" colorSpace="1-1-1 (Rec. 709)"/>'.format(
            ' name="{}"'.format(format_name(meta, fps)) if format_name(meta, fps) else "", fd, meta["width"], meta["height"]),
        '    <asset id="r2" name={} start="{}" duration="{}" hasVideo="1" format="r1" hasAudio="1" '
        'audioSources="1" audioChannels="{}" audioRate="{}">'.format(
            quoteattr(src.stem), tm(tc0), tm(total), channels, rate),
        '      <media-rep kind="original-media" src={}/>'.format(quoteattr(src.resolve().as_uri())),
        "    </asset>",
        "  </resources>",
        "  <library>",
        '    <event name={}>'.format(quoteattr("autocut")),
        '      <project name={}>'.format(quoteattr(title)),
        '        <sequence format="r1" duration="{}" tcStart="0s" tcFormat="NDF" audioLayout="{}" audioRate="{}">'.format(
            tm(offset), layout, rate_name),
        "          <spine>",
    ]
    for i, (start, dur, off, notes) in enumerate(clips, 1):
        attrs = 'ref="r2" offset="{}" name="{}" start="{}" duration="{}" format="r1" tcFormat="NDF"'.format(
            tm(off), name, tm(tc0 + start), tm(dur))
        if not notes:
            lines.append("            <asset-clip {}/>".format(attrs))
            continue
        lines.append("            <asset-clip {}>".format(attrs))
        for c in notes:
            label = "カット: " + c["reason"] + ("（{}）".format(c["text"]) if c.get("text") else "")
            lines.append('              <marker start="{}" duration="{}" value={}/>'.format(
                tm(tc0 + start), fd, quoteattr(label)))
        lines.append("            </asset-clip>")
    lines += [
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
