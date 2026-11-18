# -*- coding: utf-8 -*-
"""telop — タイトルテロップと字幕を作る部品（clip.py から使う）

文字は Pillow で透明な画像に描き、ffmpeg で動画に重ねます。
（Homebrew の ffmpeg には文字を描く機能が入っていないため、この方法にしています）
"""
from __future__ import annotations

import re
from pathlib import Path

from autocut import AutocutError

# Mac に最初から入っているフォント（上から順に探す）
MAC_FONTS_BOLD = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
]
MAC_FONTS_MEDIUM = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
]
OTHER_FONTS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# 行の先頭に来てはいけない文字（禁則）
NO_LINE_START = "、。，．,.！？!?」』）)ー〜…ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
# 字幕で区切る目印
BREAKS = "。、！？!?．，,. 　"

HALLUCINATIONS = ("大きな音楽", "おやすみなさい", "ご視聴ありがとうございました", "音楽", "拍手", "笑")


# ------------------------------------------------------------
# フォントと色
# ------------------------------------------------------------
def find_font(wanted: str, candidates: list) -> str:
    if wanted:
        path = Path(str(wanted)).expanduser()
        if not path.exists():
            raise AutocutError("config.toml で指定したフォントが見つかりません: {}".format(wanted))
        return str(path)
    for c in candidates + OTHER_FONTS:
        if Path(c).exists():
            return c
    raise AutocutError("日本語のフォントが見つかりません。config.toml の font_title / font_subtitle にフォントの場所を書いてください。")


def load_font(path: str, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(path, size)


def parse_color(text: str):
    """"#RRGGBB" または "#RRGGBBAA"（AA は透明度。00=透明, FF=不透明）"""
    t = str(text).strip().lstrip("#")
    if not t:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", t):
        raise AutocutError("色の書き方が正しくありません: {}（例: \"#FFFFFF\"）".format(text))
    vals = [int(t[i:i + 2], 16) for i in range(0, len(t), 2)]
    return tuple(vals + [255] * (4 - len(vals)))


# ------------------------------------------------------------
# 字幕の区切り方
# ------------------------------------------------------------
def split_text(text: str, max_chars: int) -> list:
    """1つの文を、max_chars 文字以内のかたまりに分ける"""
    text = re.sub(r"\s+", " ", text).strip()
    phrases, buf = [], ""
    for ch in text:
        buf += ch
        if ch in BREAKS:
            phrases.append(buf)
            buf = ""
    if buf:
        phrases.append(buf)
    chunks, cur = [], ""
    for ph in phrases:
        if len(cur) + len(ph.strip()) <= max_chars:
            cur += ph
            continue
        if cur.strip():
            chunks.append(cur)
        cur = ph
        while len(cur.strip()) > max_chars:  # 長すぎる区切りは、言葉の切れ目で分ける
            n = -(-len(cur) // max_chars)
            cut = best_cut(cur, len(cur) // n, max_chars)
            chunks.append(cur[:cut])
            cur = cur[cut:]
    if cur.strip():
        chunks.append(cur)
    return [c for c in (tidy(c) for c in chunks) if c]


PARTICLES = "はがをにでともへやのね"


def kind(ch: str) -> str:
    if "\u3040" <= ch <= "\u309f":
        return "hira"
    if "\u30a0" <= ch <= "\u30ff":
        return "kata"
    if "\u4e00" <= ch <= "\u9fff" or ch in "々〆":
        return "kanji"
    return "other"


_PARSER = []


def phrase_bounds(text: str):
    """BudouX（日本語の改行位置を決める部品）で、文節の切れ目の位置を返す。使えなければ None"""
    if not _PARSER:
        try:
            import budoux
            _PARSER.append(budoux.load_default_japanese_parser())
        except Exception:
            _PARSER.append(None)
    if _PARSER[0] is None:
        return None
    bounds, pos = set(), 0
    for ph in _PARSER[0].parse(text):
        pos += len(ph)
        bounds.add(pos)
    return bounds


def best_cut(text: str, target: int, max_chars: int, ok=None) -> int:
    """target の近くで、言葉の切れ目らしい位置を探す（助詞のあと、ひらがな→漢字/カタカナの境目）
    ok を渡すと、ok(前半) が True になる位置だけから選ぶ"""
    best, best_score = target, None
    bounds = phrase_bounds(text)
    for i in range(max(1, target - 8), min(len(text) - 1, max_chars) + 1):
        prev, nxt = text[i - 1], text[i]
        if nxt in NO_LINE_START or prev in "「『（(":
            continue
        if ok is not None and not ok(text[:i]):
            continue
        score = abs(i - target)
        depth = sum(1 for c in text[:i] if c in "「『（(") - sum(1 for c in text[:i] if c in "」』）)")
        if depth > 0:
            score += 6  # かっこの途中では切らない
        if prev in "」』）)" or nxt in "「『（(":
            score -= 7
        elif prev in "、，,！？!?　 ":
            score -= 6
        elif bounds is not None:
            score += -5 if i in bounds else 6
        elif prev in PARTICLES and kind(nxt) in ("kanji", "kata"):
            score -= 6
        elif kind(prev) == "hira" and kind(nxt) in ("kanji", "kata"):
            score -= 4
        elif kind(prev) != kind(nxt):
            score -= 1
        else:
            score += 3
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def tidy(text: str) -> str:
    """字幕として見せるときは、文末の「。」と読点のあとの空白をとる"""
    t = text.strip(" 　")
    t = re.sub(r"[。、，,．.]+$", "", t)
    return t.strip(" 　")


def subtitles_for_clip(segments: list, start: float, end: float, max_chars: int) -> list:
    """文字起こしから、クリップ内の時間（0秒始まり）の字幕 [(開始, 終了, 文字), ...] を作る"""
    subs = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        if seg["text"].strip(" 。、.!?！？()（）[]「」") in HALLUCINATIONS:
            continue
        chunks = split_text(seg["text"], max_chars)
        total = sum(len(c) for c in chunks) or 1
        t = seg["start"]
        for c in chunks:  # 文字数に合わせて時間を分ける
            d = (seg["end"] - seg["start"]) * len(c) / total
            a, b = max(t, start) - start, min(t + d, end) - start
            if b - a >= 0.2:
                subs.append([a, b, c])
            t += d
    # 次の字幕との間がほんの少しだけ空くときは、つなげてちらつきを防ぐ
    for cur, nxt in zip(subs, subs[1:]):
        if 0 < nxt[0] - cur[1] < 0.4:
            cur[1] = nxt[0]
        cur[1] = min(cur[1], nxt[0])
    return [tuple(s) for s in subs]


# ------------------------------------------------------------
# .srt（字幕ファイル）
# ------------------------------------------------------------
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def write_srt(path: Path, subs: list) -> None:
    lines = []
    for i, (a, b, text) in enumerate(subs, 1):
        lines += [str(i), "{} --> {}".format(srt_time(a), srt_time(b)), text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_srt(path: Path) -> list:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    subs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = re.match(r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", line)
            if m:
                v = [int(x) for x in m.groups()]
                a = v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1000
                b = v[4] * 3600 + v[5] * 60 + v[6] + v[7] / 1000
                body = "\n".join(lines[i + 1:]).strip()
                if body and b > a:
                    subs.append((a, b, body))
                break
    return subs


# ------------------------------------------------------------
# 文字の画像を作る
# ------------------------------------------------------------
def clean_break(left: str, right: str) -> bool:
    """left と right の間が、言葉の切れ目らしいか"""
    if not left or not right:
        return True
    a, b = left.rstrip(" 　")[-1:], right.lstrip(" 　")[:1]
    if not a or not b:
        return True
    if a in "、，,！？!?」』）)" or b in "「『（(":
        return True
    bounds = phrase_bounds(left + right)
    if bounds is not None:
        return len(left) in bounds
    return (a in PARTICLES or a in "、，,！？!?」』）)" or b in "「『（("
            or kind(a) != kind(b))


def wrap(draw, text: str, font, max_width: int, max_lines: int) -> list:
    """横幅に収まるように、言葉の切れ目で、行の長さがそろうように改行する。手で入れた改行はそのまま使う"""
    width = lambda x: draw.textlength(x.strip(" 　"), font=font)
    lines = []
    for para in text.split("\n"):
        para = para.strip(" 　")
        if not para:
            continue
        fallback = None
        for n in range(1, max_lines + 1):
            parts, rest = [], para
            for k in range(n, 1, -1):  # 残りを k 行に分けるとき、1行目をどこで切るか
                cut = best_cut(rest, round(len(rest) / k), len(rest) - 1,
                               ok=lambda x, r=rest, k=k: width(x) <= max_width
                               and width(r[len(x):]) <= max_width * (k - 1))
                parts.append(rest[:cut])
                rest = rest[cut:]
            parts.append(rest)
            if not all(width(x) <= max_width for x in parts):
                continue
            if all(clean_break(x, y) for x, y in zip(parts, parts[1:])):
                fallback = parts
                break
            fallback = fallback or parts  # 言葉の途中で切れるなら、もう1行増やせないか試す
        if fallback is None:
            return []  # 収まらないので、文字を小さくしてもらう
        lines += [x.strip(" 　") for x in fallback]
    if len(lines) > max_lines:
        return []
    return lines


def fit_lines(draw, text: str, font_path: str, size: int, max_width: int, max_lines: int):
    """収まるまで文字を少しずつ小さくする"""
    while True:
        font = load_font(font_path, size)
        lines = wrap(draw, text, font, max_width, max_lines)
        if lines or size <= 20:
            return font, (lines or [text])
        size = int(size * 0.92)


def draw_block(img, text: str, style: dict, center_y: float) -> None:
    """text を、画面の縦 center_y（0〜1）を中心に描く"""
    from PIL import Image, ImageDraw
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font, lines = fit_lines(draw, text, style["font"], style["size"], int(W * 0.9), style["max_lines"])
    stroke = style["stroke"]
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.12)
    widths = [draw.textlength(l, font=font) for l in lines]
    block_h = line_h * len(lines)
    top = int(H * center_y - block_h / 2)
    top = max(int(H * 0.02), min(top, H - block_h - int(H * 0.02)))
    if style.get("band"):  # 文字の後ろの帯
        pad_x, pad_y = int(style["size"] * 0.45), int(style["size"] * 0.25)
        band = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        bw = max(widths)
        bd.rounded_rectangle([W / 2 - bw / 2 - pad_x, top - pad_y, W / 2 + bw / 2 + pad_x, top + block_h + pad_y],
                             radius=int(style["size"] * 0.2), fill=style["band"])
        img.alpha_composite(band)
        draw = ImageDraw.Draw(img)
    for i, (line, w) in enumerate(zip(lines, widths)):
        draw.text((W / 2 - w / 2, top + i * line_h), line, font=font, fill=style["color"],
                  stroke_width=stroke, stroke_fill=style["outline"])


def styles(cfg: dict, w: int, h: int, shape: str) -> tuple:
    """config.toml の [title] [subtitle] から、この書き出しサイズでの見た目を決める"""
    scale = min(w, h) / 1080.0
    t, s = cfg["title"], cfg["subtitle"]
    tall = shape in ("tate", "feed")
    title = {
        "font": find_font(t.get("font", ""), MAC_FONTS_BOLD),
        "size": int(t["size"] * scale), "stroke": int(t["outline_width"] * scale),
        "color": parse_color(t["color"]), "outline": parse_color(t["outline"]),
        "band": parse_color(t.get("band", "")), "max_lines": 3,
        "y": (t["position_tate"] if tall else t["position_yoko"]) / 100.0,
    }
    sub = {
        "font": find_font(s.get("font", ""), MAC_FONTS_MEDIUM),
        "size": int(s["size"] * scale), "stroke": int(s["outline_width"] * scale),
        "color": parse_color(s["color"]), "outline": parse_color(s["outline"]),
        "band": parse_color(s.get("band", "")), "max_lines": 2,
        "y": (s["position_tate"] if tall else s["position_yoko"]) / 100.0,
    }
    return title, sub


def build_overlay(work: Path, w: int, h: int, length: float, title: str, title_secs: float,
                  subs: list, title_style: dict, sub_style: dict) -> Path:
    """時間ごとに「その時に見えている文字」を描いた透明な画像を作り、ffmpeg 用の並び順ファイルを返す"""
    from PIL import Image
    events = []
    if title and title_secs > 0:
        events.append((0.0, min(title_secs, length), "title", title))
    for a, b, text in subs:
        events.append((a, min(b, length), "sub", text))
    cuts = sorted({0.0, length} | {round(e[0], 3) for e in events} | {round(e[1], 3) for e in events})
    cuts = [c for c in cuts if 0 <= c <= length]
    work.mkdir(parents=True, exist_ok=True)
    entries, cache = [], {}
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 0.001:
            continue
        mid = (a + b) / 2
        visible = tuple((kind, text) for s, e, kind, text in events if s <= mid < e)
        if visible not in cache:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            for kind, text in visible:
                st = title_style if kind == "title" else sub_style
                draw_block(img, text, st, st["y"])
            path = work / "t{:04d}.png".format(len(cache))
            img.save(path)
            cache[visible] = path
        entries.append((cache[visible], b - a))
    lst = work / "list.txt"
    lines = ["ffconcat version 1.0"]
    for path, d in entries:
        lines += ["file '{}'".format(path.name), "duration {:.3f}".format(d)]
    lines.append("file '{}'".format(entries[-1][0].name))  # 最後の画像は2回書く決まり
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lst
