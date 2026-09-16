# -*- coding: utf-8 -*-
"""
版面层：符号修复、文本行重建、表格还原、图区定位、裁图。

这一层只跟 PDF 的坐标打交道，不生产节点——节点由 parser.py 组装。
所有规则都来自对 JGJ/T 231-2021 的实测（12 类排印陷阱）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

# Symbol 字体的私有区编码：实测 14 个码位，覆盖原 PDF 与测试件
PUA_SYMBOLS = {
    "\uf03d": "=", "\uf067": "γ", "\uf0a3": "≤", "\uf02b": "+",
    "\uf0e5": "∑", "\uf0b4": "×", "\uf06a": "φ", "\uf06d": "μ",
    "\uf0d7": "·", "\uf062": "β", "\uf068": "η", "\uf06c": "λ",
    "\uf0b3": "≥", "\uf053": "∑",
}

# 行首条号 / 节号 / 表标题 / 图题 / 公式号 / 图注
CLAUSE_RE = re.compile(r"^(\d+\.\d+\.\d+)\s*(?![）)、。；：，])")
SECTION_RE = re.compile(r"^(\d+\.\d+)(?:\s+|(?=\S))([^\d].*)$")
CHAPTER_RE = re.compile(r"^(\d+)\s+(\S.*)$")
TABLE_TITLE_RE = re.compile(r"^(?:按)?表\s*([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)\s*(.*)$")
FIGURE_TITLE_RE = re.compile(r"^图\s*(\d+(?:\.\d+)*(?:-\d+)?)\s+(.*)$")
EQNUM_RE = re.compile(r"[（(]\s*(\d+\.\d+\.\d+(?:-\d+)?)\s*[）)]")
# 公式编号必须出现在行尾，才是公式本体；出现在行中说明只是引用
EQNUM_TAIL_RE = re.compile(r"[（(]\s*\d+\.\d+\.\d+(?:-\d+)?\s*[）)]\s*$")
ITEM_RE = re.compile(r"^(\d{1,2})\s*[）)]?\s*(\S.*)$")
LEGEND_RE = re.compile(r"^\d+\s*[—\-－]")


def fix_symbols(text: str) -> str:
    """修掉 Symbol 字体的私有区编码，让公式符号变回可读字符。"""
    if not text:
        return ""
    return "".join(PUA_SYMBOLS.get(ch, ch) for ch in text)


def norm_text(text: str, collapse_spaces: bool = True) -> str:
    """
    保守规范化：修符号 + 统一换行 + 压缩行内连续空格。

    故意不做 NFKC——那会把中文全角标点变半角、破坏原文面貌。
    行的切分保留：条文里的"1 2 3"款号和公式行都有语义。
    """
    if not text:
        return ""
    text = fix_symbols(text).replace("\r\n", "\n").replace("\r", "\n")
    if collapse_spaces:
        text = re.sub(r"[ \t\u3000]{2,}", " ", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def norm_dash(text: str) -> str:
    """
    图注里的构件编号破折号归一化：原文混用了 U+FF0D（全文仅 3 处，都在图 5.1.4 图注里）
    和 U+2014（全文 90 处）。不归一化的话，用户按常规打长破折号会精确匹配失败。
    """
    return re.sub(r"(?<=\d)\s*[－–—]\s*(?=[\u4e00-\u9fff])", "—", text or "")


# ---------------------------------------------------------------- 文本行重建

def page_lines(page, y_tol: float = 4.0) -> list[dict]:
    """
    把字符按纵坐标聚成行。中文没有空格分词，所以必须用字符坐标而不是 extract_words。
    返回按 (top, x0) 排序的行，每行带 text 与 bbox。
    """
    chars = [dict(c, text=fix_symbols(c["text"])) for c in page.chars]
    rows: list[list[dict]] = []
    for ch in sorted(chars, key=lambda c: ((c["top"] + c["bottom"]) / 2, c["x0"])):
        center = (ch["top"] + ch["bottom"]) / 2
        if rows:
            ref = sum((c["top"] + c["bottom"]) / 2 for c in rows[-1]) / len(rows[-1])
            if abs(center - ref) > y_tol:
                rows.append([])
        else:
            rows.append([])
        rows[-1].append(ch)

    out = []
    for group in rows:
        group.sort(key=lambda c: c["x0"])
        text = fix_symbols("".join(c["text"] for c in group)).strip()
        if not text:
            continue
        out.append({
            "text": text,
            "x0": round(min(c["x0"] for c in group), 2),
            "x1": round(max(c["x1"] for c in group), 2),
            "top": round(min(c["top"] for c in group), 2),
            "bottom": round(max(c["bottom"] for c in group), 2),
            "chars": group,
        })
    out.sort(key=lambda l: (l["top"], l["x0"]))
    return out


def drop_running_heads(lines: list[dict], page) -> list[dict]:
    """页脚页码：纯数字且落在页面底部 80pt 内。页码本身只进 pages 字段，不进 content。"""
    return [l for l in lines
            if not (re.fullmatch(r"\d{1,3}", l["text"]) and l["top"] > page.height - 80)]


def repair_broken_lines(lines: list[dict]) -> list[dict]:
    """
    排印陷阱：上一行以"（图"或"（表"结尾，下一行以编号+右括号开头
    （如"…悬臂长度（图" + "6.2.4）不应超过 650mm…"）。合并回一行，避免条号误判。
    """
    out: list[dict] = []
    for line in lines:
        if out and re.search(r"[（(](图|表|式)$", out[-1]["text"]) and re.match(
                r"^[A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?\s?[）)]", line["text"]):
            prev = out[-1]
            prev["text"] += line["text"]
            prev["x1"] = max(prev["x1"], line["x1"])
            prev["bottom"] = max(prev["bottom"], line["bottom"])
            prev["chars"] += line["chars"]
            continue
        out.append(line)
    return out


def line_bbox(lines: Sequence[dict]) -> list[float]:
    return [round(min(l["x0"] for l in lines), 2), round(min(l["top"] for l in lines), 2),
            round(max(l["x1"] for l in lines), 2), round(max(l["bottom"] for l in lines), 2)]


def in_box(line: dict, box: Sequence[float], pad: float = 2.0) -> bool:
    return (line["x0"] >= box[0] - pad and line["x1"] <= box[2] + pad
            and line["top"] >= box[1] - pad and line["bottom"] <= box[3] + pad)


# ------------------------------------------------------------------ 表格还原

def cell_text(page, bbox) -> str:
    """
    按字符坐标重建单元格：下标直接贴回主字符（G1+G2 而不是 G 1 +G 2），
    只有横向间隔 > 9pt 才补空格；单元格内多行用 " / " 连接。
    """
    if not bbox:
        return ""
    x0, top, x1, bottom = bbox
    chars = [c for c in page.chars
             if c["x0"] >= x0 - 0.6 and c["x1"] <= x1 + 0.6
             and c["top"] >= top - 0.6 and c["bottom"] <= bottom + 0.6]
    if not chars:
        return ""
    rows: list[dict] = []
    for ch in sorted(chars, key=lambda c: ((c["top"] + c["bottom"]) / 2, c["x0"])):
        center = (ch["top"] + ch["bottom"]) / 2
        for row in rows:
            if abs(center - row["center"]) <= 5.0:
                row["chars"].append(ch)
                break
        else:
            rows.append({"center": center, "chars": [ch]})
    parts = []
    for row in sorted(rows, key=lambda r: r["center"]):
        row["chars"].sort(key=lambda c: c["x0"])
        buf = ""
        for i, ch in enumerate(row["chars"]):
            if i and ch["x0"] - row["chars"][i - 1]["x1"] > 9:
                buf += " "
            buf += fix_symbols(ch["text"])
        parts.append(re.sub(r"\s{2,}", " ", buf).strip())
    return " / ".join(p for p in parts if p)


def _guess_header_rows(rows: list[list[str]]) -> int:
    """表头行数：首行之后，第一列连续为空的行也属于表头（合并表头）。"""
    if len(rows) <= 1:
        return 1
    n = 1
    for row in rows[1:]:
        if not (row and row[0].strip()):
            n += 1
        else:
            break
    return min(n, len(rows) - 1) if len(rows) > 1 else 1


def _fill_down(rows: list[list[str]], header_rows: int) -> bool:
    """
    纵向合并单元格向下填充——表 5.1.9 的"标准型（B型）"跨两行，
    不填充则第二行的"可调托撑 100"无从归属。只填数据行，不碰表头。
    """
    changed = False
    for r in range(header_rows, len(rows)):
        for c in range(len(rows[r])):
            if not rows[r][c].strip() and r > 0 and c < len(rows[r - 1]) and rows[r - 1][c].strip():
                rows[r][c] = rows[r - 1][c]
                changed = True
    return changed


def to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """（表格渲染见下方；这里先定义 _union_box 供屏蔽区计算使用）"""
    rows = [[("" if c is None else str(c)).replace("|", "/").replace(" / ", "<br>")
             for c in r] for r in rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _guess_units(*texts: str) -> "str | None":
    for t in texts:
        for m in re.finditer(r"[（(]([^（）()]{1,20})[）)]", t or ""):
            value = m.group(1).strip()
            if re.search(r"(kN|mm|m2|m3|N/|MPa|kPa|kg)", value):
                return value
    return None


def _union_box(boxes: Sequence[Sequence[float]]) -> list[float]:
    boxes = [b for b in boxes if b]
    return [round(min(b[0] for b in boxes), 2), round(min(b[1] for b in boxes), 2),
            round(max(b[2] for b in boxes), 2), round(max(b[3] for b in boxes), 2)]


def flat_header(rows: Sequence[Sequence[str]], header_rows: int) -> list[str]:
    """
    把多层/合并表头压成单层列名，供检索与 LLM 使用。

    两步：
      1. 横向传播——表头行里空单元格继承左侧最近的非空值（"连墙件布置"横跨两列）
      2. 纵向拼接——同列各表头行用 "-" 连接（"连墙件布置-2步3跨"）

    例（表 5.4.1）：
        类别 | 连墙件布置 |            →  类别 | 连墙件布置-2步3跨 | 连墙件布置-3步3跨
             | 2步3跨     | 3步3跨
    """
    if not rows or header_rows <= 0:
        return []
    n_cols = max(len(r) for r in rows)
    grid = [[_clean_header_cell(c) for c in list(r) + [""] * (n_cols - len(r))]
            for r in rows[:header_rows]]
    for row in grid:                       # 横向传播
        last = ""
        for c in range(n_cols):
            cell = (row[c] or "").strip()
            if cell:
                last = cell
            else:
                row[c] = last
    flat = []
    for c in range(n_cols):
        parts = [grid[r][c].strip() for r in range(len(grid)) if grid[r][c].strip()]
        seen, uniq = set(), []
        for p in parts:                    # 去掉纵向重复（"类别/类别"→"类别"）
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        flat.append("-".join(uniq))
    return flat


def _clean_header_cell(cell: str) -> str:
    """
    表头单元格清洗：多行合并的 " / " 直接相连（"承载力设计值 / （kN）"→"承载力设计值（kN）"），
    中文字间空格去掉（"构 件"→"构件"）。列名会进入检索键，必须干净。
    """
    cell = (cell or "").replace(" / ", "")
    cell = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cell)
    return re.sub(r"\s{2,}", " ", cell).strip()


def to_markdown_flat(rows: Sequence[Sequence[str]], header_rows: int,
                     flat: Sequence[str]) -> str:
    """扁平表头版 Markdown：单行表头 + 数据行，喂 LLM 用这个，别用带合并表头的原表。"""
    if not rows or not flat:
        return ""
    body = [list(r) + [""] * (len(flat) - len(r)) for r in rows[header_rows:]]
    out = ["| " + " | ".join(flat) + " |",
           "| " + " | ".join(["---"] * len(flat)) + " |"]
    for r in body:
        if any((c or "").strip() for c in r):
            out.append("| " + " | ".join((c or "").replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def real_tables(page, lines: list[dict]) -> list[dict]:
    """
    抽表格并过滤假阳性（矢量图被 find_tables 误判成表格时，区域内没有文字单元格）。
    同时还原表标题、表注、表头行数、纵向合并填充。
    """
    found: list[dict] = []
    for tb in page.find_tables():
        cells = [[cell_text(page, c) for c in row.cells] for row in tb.rows]
        filled = [c for row in cells for c in row if c]
        n_chars = len([c for c in page.chars
                       if c["x0"] >= tb.bbox[0] and c["x1"] <= tb.bbox[2]
                       and c["top"] >= tb.bbox[1] and c["bottom"] <= tb.bbox[3]])
        if len(filled) < 2 or n_chars < 6:
            found.append({"bbox": [round(v, 2) for v in tb.bbox], "is_real_table": False,
                          "n_chars": n_chars, "cells": cells})
            continue
        bbox = [round(v, 2) for v in tb.bbox]
        titles = [l for l in lines if TABLE_TITLE_RE.match(l["text"])
                  and bbox[1] - 50 <= l["top"] <= bbox[1] + 10]
        title_line = max(titles, key=lambda l: l["top"]) if titles else None
        title, table_num = "", None
        if title_line is not None:
            m = TABLE_TITLE_RE.match(title_line["text"])
            table_num = re.sub(r"\s+", "", m.group(1))
            title = f"表{table_num} {m.group(2).strip()}".strip()
        notes, started = [], False
        note_lines: list[dict] = []
        for l in lines:
            if l["top"] <= bbox[3]:
                continue
            if l["top"] > bbox[3] + 130:
                break
            if re.match(r"^注[：:]", l["text"]):
                started = True
                notes.append(l["text"])
                note_lines.append(l)
                continue
            if started:
                if (CLAUSE_RE.match(l["text"]) or SECTION_RE.match(l["text"])
                        or TABLE_TITLE_RE.match(l["text"]) or FIGURE_TITLE_RE.match(l["text"])):
                    break
                notes.append(l["text"])
                note_lines.append(l)
        header_rows = _guess_header_rows(cells)
        fill_down = _fill_down(cells, header_rows)
        title_text = title_line["text"] if title_line is not None else ""
        found.append({
            "bbox": bbox, "is_real_table": True, "cells": cells,
            "table_id": table_num, "title": title, "notes": notes,
            "header_rows": header_rows, "fill_down": fill_down,
            "units": _guess_units(title_text, *cells[0] if cells else []),
            "n_chars": n_chars, "n_rows": len(cells),
            "n_cols": max((len(r) for r in cells), default=0),
            # 屏蔽区：表体外还要盖住表标题与表注，否则它们会混进条文正文
            "block_bbox": _union_box([bbox] + ([line_bbox([title_line])] if title_line else [])
                                     + [line_bbox([l]) for l in note_lines]),
        })
    return found


def merge_cross_page_tables(prev_page_tables: list[dict], cur_page_tables: list[dict],
                            prev_page_no: int, cur_page_no: int) -> list[dict]:
    """上一页页尾的表头 + 本页页首的表体，合成一张逻辑表（表头列数补齐）。"""
    if not prev_page_tables or not cur_page_tables:
        return cur_page_tables
    last = [t for t in prev_page_tables if t.get("is_real_table")]
    first = [t for t in cur_page_tables if t.get("is_real_table")]
    if not last or not first:
        return cur_page_tables
    a, b = last[-1], first[0]
    same_shape = abs(a["bbox"][0] - b["bbox"][0]) < 6 and abs(a["bbox"][2] - b["bbox"][2]) < 6
    if not (same_shape and a["bbox"][3] > 720 and b["bbox"][1] < 120):
        return cur_page_tables
    width = max(len(r) for r in b["cells"])
    header = [list(r) + [""] * (width - len(r)) for r in a["cells"] if any(x.strip() for x in r)]
    b["cells"] = header + b["cells"]
    b["n_rows"] = len(b["cells"])
    b["header_rows"] = len(header)
    b["cross_page"] = [prev_page_no, cur_page_no]
    b["title"] = a.get("title") or b.get("title")
    b["table_id"] = a.get("table_id") or b.get("table_id")
    b["notes"] = b.get("notes") or a.get("notes")
    a["merged_into_next"] = True
    return cur_page_tables


# -------------------------------------------------------------------- 图区定位

def graphic_clusters(page, gap: float = 14.0, min_w: float = 60.0,
                     min_h: float = 60.0) -> list[list[float]]:
    """
    把矢量线条/曲线按邻近关系聚成图区。
    本规范的图几乎全是矢量绘制，位图接口取不到，只能靠图元聚类。
    """
    boxes = [[d["x0"], d["top"], d["x1"], d["bottom"]]
             for d in list(page.curves) + list(page.lines)]
    clusters: list[list[float]] = []
    for box in sorted(boxes, key=lambda b: b[0]):
        for cl in clusters:
            if (box[0] <= cl[2] + gap and box[2] >= cl[0] - gap
                    and box[1] <= cl[3] + gap and box[3] >= cl[1] - gap):
                cl[0] = min(cl[0], box[0]); cl[1] = min(cl[1], box[1])
                cl[2] = max(cl[2], box[2]); cl[3] = max(cl[3], box[3])
                break
        else:
            clusters.append(list(box))
    return [[round(v, 2) for v in c] for c in clusters
            if (c[2] - c[0]) > min_w and (c[3] - c[1]) > min_h]


def group_figures(clusters: list[list[float]], lines: list[dict],
                  table_boxes: Sequence[Sequence[float]]) -> list[dict]:
    """
    按图题把多个子图合并成一张图，并把图题、图注、子图标签一并归属。
    图题在图的下方；一张图可能由左右两个子图组成（图元聚类会拆成两块）。
    """
    caps = sorted([l for l in lines if FIGURE_TITLE_RE.match(l["text"])],
                  key=lambda l: l["top"])
    free = [c for c in clusters
            if not any(abs(c[0] - t[0]) < 3 and abs(c[1] - t[1]) < 3 for t in table_boxes)]
    figures: list[dict] = []
    used: set[int] = set()
    for cap in caps:
        m = FIGURE_TITLE_RE.match(cap["text"])
        members = [(i, c) for i, c in enumerate(free)
                   if i not in used and c[3] <= cap["top"] + 3 and cap["top"] - c[3] <= 140]
        if not members:
            continue
        for i, _ in members:
            used.add(i)
        boxes = [c for _, c in members]
        top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
        legend = [l for l in lines if LEGEND_RE.match(l["text"])
                  and cap["top"] < l["top"] < cap["top"] + 30]
        if legend:
            bottom = max(bottom, max(l["bottom"] for l in legend))
        sub_labels = [l["text"] for l in lines
                      if re.search(r"[（(]?\s*[ab]\s*[）)]", l["text"])
                      and top - 40 < l["top"] < cap["top"]]
        outer = _union_box(boxes + [line_bbox([cap])] + [line_bbox([l]) for l in legend])
        figures.append({
            "figure_id": re.sub(r"\s+", "", m.group(1)),
            "title": f"图{m.group(1)} {m.group(2).strip()}".strip(),
            "legend": [norm_dash(l["text"]) for l in legend],
            "sub_labels": sub_labels,
            "bbox": outer,   # 节点坐标含图题与图注；不盖住它们，图题会被当成正文
            "art_bbox": [round(min(b[0] for b in boxes), 2), round(top, 2),
                         round(max(b[2] for b in boxes), 2), round(bottom, 2)],
            "top": top,
        })
    return figures


def crop_png(page, bbox: Sequence[float], path: "str | Path",
             resolution: int = 200, pad: float = 4.0) -> str:
    """按坐标裁图存 PNG（pdfplumber 的 PageImage 没有 crop，要裁页面再渲染）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    box = (max(0.0, bbox[0] - pad), max(0.0, bbox[1] - pad),
           min(page.width, bbox[2] + pad), min(page.height, bbox[3] + pad))
    page.crop(box).to_image(resolution=resolution).save(path)
    return str(path)


def formula_regions(lines: Sequence[dict], exclude=()) -> list[dict]:
    """
    公式区识别。判据（三条，缺一不可）：
      1. 不含中文——含中文的是叙述行（"不组合风荷载时："）
      2. 行宽 < 300pt——正文行宽约 450pt，公式居中且窄
      3. 公式编号出现在行尾——"…按本标准式（5.3.3-1）计算。" 是引用，不是公式
    """
    def is_formulaish(l: dict) -> bool:
        text = l["text"]
        if any(pat(text) for pat in exclude):
            return False
        if EQNUM_TAIL_RE.search(text):
            return True
        if re.search(r"[\u4e00-\u9fff]", text):
            return False
        return (l["x1"] - l["x0"]) < 300

    regions, buf = [], []
    for line in lines:
        if is_formulaish(line):
            buf.append(line)
            continue
        if buf:
            regions.append(buf)
            buf = []
    if buf:
        regions.append(buf)

    out = []
    for group in regions:
        # 一组连续窄行里可能含多个编号公式（如 5.3.1-1 与 5.3.1-2），必须在编号处切开
        cur: list[dict] = []
        for line in group:
            cur.append(line)
            m = EQNUM_TAIL_RE.search(line["text"])
            if m:
                eid = EQNUM_RE.search(line["text"]).group(1)
                out.append({"equation_id": eid, "lines": list(cur),
                            "bbox": line_bbox(cur),
                            "linear_text": "\n".join(l["text"] for l in cur)})
                cur = []
        if cur and out:            # 编号之后的分式下沿，并回前一个公式
            out[-1]["lines"].extend(cur)
            out[-1]["bbox"] = line_bbox(out[-1]["lines"])
            out[-1]["linear_text"] = "\n".join(l["text"] for l in out[-1]["lines"])
    return out
