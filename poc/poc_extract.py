# -*- coding: utf-8 -*-
"""
JGJ/T 231-2021 切分 POC（第 2 版）

测试页：第 10、11、22 页
  第 10-11 页：表 4.3.1 跨页（表头在 p10、表体在 p11）+ 4.4.3 公式 + 表 4.4.4 公式型单元格
  第 22 页：两张矢量图（每张含 a/b 两个子图）+ 6.2.3/6.2.4 排印断行陷阱

第 1 版发现的 5 个问题，本版逐个处理：
  1. 跨页表格 → 合并（恢复表头）
  2. 表格单元格内下标被空格拆开（G1+G2 → G 1 +G 2）→ 按字符坐标重建单元格文本
  3. 条文号误判："（图" 断行导致下一行行首的 "6.2.4）" 被当成新条文 → 断行修复 + 负向断言
  4. 一张图被左右两个子图拆成两个图区 → 按图题分组合并
  5. 图题/图注混进正文、页码混进条文 → 归属规则 + 页脚过滤
"""
import json
import re
from pathlib import Path

import pdfplumber

PDF = Path(__file__).resolve().parent.parent / "盘扣规范" / "jgj 231-2021.pdf"
PAGES = [9, 10, 11, 22]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

PUA_MAP = {
    0xF03D: "=", 0xF067: "γ", 0xF0A3: "≤", 0xF02B: "+",
    0xF0E5: "∑", 0xF0B4: "×", 0xF06A: "φ", 0xF06D: "μ",
    0xF0D7: "·", 0xF062: "β", 0xF068: "η", 0xF06C: "λ",
    0xF0B3: "≥", 0xF053: "∑",
}

CLAUSE_RE = re.compile(r"^(\d+\.\d+\.\d+)\s*(?![）)、。；：，])")
SECTION_RE = re.compile(r"^(\d+\.\d+)\s+([^\d].*)$")
FIGURE_RE = re.compile(r"^图\s?(\d+(?:\.\d+)*(?:-\d+)?)\s+(.*)$")
TABLE_RE = re.compile(r"^表\s?([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)\s+(.*)$")
EQNUM_RE = re.compile(r"[（(]\s?[A-Z]?\.?\d+(?:\.\d+)+(?:-\d+)?\s?[）)]")
LEGEND_RE = re.compile(r"^\d+\s?[—\-－]")


def fix_pua(text: str) -> str:
    return "".join(PUA_MAP.get(ord(ch), ch) for ch in text)


# ---------------------------------------------------------------- 文本行重建
def page_lines(page) -> list[dict]:
    chars = [dict(c, text=fix_pua(c["text"])) for c in page.chars]
    lines, current = [], []
    for ch in sorted(chars, key=lambda c: ((c["top"] + c["bottom"]) / 2, c["x0"])):
        center = (ch["top"] + ch["bottom"]) / 2
        if current:
            ref = sum((c["top"] + c["bottom"]) / 2 for c in current) / len(current)
            if abs(center - ref) > 4.0:
                lines.append(current)
                current = []
        current.append(ch)
    if current:
        lines.append(current)

    out = []
    for group in lines:
        group.sort(key=lambda c: c["x0"])
        text = "".join(c["text"] for c in group).strip()
        if not text:
            continue
        out.append({"text": text,
                    "x0": round(min(c["x0"] for c in group), 1),
                    "x1": round(max(c["x1"] for c in group), 1),
                    "top": round(min(c["top"] for c in group), 1),
                    "bottom": round(max(c["bottom"] for c in group), 1)})
    out.sort(key=lambda l: (l["top"], l["x0"]))
    return out


def drop_running_heads(lines, page) -> list[dict]:
    """页脚页码：纯数字且落在页面底部 80pt 内。"""
    return [l for l in lines
            if not (re.fullmatch(r"\d{1,3}", l["text"]) and l["top"] > page.height - 80)]


def repair_broken_lines(lines) -> list[dict]:
    """排印陷阱：'…悬臂长度（图' 换行后接着 '6.2.4）不应超过650mm…'。合并回一行。"""
    out = []
    for line in lines:
        if out and re.search(r"[（(](图|表)$", out[-1]["text"]) and re.match(
                r"^[A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?\s?[）)]", line["text"]):
            out[-1]["text"] += line["text"]
            out[-1]["x1"] = max(out[-1]["x1"], line["x1"])
            out[-1]["bottom"] = max(out[-1]["bottom"], line["bottom"])
            continue
        out.append(line)
    return out


# ------------------------------------------------------------------ 表格还原
def cell_text(page, bbox) -> str:
    """按字符坐标重建单元格：下标直接贴回主字符，字间大空隙才留空格。"""
    if not bbox:
        return ""
    x0, top, x1, bottom = bbox
    chars = [c for c in page.chars
             if c["x0"] >= x0 - 0.6 and c["x1"] <= x1 + 0.6
             and c["top"] >= top - 0.6 and c["bottom"] <= bottom + 0.6]
    if not chars:
        return ""
    rows = []
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
            buf += ch["text"]
        parts.append(buf.strip())
    return " / ".join(p for p in parts if p)


def real_tables(page, lines) -> list[dict]:
    found = []
    for tb in page.find_tables():
        # pdfplumber 的 Table.cells 是扁平列表，按行取要用 tb.rows[i].cells
        cells = [[cell_text(page, c) for c in row.cells] for row in tb.rows]
        filled = [c for row in cells for c in row if c]
        n_chars = len([c for c in page.chars
                       if c["x0"] >= tb.bbox[0] and c["x1"] <= tb.bbox[2]
                       and c["top"] >= tb.bbox[1] and c["bottom"] <= tb.bbox[3]])
        bbox = [round(v, 1) for v in tb.bbox]
        # 表标题：表格上方 45pt 内最靠近的那一条
        titles = [l for l in lines if TABLE_RE.match(l["text"])
                  and bbox[1] - 45 <= l["top"] <= bbox[1] + 8]
        title = max(titles, key=lambda l: l["top"])["text"] if titles else ""
        # 表注：从“注：…”开始，到下一个条号/节标题/表标题之前的全部行（表注常有多行）
        notes, started = [], False
        for l in lines:
            if l["top"] <= bbox[3]:
                continue
            if l["top"] > bbox[3] + 130:
                break
            if re.match(r"^注[：:]", l["text"]):
                started = True
                notes.append(l["text"])
                continue
            if started:
                if (CLAUSE_RE.match(l["text"]) or SECTION_RE.match(l["text"])
                        or TABLE_RE.match(l["text"]) or FIGURE_RE.match(l["text"])):
                    break
                notes.append(l["text"])
        found.append({"bbox": [round(v, 1) for v in tb.bbox],
                      "n_rows": len(tb.rows), "n_cols": len(tb.columns),
                      "cells": cells, "n_chars": n_chars, "title": title, "notes": notes,
                      # 门槛放宽到 2 个单元格：跨页的表格头可能只剩 1 行 2 格
                      "is_real_table": len(filled) >= 2 and n_chars >= 6})
    return found


def merge_cross_page_tables(prev, cur) -> list[dict]:
    """上一页页尾的表头 + 本页页首的表体，合成一张逻辑表。"""
    if not prev or not cur:
        return cur
    last, first = prev[-1], cur[0]
    same_shape = abs(last["bbox"][0] - first["bbox"][0]) < 5 and abs(last["bbox"][2] - first["bbox"][2]) < 5
    if (last["is_real_table"] and first["is_real_table"] and same_shape
            and last["bbox"][3] > 720 and first["bbox"][1] < 120):
        header_rows = [r for r in last["cells"] if any(r)]
        merged = dict(first)
        width = first["n_cols"]
        # 页尾的表头往往只剩 2 列（如“验算项目 | 荷载分项系数”），补齐到表体列数
        merged["cells"] = [list(r) + [""] * (width - len(r)) for r in header_rows] + first["cells"]
        merged["cross_page"] = [last["page"], first["page"]]
        merged["title"] = last.get("title") or first.get("title")
        merged["notes"] = first.get("notes") or last.get("notes")
        last["merged_into_next"] = True  # 页尾表头碎片已并入下一页表体，不再单独输出
        cur[0] = merged
    return cur


# -------------------------------------------------------------------- 图区识别
def graphic_clusters(page, gap: float = 14.0) -> list[list[float]]:
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
    return [c for c in clusters if (c[2] - c[0]) > 60 and (c[3] - c[1]) > 60]


def group_figures(clusters, lines, table_boxes) -> list[dict]:
    """按图题把左右子图合并成一张图，并把图题、图注、子图标签一并归属。"""
    caps = sorted([l for l in lines if FIGURE_RE.match(l["text"])], key=lambda l: l["top"])
    free = [c for c in clusters
            if not any(abs(c[0] - t[0]) < 3 and abs(c[1] - t[1]) < 3 for t in table_boxes)]
    figures, used = [], set()
    for cap in caps:
        members = [(i, c) for i, c in enumerate(free)
                   if i not in used and c[3] <= cap["top"] + 2 and cap["top"] - c[3] <= 130]
        if not members:
            continue
        for i, _ in members:
            used.add(i)
        boxes = [c for _, c in members]
        top = min(b[1] for b in boxes)
        bottom = max(b[3] for b in boxes)
        legend = [l for l in lines if LEGEND_RE.match(l["text"])
                  and cap["top"] < l["top"] < cap["top"] + 30]
        if legend:
            bottom = max(bottom, max(l["bottom"] for l in legend))
        sublabels = [l["text"] for l in lines
                     if re.search(r"[（(]?[ab][）)]", l["text"]) and top - 30 < l["top"] < cap["top"]]
        figures.append({
            "figure_id": FIGURE_RE.match(cap["text"]).group(1),
            "caption": cap["text"],
            "legend": [l["text"] for l in legend],
            "sub_labels": sublabels,
            "bbox": [round(min(b[0] for b in boxes), 1), round(top, 1),
                     round(max(b[2] for b in boxes), 1), round(bottom, 1)],
        })
    return figures


# ------------------------------------------------------------------- 条文切分
def in_box(line, box, pad=2.0) -> bool:
    return (line["x0"] >= box[0] - pad and line["x1"] <= box[2] + pad
            and line["top"] >= box[1] - pad and line["bottom"] <= box[3] + pad)


def segment(lines, blocked) -> list[dict]:
    chunks, current = [], None
    for line in lines:
        if any(in_box(line, b) for b in blocked):
            continue
        m, sec = CLAUSE_RE.match(line["text"]), SECTION_RE.match(line["text"])
        if m:
            if current:
                chunks.append(current)
            current = {"type": "clause", "clause_id": m.group(1), "text": line["text"], "lines": [line]}
        elif sec and len(line["text"]) < 30:
            if current:
                chunks.append(current)
            current = {"type": "section", "clause_id": sec.group(1), "text": line["text"], "lines": [line]}
        elif current:
            current["text"] += "\n" + line["text"]
            current["lines"].append(line)
        else:
            current = {"type": "preamble", "clause_id": None, "text": line["text"], "lines": [line]}
    if current:
        chunks.append(current)
    return chunks


def merge_same_id(segs) -> list[dict]:
    """条文被表格/插图打断时会重复出现同一个条号（如 4.2.5 被表 4.2.5 隔开），按条号并回去。"""
    merged = []
    for s in segs:
        if merged and s["clause_id"] and merged[-1]["clause_id"] == s["clause_id"]:
            merged[-1]["text"] += "\n" + s["text"]
            merged[-1]["lines"] += s["lines"]
        else:
            merged.append(s)
    return merged


def extract_formula(chunk, page_no) -> dict | None:
    """'…符合下式要求：' 与 '式中：' 之间就是公式区：裁图 + 抽变量表。"""
    ls = chunk["lines"]
    idx_where = next((i for i, l in enumerate(ls) if l["text"].strip().startswith("式中")), None)
    if idx_where is None:
        return None
    # 取“式中”之前最后一个以“：”结尾的行（条文里可能有“下列规定：”在前面）
    idx_colon = max((i for i, l in enumerate(ls[:idx_where])
                     if l["text"].strip().endswith("：")), default=None)
    if idx_colon is None or idx_where <= idx_colon + 1:
        return None
    fls = ls[idx_colon + 1:idx_where]
    if not fls:
        return None
    eq_no = next((EQNUM_RE.search(l["text"]).group(0) for l in fls if EQNUM_RE.search(l["text"])), None)
    var_text = "\n".join(l["text"] for l in ls[idx_where:])
    variables = [{"symbol": m.group(1), "meaning": m.group(2).strip()}
                 for m in re.finditer(r"([A-Za-z\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff]*)\s*——\s*([^；;\n]+)", var_text)]
    return {"type": "formula", "page": page_no, "equation_id": eq_no,
            "linear": "\n".join(l["text"] for l in fls),
            "variables": variables,
            "bbox": [round(min(l["x0"] for l in fls), 1), round(min(l["top"] for l in fls), 1),
                     round(max(l["x1"] for l in fls), 1), round(max(l["bottom"] for l in fls), 1)]}


def finalize(chunk, page_no) -> dict:
    ls = chunk.pop("lines")
    chunk["page"] = page_no
    chunk["bbox"] = [round(min(l["x0"] for l in ls), 1), round(min(l["top"] for l in ls), 1),
                     round(max(l["x1"] for l in ls), 1), round(max(l["bottom"] for l in ls), 1)]
    t = chunk["text"]
    chunk["refs_tables"] = ["表" + x for x in re.findall(r"表\s?([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)", t)]
    chunk["refs_figures"] = ["图" + x for x in re.findall(r"图\s?(\d+(?:\.\d+)*(?:-\d+)?)", t)]
    chunk["refs_appendix"] = re.findall(r"附录\s?([A-Z])", t)
    return chunk


def to_md(cells) -> str:
    rows = [[(c or "").replace("|", "/").replace(" / ", "<br>") for c in row] for row in cells]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def main():
    report = ["# JGJ/T 231-2021 切分 POC 验收报告（第 2 版）", "",
              f"测试页：第 {'、'.join(map(str, PAGES))} 页", ""]
    per_page, prev_tables = [], []

    # --- 第一遍：逐页解析，不做最终定稿（后处理需要跨页信息） ---
    with pdfplumber.open(PDF) as pdf:
        for pno in PAGES:
            page = pdf.pages[pno - 1]
            raw_lines = page_lines(page)
            lines = repair_broken_lines(drop_running_heads(raw_lines, page))
            tables = real_tables(page, lines)
            for t in tables:
                t["page"] = pno
            merge_cross_page_tables(prev_tables, tables)
            prev_tables = tables

            real = [t for t in tables if t["is_real_table"] and not t.get("merged_into_next")]
            tbl_boxes = [t["bbox"] for t in real]
            note_texts = {n for t in real for n in t.get("notes", [])}
            clusters = graphic_clusters(page)
            figures = group_figures(clusters, lines, tbl_boxes)
            blocked = tbl_boxes + [f["bbox"] for f in figures]
            segs = segment([l for l in lines if l["text"] not in note_texts], blocked)

            per_page.append({"pno": pno, "page": page, "raw": raw_lines, "lines": lines,
                             "tables": tables, "real": real, "figures": figures,
                             "clusters": len(clusters), "segs": segs})

    # --- 第二遍：跨页续接 + 同号合并（4.2.3 的“式中”跑到下一页、4.2.5 被表格打断） ---
    flat = [(pp["pno"], s) for pp in per_page for s in pp["segs"]]
    stitched = []
    log = []
    for pno, seg in flat:
        if stitched:
            prev_pno, prev = stitched[-1]
            if seg["type"] == "preamble":
                prev["text"] += "\n" + seg["text"]
                prev["lines"] += seg["lines"]
                log.append(f"- 跨页续接：第 {pno} 页开头 {len(seg['lines'])} 行并回上一条文 {prev['clause_id']}")
                continue
            if seg["clause_id"] and prev["clause_id"] == seg["clause_id"]:
                prev["text"] += "\n" + seg["text"]
                prev["lines"] += seg["lines"]
                log.append(f"- 同号合并：第 {pno} 页的 {seg['clause_id']} 被表格/插图打断，已并回")
                continue
        stitched.append((pno, seg))

    # --- 第三遍：定稿 + 产出 ---
    chunks = []
    by_page = {}
    for start_page, seg in stitched:
        formula = extract_formula(seg, start_page)
        ch = finalize(seg, start_page)
        chunks.append(ch)
        by_page.setdefault(start_page, []).append(ch)
        if formula:
            chunks.append(formula)
            by_page[start_page].append(formula)

    report += ["## 全局后处理", ""] + (log or ["（无）"]) + [""]
    for pp in per_page:
        pno, page = pp["pno"], pp["page"]
        report += [f"## 第 {pno} 页", "",
                   f"- 文本行 {len(pp['raw'])} → 修复后 {len(pp['lines'])}",
                   f"- 表格候选 {len(pp['tables'])} 个（真表格 {len(pp['real'])} 个）",
                   f"- 矢量图元簇 {pp['clusters']} 个 → 按图题合并为 {len(pp['figures'])} 张图", ""]
        for t in pp["tables"]:
            tag = "✅ 真表格" if t["is_real_table"] else "❌ 误判的矢量图（无文字单元格）"
            extra = f"（跨页合并自 p{t['cross_page']}）" if t.get("cross_page") else ""
            report.append(f"  - bbox={t['bbox']} {t['n_rows']}行×{t['n_cols']}列 "
                          f"有字单元格={sum(1 for r in t['cells'] for c in r if c)} → {tag}{extra}")
        report.append("")

        report += ["### 条文切分", ""]
        for ch in by_page.get(pno, []):
            if ch["type"] == "formula":
                report += [f"  ↳ 公式 {ch['equation_id']} bbox={ch['bbox']} "
                           f"变量 {len(ch['variables'])} 个", "", "```", ch["linear"], "```", ""]
                continue
            report += [f"**[{ch['clause_id'] or ch['type']}]** bbox={ch['bbox']}", "",
                       "```", ch["text"], "```", ""]

        report += ["### 表格结构化", ""]
        final_tables = [t for t in pp["real"] if not t.get("merged_into_next")]
        for i, t in enumerate(final_tables, 1):
            tm = TABLE_RE.match(t.get("title") or "")
            report += [f"**{t.get('title') or f'表格{i}'}**"
                       f"{'（跨页合并）' if t.get('cross_page') else ''}", "", to_md(t["cells"]), ""]
            if t.get("notes"):
                report += ["表注："] + [f"  - {n}" for n in t["notes"]] + [""]
            chunks.append({"page": pno, "type": "table", "table_id": tm.group(1) if tm else None,
                           "title": t.get("title", ""), "bbox": t["bbox"],
                           "cross_page": t.get("cross_page"), "notes": t.get("notes", []),
                           "markdown": to_md(t["cells"]), "rows": t["cells"]})

        if pp["figures"]:
            report += ["### 图区", ""]
        page.to_image(resolution=150).save(OUT / f"page{pno}.png")
        for i, f in enumerate(pp["figures"], 1):
            f["image"] = f"p{pno}_figure{i}.png"
            report += [f"- 图{f['figure_id']} bbox={f['bbox']}", f"  子图标签：{f['sub_labels']}",
                       f"  图题：{f['caption']}", f"  图注：{f['legend']}", ""]
            chunks.append({"page": pno, "type": "figure", **f})
            pad = 4
            page.crop((max(0, f["bbox"][0] - pad), max(0, f["bbox"][1] - pad),
                       min(page.width, f["bbox"][2] + pad),
                       min(page.height, f["bbox"][3] + pad))
                      ).to_image(resolution=200).save(OUT / f["image"])
        for i, t in enumerate(final_tables, 1):
            page.crop(t["bbox"]).to_image(resolution=200).save(OUT / f"p{pno}_table{i}.png")

    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    kinds = {}
    for c in chunks:
        kinds[c["type"]] = kinds.get(c["type"], 0) + 1
    print("块统计:", kinds)
    print("输出目录:", OUT)


if __name__ == "__main__":
    main()
