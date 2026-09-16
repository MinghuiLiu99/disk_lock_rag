# -*- coding: utf-8 -*-
"""
主解析器：PDF → 结构化节点（字段规范 v1.1）。

流程：
  逐页 → 文本行重建 → 表格还原 → 图区定位 → 屏蔽已归属区域
      → 条文/节/章切分 → 跨页续接 → 同号合并
      → 组装节点（条 / 款 / 表 / 图 / 公式）→ 建立关系 → 校验

所有阈值都来自对 JGJ/T 231-2021 的实测，改之前先看 layout.py 的注释。
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Sequence

import pdfplumber

from . import layout as L
from .schema import NodeBuilder, node_id_of, link_structure, extract_refs


class DocumentParser:
    """
    参数
      pdf_path       测试件或完整版 PDF
      standard_id    标准简称，如 JGJ231（用于 node_id 前缀）
      standard_code  完整标准号，如 "JGJ/T 231-2021"（答案引用展示）
      doc_id         文档标识，如 jgj231_2021_ch5_test
      version        版本年，如 "2021"
      page_offset    文件内第 1 页对应的原书页码偏移（page 字段记原书页码）
      figure_dir     裁切图的输出目录（相对路径写进 body.image_path）
    """

    def __init__(self, pdf_path, standard_id: str, standard_code: str, doc_id: str,
                 version: str, *, page_offset: int = 0, figure_dir: str | Path | None = None,
                 y_tol: float = 6.0, verbose: bool = False):
        self.pdf_path = Path(pdf_path)
        self.builder = NodeBuilder(standard_id, standard_code, doc_id, version)
        self.standard_id = standard_id
        self.page_offset = int(page_offset)
        self.figure_dir = Path(figure_dir) if figure_dir else None
        self.y_tol = y_tol
        self.verbose = verbose

    # ------------------------------------------------------------- 逐页采集
    def _collect(self, pdf) -> list[dict]:
        pages = []
        prev_tables: list[dict] = []
        prev_page_no = None
        for idx, page in enumerate(pdf.pages, start=1):
            book_page = idx + self.page_offset
            lines = L.repair_broken_lines(L.drop_running_heads(
                L.page_lines(page, y_tol=self.y_tol), page))
            tables = L.real_tables(page, lines)
            for t in tables:
                t["page"] = book_page
            if prev_tables:
                L.merge_cross_page_tables(prev_tables, tables, prev_page_no, book_page)
            prev_tables, prev_page_no = tables, book_page

            real = [t for t in tables if t.get("is_real_table") and not t.get("merged_into_next")]
            table_boxes = [t["bbox"] for t in real]
            figures = L.group_figures(L.graphic_clusters(page), lines, table_boxes)

            skip: set[str] = set()
            for t in real:
                if t.get("title"):
                    skip.add(L.norm_text(t["title"]))
                skip.update(L.norm_text(n) for n in (t.get("notes") or []))
            for f in figures:
                skip.add(L.norm_text(f["title"]))
                skip.update(L.norm_text(x) for x in (f.get("legend") or []))
                skip.update(L.norm_text(x) for x in (f.get("sub_labels") or []))

            # 屏蔽区要盖住表标题、表注、图题、图注：否则它们会被当成正文吸进条文
            blocked = [t.get("block_bbox") or t["bbox"] for t in real] + [f["bbox"] for f in figures]

            pages.append({"index": idx, "book_page": book_page, "page": page,
                          "lines": lines, "tables": real,
                          "figures": figures, "skip": skip,
                          "blocked": blocked,
                          "body_size": _body_size(page)})
        return pages

    # ------------------------------------------------------------- 条文切分
    def _segments(self, pg: dict) -> list[dict]:
        """把一页的文本行切成 章/节/条 片段；表格、图、图题、表注所在的区域先屏蔽。"""
        segs: list[dict] = []
        cur: dict | None = None
        for line in pg["lines"]:
            blocked = (any(L.in_box(line, b) for b in pg["blocked"])
                       or L.norm_text(line["text"]) in pg["skip"])
            if blocked:
                if cur:
                    segs.append(cur); cur = None
                continue
            kind = _heading_kind(line, pg["body_size"])
            if kind is None and L.CLAUSE_RE.match(line["text"]):
                kind = "clause"
            if kind:
                if cur:
                    segs.append(cur)
                cur = {"kind": kind, "lines": [line]}
            elif cur:
                cur["lines"].append(line)
            else:
                cur = {"kind": "continuation", "lines": [line]}
        if cur:
            segs.append(cur)
        for s in segs:
            s["text"] = "\n".join(l["text"] for l in s["lines"])
            s["bbox"] = L.line_bbox(s["lines"])
            s["top"] = min(l["top"] for l in s["lines"])
        return segs

    # ------------------------------------------------------------- 元素排序
    def _elements(self, pages: list[dict]) -> list[dict]:
        els: list[dict] = []
        for pg in pages:
            for s in self._segments(pg):
                els.append({"kind": "seg", "page": pg, "top": s["top"], "seg": s})
            for t in pg["tables"]:
                els.append({"kind": "table", "page": pg, "top": t["bbox"][1], "table": t})
            for f in pg["figures"]:
                els.append({"kind": "figure", "page": pg, "top": f["top"], "figure": f})
        els.sort(key=lambda e: (e["page"]["index"], e["top"]))
        return els

    def _crop(self, page, fig: dict) -> "str | None":
        if self.figure_dir is None:
            return None
        name = f"{self.standard_id}_fig{fig['figure_id'].replace('.', '_')}.png"
        path = self.figure_dir / name
        L.crop_png(page, fig["bbox"], path)
        # image_path 统一记为"相对项目根目录"的路径，前端与索引层都能直接解析
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _emit_formulas(self, nodes: list[dict], lines: Sequence[dict], bp: int) -> None:
        """从一组行里抽公式节点。公式必须独立成节点：线性化会丢分式结构，只能截图+变量表。"""
        for fm in L.formula_regions(lines, exclude=(
                lambda t: bool(L.TABLE_TITLE_RE.match(t)),
                lambda t: bool(L.FIGURE_TITLE_RE.match(t)),
                lambda t: bool(L.LEGEND_RE.match(t)))):
            nodes.append(self.builder.formula(
                fm["equation_id"], _variables_from(lines, fm), [bp], [fm["bbox"]],
                linear_text=L.norm_text(fm["linear_text"]),
                image_path=None,
                keywords=[f"式（{fm['equation_id']}）", fm["equation_id"]],
                status="need_review", status_reason="formula_linear_only"))

    def _consume_items(self, nodes: list[dict], lines: Sequence[dict], clause_num: str,
                       bp: int, state: dict, *, require_list: bool, seg_text: str = "") -> None:
        """
        切款/项。跨页时条文的款会落在续接片段里，所以编号必须能跨段连续，
        state 在条文之间传递（next = 下一个期望的款号）。
        """
        if require_list and "下列" not in seg_text and "应包括" not in seg_text:
            return
        body = lines[1:] if require_list else lines
        for line in body:
            text = line["text"].strip()
            m = L.ITEM_RE.match(text)
            if m and int(m.group(1)) == state["next"] and not L.CLAUSE_RE.match(text):
                node = self.builder.item(
                    clause_num, m.group(1), L.norm_text(text), [bp], [L.line_bbox([line])],
                    keywords=[f"{clause_num}-{m.group(1)}", clause_num])
                nodes.append(node)
                state["last"] = node
                state["next"] += 1
            elif state.get("last") is not None:
                _append_line(state["last"], bp, line)

    # --------------------------------------------------------------- 组装
    def _assemble(self, els: list[dict]) -> list[dict]:
        nodes: list[dict] = []
        b = self.builder
        current: dict | None = None          # 当前条文节点（表/图/公式的挂载点）
        item_state: dict = {"clause": None, "next": 1, "last": None}
        for el in els:
            pg, bp = el["page"], el["page"]["book_page"]
            page_obj = pg["page"]

            if el["kind"] == "table":
                t = el["table"]
                header_flat = L.flat_header(t["cells"], t["header_rows"])
                node = b.table(t["table_id"], t["title"], t["cells"], [bp], [t["bbox"]],
                               notes=t["notes"], header_rows=t["header_rows"],
                               fill_down=t["fill_down"], units=t["units"],
                               markdown=L.to_markdown(t["cells"]),
                               header_flat=header_flat,
                               markdown_flat=L.to_markdown_flat(t["cells"], t["header_rows"],
                                                                header_flat),
                               cross_page=t.get("cross_page"),
                               keywords=[f"表{t['table_id']}", t["table_id"]])
                nodes.append(node)
                continue

            if el["kind"] == "figure":
                f = el["figure"]
                node = b.figure(f["figure_id"], f["title"], [bp], [f["bbox"]],
                                legend=f["legend"], sub_labels=f["sub_labels"],
                                image_path=self._crop(page_obj, f),
                                keywords=[f"图{f['figure_id']}", f["figure_id"]])
                nodes.append(node)
                continue

            seg = el["seg"]
            kind = seg["kind"]

            if kind in ("chapter", "section"):
                num, title = _split_heading(seg["lines"][0]["text"])
                if kind == "chapter":
                    b.set_context(chapter=f"{num} {title}", section=None)
                    nodes.append(b.chapter_node(num, title, [bp], [seg["bbox"]]))
                else:
                    b.set_context(section=f"{num} {title}")
                    nodes.append(b.section(num, title, [bp], [seg["bbox"]]))
                current = None
                continue

            if kind == "clause":
                num = L.CLAUSE_RE.match(seg["lines"][0]["text"]).group(1)
                if current is not None and current["num"] == num:
                    # 条文被表格/插图打断后又续上同一个条号
                    _merge_continuation(current, bp, seg["bbox"], seg["text"])
                    current["status"], current["status_reason"] = "need_review", "split_by_table"
                    continue
                text = L.norm_text(seg["text"])
                node = b.clause(num, text, [bp], [seg["bbox"]],
                                keywords=_structural_keywords(num, b.section_title))
                nodes.append(node)
                item_state = {"clause": num, "next": 1, "last": None}
                self._consume_items(nodes, seg["lines"], num, bp, item_state,
                                    require_list=True, seg_text=seg["text"])
                self._emit_formulas(nodes, seg["lines"], bp)
                current = node
                continue

            # 无条号开头 → 上一条的跨页续接
            self._emit_formulas(nodes, seg["lines"], bp)
            if current is not None:
                if item_state.get("clause") == current["num"]:
                    self._consume_items(nodes, seg["lines"], current["num"], bp, item_state,
                                        require_list=False)
                _merge_continuation(current, bp, seg["bbox"], seg["text"])
                same_page = bp == current["pages"][0]
                current["status"] = "need_review"
                current["status_reason"] = "split_by_table" if same_page else "cross_page_clause"
            else:
                nodes.append(b.preamble(L.norm_text(seg["text"]), [bp], [seg["bbox"]]))
        return nodes

    # --------------------------------------------------------------- 入口
    def parse(self, max_pages: "int | None" = None) -> list[dict]:
        with pdfplumber.open(self.pdf_path) as pdf:
            pages = self._collect(pdf)
            pages = pages[:max_pages] if max_pages else pages
            els = self._elements(pages)
            nodes = self._assemble(els)   # 裁图在 _assemble 内完成，此时 pdf 仍打开
        nodes = link_structure(nodes)
        self._refine_formula_variables(nodes)
        return nodes

    def _refine_formula_variables(self, nodes: list[dict]) -> None:
        """
        用条文完整正文重新分配公式变量。两个原因必须做这步：
          1. 一条文里多个公式常共用同一个"式中："段落（如 5.3.3-1 与 5.3.3-2）
          2. "式中"段落会跨页，在公式区所在的片段里取不全（如 5.3.3 的 A 在第 17 页）
        """
        by_id = {n["node_id"]: n for n in nodes}
        for node in nodes:
            if node["type"] != "clause":
                continue
            blocks = _where_blocks(node["content"])
            fids = [c for c in node["child_ids"]
                    if by_id.get(c, {}).get("type") == "formula"]
            if not blocks or not fids:
                continue
            if len(blocks) == 1:                      # 共用一段 → 全部公式一致
                assign = {fid: blocks[0] for fid in fids}
            elif len(blocks) == len(fids):            # 一一对应
                assign = dict(zip(fids, blocks))
            else:                                     # 数量对不上就不冒险映射
                continue
            for fid, variables in assign.items():
                f = by_id[fid]
                f["body"]["variables"] = variables
                f["content"] = _formula_content(f["num"], variables)


# ---------------------------------------------------------------- 辅助函数

def _body_size(page) -> float:
    """页内正文字号（中位数），用于区分标题行与正文行。"""
    sizes = [c.get("size") or 0 for c in page.chars if (c.get("size") or 0) > 0]
    return statistics.median(sizes) if sizes else 10.0


def _heading_kind(line: dict, body_size: float) -> "str | None":
    """
    判定章 / 节标题。条号交给 CLAUSE_RE 处理，这里必须先排除，
    否则 SECTION_RE 会把 "5.1.1 …" 误切成 节 5.1。

    判据用"居中"（x0 > 200）而不是字号——本 PDF 里章标题与正文同为 14pt，
    字号区分不了；居中 + 编号形态才可靠（实测：章 x0=259，节 x0=254，正文 x0=71）。
    """
    text = line["text"].strip()
    if L.CLAUSE_RE.match(text) or L.TABLE_TITLE_RE.match(text) or L.FIGURE_TITLE_RE.match(text):
        return None
    if text.endswith(("；", ";", "。", "，", ",", "：")) or len(text) > 30:
        return None
    if line["x0"] <= 200:
        return None
    # 标题必须以中文字开头：否则公式行 "0.9×1.5wklah2" 会被 SECTION_RE 当成 "节 0.9"
    m_sec = L.SECTION_RE.match(text)
    if m_sec and re.match(r"[\u4e00-\u9fff]", m_sec.group(2).strip()):
        return "section"
    m_chp = L.CHAPTER_RE.match(text)
    if (m_chp and len(text) <= 25
            and re.match(r"[\u4e00-\u9fff]", m_chp.group(2).strip())):
        return "chapter"
    return None


def _split_heading(text: str) -> tuple[str, str]:
    m = L.SECTION_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    m = L.CHAPTER_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", text.strip()


def _split_items(seg: dict) -> list[tuple[str, list[dict]]]:
    """
    切款/项：只在条文含"下列…"时启用，且款号必须从 1 开始连续，
    避免把续行里出现的数字误当款号。
    """
    body = seg["text"]
    if "下列" not in body and "应包括" not in body:
        return []
    items: list[tuple[str, list[dict]]] = []
    expect = 1
    head = seg["lines"][0]["text"]
    for line in seg["lines"][1:]:
        text = line["text"].strip()
        m = L.ITEM_RE.match(text)
        if m and int(m.group(1)) == expect and not L.CLAUSE_RE.match(text):
            items.append((m.group(1), [line]))
            expect += 1
        elif items:
            items[-1][1].append(line)
        elif text != head:
            continue
    return items


def _variables_from(lines: Sequence[dict], formula: dict) -> list[dict]:
    """从公式区之后的"式中："段落抽变量表；符号里的下标在行重建时已贴回主字符。"""
    try:
        start = next(i for i, l in enumerate(lines) if l is formula["lines"][-1])
    except StopIteration:
        start = -1
    block: list[str] = []
    seen_where = False
    for line in lines[start + 1:]:
        text = line["text"]
        if text.startswith("式中"):
            seen_where = True
        if seen_where:
            block.append(text)
        if seen_where and L.CLAUSE_RE.match(text):
            break
    return _parse_vars("".join(block))


def _parse_vars(blob: str) -> list[dict]:
    """从一段"式中"文本里抽变量：符号——释义（单位）。"""
    out = []
    for m in re.finditer(r"([A-Za-z\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff]*)\s*——\s*([^；;]+)", blob):
        meaning = m.group(2).strip()
        unit = None
        um = re.search(r"[（(]([^（）()]{1,12})[）)]", meaning)
        if um and re.search(r"(kN|mm|m2|m3|N/|MPa|kPa|kg|N·m)", um.group(1)):
            unit = um.group(1)
        out.append({"symbol": m.group(1), "meaning": meaning, "unit": unit})
    return out


def _where_blocks(clause_text: str) -> list[list[dict]]:
    """按出现顺序抽出一个条文里的所有"式中："段落，每段转成变量列表。"""
    marks = [m.end() for m in re.finditer(r"式中\s*[：:]", clause_text)]
    blocks = []
    for i, start in enumerate(marks):
        end = marks[i + 1] - len("式中：") if i + 1 < len(marks) else len(clause_text)
        blocks.append(_parse_vars(clause_text[start:end]))
    return [b for b in blocks if b]


def _formula_content(num: str, variables: Sequence[dict]) -> str:
    """公式的检索代理文本：式号 + 变量语义（公式本体符号对 embedding 无意义）。"""
    return " ".join(x for x in [
        f"式（{num}）",
        "；".join(f"{v['symbol']}={v['meaning']}" for v in variables),
    ] if x)


def _structural_keywords(num: str, section: "str | None") -> list[str]:
    """结构型关键词：编号本身。BM25 对编号的精确匹配远强于向量。"""
    kws = [num, f"第{num}条"]
    if section:
        kws.append(section.split()[0])
    return kws


def _merge_continuation(node: dict, page_no: int, bbox: list[float], text: str) -> None:
    """把跨页/被表格打断的续接文本并回上一条文，并合并页码与坐标。"""
    node["content"] = (node["content"] + "\n" + L.norm_text(text)).strip()
    _merge_page_bbox(node, page_no, bbox)
    node["refs"] = extract_refs(node["content"], node["standard_id"])


def _append_line(node: dict, page_no: int, line: dict) -> None:
    """把一行续接到已有节点（款/项的跨页续写），同时更新内容与坐标。"""
    node["content"] = (node["content"] + "\n" + L.norm_text(line["text"])).strip()
    _merge_page_bbox(node, page_no, L.line_bbox([line]))
    node["refs"] = extract_refs(node["content"], node["standard_id"])


def _merge_page_bbox(node: dict, page_no: int, bbox: list[float]) -> None:
    if page_no in node["pages"]:
        i = node["pages"].index(page_no)
        cur = node["bboxes"][i]
        node["bboxes"][i] = [min(cur[0], bbox[0]), min(cur[1], bbox[1]),
                             max(cur[2], bbox[2]), max(cur[3], bbox[3])]
    else:
        node["pages"].append(page_no)
        node["bboxes"].append(list(bbox))
