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
from . import tables as T
from .schema import NodeBuilder, node_id_of, link_structure, extract_refs, table_agent_text


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
                 version: str, *, standard_name: str = "", page_offset: int = 0,
                 figure_dir: str | Path | None = None,
                 y_tol: float = 6.0, verbose: bool = False):
        self.pdf_path = Path(pdf_path)
        self.builder = NodeBuilder(standard_id, standard_code, doc_id, version,
                                   standard_name=standard_name)
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
        notes_from: "int | None" = None      # 条文说明区的起始页
        for idx, page in enumerate(pdf.pages, start=1):
            book_page = idx + self.page_offset
            lines = L.repair_broken_lines(L.drop_running_heads(
                L.page_lines(page, y_tol=self.y_tol), page))
            # 条文说明的封面页有"条文说明"字样，从这一页起进入说明区。
            # 说明区的条号与正文完全相同（1.0.1 等），不区分会产生重复 node_id。
            #
            # 判据必须收紧成"像标题页"，否则会误判：
            #   · JGJ/T 231 第一次出现在 p42 的独立标题页（整页仅 60 字符）✓
            #   · DB11/T 2100 第一次出现在 p6 的**目次页**（整页 4446 字符，
            #     其中一行是目录条目"附：条文说明 ......"）——只看"是否出现"的话，
            #     从第 6 页起整本都会被当成条文说明（实测 244 个 explanation、0 个 clause）。
            # 所以要求：该行去掉空格后基本就是"条文说明"本身，且整页字符很少。
            if notes_from is None:
                page_chars = sum(len(l["text"]) for l in lines)
                if page_chars < 800:
                    for l in lines:
                        t = re.sub(r"\s+", "", l["text"])
                        if "条文说明" in t and len(t) <= 12:
                            notes_from = book_page
                            break
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
                          "notes": notes_from is not None and book_page >= notes_from,
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
            # 目录行直接跳过（不产出节点，也不中断当前段落）
            if not blocked and L.TOC_LINE_RE.search(line["text"]):
                continue
            if blocked:
                if cur:
                    segs.append(cur); cur = None
                continue
            kind = _heading_kind(line, pg["body_size"])
            if kind is None:
                clause_re = L.NOTES_CLAUSE_RE if pg.get("notes") else L.CLAUSE_RE
                if clause_re.match(line["text"]) or L.APPENDIX_CLAUSE_RE.match(line["text"]):
                    kind = "clause"
                elif L.APPENDIX_TITLE_RE.match(line["text"]):
                    kind = "appendix"
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
        # 续接片段里不新起款：只有本条已至少有一款时才继续编号，
        # 否则会把续接进来的任意"数字开头"的行（如引用标准名录的"9 《…》"）误判成款。
        if not require_list and state.get("next", 1) <= 1:
            return
        body = lines[1:] if require_list else lines
        for line in body:
            text = line["text"].strip()
            m = L.ITEM_RE.match(text)
            if m and int(m.group(1)) == state["next"] and not (
                    L.CLAUSE_RE.match(text) or L.APPENDIX_CLAUSE_RE.match(text)):
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
                # 跨页表头碎片：其表头已被并进下一页的表体，这里不能再单独产出，
                # 否则同一张表会出现两条同名节点（实测表 4.3.1、表 4.4.4 都中招）。
                if t.get("merged_into_next"):
                    continue
                header_flat = L.flat_header(t["cells"], t["header_rows"])
                # 表号可能识别不出来（原文漏印"表"字、或是跨页续表的碎片）。
                # 这类表内容往往是真知识，不能丢，但必须标出来让人复核；
                # 同时不能生成 "表None" 这种垃圾关键词——它会让混库建索引直接崩。
                has_num = bool(t["table_id"])
                kws = ([f"表{t['table_id']}", t["table_id"]] if has_num else ["未编号表"])
                # 附表工具化：查值表（稳定系数/风压系数/截面特性）与表单模板不再把
                # 几百个数字灌进 content——它们的代理文本换成"表头 + 参数 + 示例值"，
                # 完整数据留在 body.rows 里，运行时由 TableTools 提供精确查询。
                plain = table_agent_text(t["title"], t["cells"], t["notes"])
                tool = T.build_tool(t["title"], t["cells"], t["header_rows"],
                                    min_chars=800, orig_chars=len(plain))
                brief = T.brief_content(t["title"], tool) if tool else None
                node = b.table(t["table_id"], t["title"], t["cells"], [bp], [t["bbox"]],
                               notes=t["notes"], header_rows=t["header_rows"],
                               fill_down=t["fill_down"], units=t["units"],
                               markdown=L.to_markdown(t["cells"]),
                               header_flat=header_flat,
                               markdown_flat=L.to_markdown_flat(t["cells"], t["header_rows"],
                                                                header_flat),
                               cross_page=t.get("cross_page"),
                               agent_text=brief,
                               tool=tool,
                               keywords=kws,
                               status="ok" if has_num else "need_review",
                               status_reason=None if has_num else "table_title_missing")
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
            notes = pg["notes"]

            if notes and kind in ("chapter", "section"):
                # 说明区的章/节只用来维护上下文，不单独成节点——它们会与正文章节点重号，
                # 而且作为导航节点价值很低。
                num, title = _split_heading(seg["lines"][0]["text"])
                if kind == "chapter":
                    b.set_context(chapter=f"{num} {title}", section=None)
                else:
                    b.set_context(section=f"{num} {title}")
                current = None
                continue

            if notes and kind == "clause":
                targets = _explanation_targets(seg["text"])
                num = targets[0] if targets else _clause_num(seg["lines"][0]["text"])
                node = b.explanation(num, L.norm_text(seg["text"]), [bp], [seg["bbox"]],
                                     keywords=[num, f"第{num}条", "条文说明"])
                nodes.append(node)
                self._emit_formulas(nodes, seg["lines"], bp)
                current = node
                continue

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

            if kind == "appendix":
                m = L.APPENDIX_TITLE_RE.match(seg["lines"][0]["text"])
                letter, title = m.group(1), m.group(2).strip()
                b.set_context(chapter=f"附录{letter} {title}", section=None)
                nodes.append(b.appendix(letter, title, [bp], [seg["bbox"]]))
                current = None
                continue

            if kind == "clause":
                num = _clause_num(seg["lines"][0]["text"])
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
        self._link_explanations(nodes)
        return nodes

    @staticmethod
    def _link_explanations(nodes: list[dict]) -> None:
        """
        条文说明 ↔ 正文条文双向关联。
        说明区的条号与正文一一对应（实测覆盖率约 34%，其余条文没有官方说明），
        这是"这条规定的依据是什么"唯一能回答的来源，必须连上。
        """
        by_id = {n["node_id"]: n for n in nodes}
        clauses = {n["num"]: n for n in nodes if n["type"] == "clause"}
        for n in nodes:
            if n["type"] != "explanation":
                continue
            # 一条说明可能覆盖多条条文（"7.4.1~7.4.3" / "7.4.11、7.4.12"），
            # 全都要连上——否则用户问被覆盖的那几条时查不到依据。
            for num in (_explanation_targets(n["content"]) or [n["num"]]):
                c = clauses.get(num)
                if c is not None:
                    if c["node_id"] not in n["explains"]:
                        n["explains"].append(c["node_id"])
                    if n["node_id"] not in c["explained_by"]:
                        c["explained_by"].append(n["node_id"])

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


def _clause_num(text: str) -> str:
    """条文号：正文是 5.3.3，附录是 A.0.1，两种形态统一在这里取。"""
    m = L.CLAUSE_RE.match(text)
    if m:
        return m.group(1)
    m = L.APPENDIX_CLAUSE_RE.match(text)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return ""


_RANGE_RE = re.compile(r"(\d+\.\d+)\.(\d+)\s*[~～]\s*(?:\d+\.\d+\.)?(\d+)")


def _explanation_targets(text: str) -> list[str]:
    """
    条文说明开头覆盖的条文号。
    实测四种写法都要认：
      "1.0.1 本条是…"         单条
      "7.4.1~7.4.3 明确了…"    区间（需展开成 3 条）
      "7.4.11、7.4.12 明确…"   列举
      "5.1.2、5.1.3 给出了…"   列举
    """
    head = re.split(r"[。\n]", text, maxsplit=1)[0][:60]
    nums: list[str] = []
    for m in _RANGE_RE.finditer(head):      # 区间展开
        sec, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        nums += [f"{sec}.{i}" for i in range(a, b + 1)]
    nums += re.findall(r"\d+\.\d+\.\d+", head)
    out, seen = [], set()
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


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
