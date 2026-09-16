# -*- coding: utf-8 -*-
"""
节点字段规范 v1.1 —— 枚举、构造、校验、关联、序列化。

26 个字段（NODE_FIELDS）为所有节点共有；类型专属内容放 body。
解析器、索引层、前端全部依赖本模块，改字段先改这里。

三条设计原则：
  1. 每个字段只服务于检索 / 生成 / 溯源三者之一
  2. 抽不到写 null，确定为空集合写 []；禁止空字符串
  3. content 是检索表示（可为代理文本），body 是原文载体，两者不可互换
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------- 枚举

# 层级：preamble=0 无编号内容块；表/图/公式/说明与"条"同级
TYPE_LEVEL = {
    "preamble": 0,
    "chapter": 1, "section": 2, "clause": 3, "item": 4,
    "table": 3, "figure": 3, "formula": 3, "explanation": 3,
    # 附录与章同级。编号形态不同（附录 A / A.0.1），且引用时 target_type=appendix
    # 需要能与 chapter 区分开，否则 refs 的 in_scope 判定会对不上。
    "appendix": 1,
}
ALLOWED_TYPES = tuple(TYPE_LEVEL)

REF_TYPES = ("clause", "table", "figure", "formula", "appendix", "external_standard")

# 用词严格程度（对应规范"用词说明"）；"不应/应"、"不宜/宜"极性不同必须分开存
MODALITY_LEVEL = {
    "严禁": 1, "必须": 1,
    "应": 2, "不应": 2,
    "宜": 3, "不宜": 3,
    "可": 4,
}
_MODALITY_PRIORITY = ("严禁", "必须", "不应", "应", "不宜", "宜", "可")

KEYWORD_SOURCES = ("structural", "extracted", "llm", "mixed")
STATUSES = ("ok", "need_review")

# 待复核原因：对应实测出的 12 类排印陷阱，便于按类型批量人工复核
STATUS_REASONS = (
    "cross_page_table_header",  # 跨页表格，表头在上页页尾
    "cross_page_clause",        # 条文跨页续接
    "split_by_table",           # 条文被表格打断，同号已合并
    "broken_line_clause_num",   # 断行导致条号错位
    "table_false_positive",     # 矢量图被误判为表格
    "merged_cell_unknown",      # 合并单元格关系不可靠
    "formula_linear_only",      # 公式只有线性文本、无结构
    "figure_multi_part",        # 图由多个子图组成
    "note_line_merged",         # 表注多行合并
    "running_head_included",    # 页眉页脚混入
    "out_of_scope_ref",         # 存在超出文档范围的引用目标
    "encoding_repaired",        # 做过符号编码修复，需抽检
)

NODE_FIELDS = (
    "node_id", "standard_id", "standard_code", "doc_id", "version", "type", "num",
    "level", "chapter", "section", "path", "pages", "bboxes",
    "content", "body",
    "parent_id", "child_ids", "refs", "referenced_by", "explains", "explained_by",
    "modality", "keywords", "keywords_source",
    "status", "status_reason",
)

REF_FIELDS = ("target_type", "target_num", "target_node_id", "target_display", "in_scope")

DISPLAY_PREFIX = {
    "clause": "第{num}条", "table": "表{num}", "figure": "图{num}",
    "formula": "式（{num}）", "appendix": "附录{num}", "external_standard": "{num}",
}

# 引用抽取正则（已在 JGJ/T 231-2021 全量实测命中形态）
_REF_PATTERNS = (
    ("clause", re.compile(r"第\s*(\d+(?:\.\d+){1,2})\s*条")),
    ("table", re.compile(r"表\s*([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)")),
    ("figure", re.compile(r"图\s*(\d+(?:\.\d+)*(?:-\d+)?)")),
    ("appendix", re.compile(r"附录\s*([A-Z])")),
    ("formula", re.compile(r"式\s*[（(]\s*([A-Z]?\.?\d+(?:\.\d+)+(?:-\d+)?)\s*[）)]")),
    ("external_standard", re.compile(r"((?:GB|JGJ|JG)(?:\s*/\s*T)?\s*\d+)")),
)


# --------------------------------------------------------------------- 工具

def node_id_of(standard_id: str, node_type: str, num: "str | None",
               fallback: "str | None" = None) -> str:
    """三段式 node_id：标准代号:类型:编号。必须带类型段（条 4.3.1 与表 4.3.1 撞号）。"""
    return f"{standard_id}:{node_type}:{num or fallback or 'unknown'}"


def detect_modality(text: str) -> "str | None":
    """抽用词等级，返回原文用词，取最严格者。"""
    if not text:
        return None
    for word in _MODALITY_PRIORITY:
        if word in text:
            return word
    return None


def _target_node_id(standard_id: str, target_type: str, target_num: str) -> "str | None":
    if target_type == "external_standard":
        return None
    return node_id_of(standard_id, target_type, target_num)


def extract_refs(text: str, standard_id: str) -> list[dict]:
    """
    抽出结构化出边引用。in_scope 先留 None，全部节点建完后由 link_structure() 判定。
    附录按 {标准}:appendix:{字母} 命名，附录尚未入库时 in_scope 自然为 False。
    """
    if not text:
        return []
    refs, seen = [], set()
    for target_type, pattern in _REF_PATTERNS:
        for raw in pattern.findall(text):
            num = re.sub(r"\s+", "", raw)
            if target_type == "external_standard":
                num = re.sub(r"(?i)^(GB|JGJ|JG)/?T", r"\1/T", num)
                num = re.sub(r"^(GB|JGJ|JG)(\d)", r"\1 \2", num)
            key = (target_type, num)
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "target_type": target_type,
                "target_num": num,
                "target_node_id": _target_node_id(standard_id, target_type, num),
                "target_display": DISPLAY_PREFIX[target_type].format(num=num),
                "in_scope": None,
            })
    return refs


# --------------------------------------------------------------------- 构造

def table_agent_text(title: str, rows, notes=None) -> str:
    """
    表格检索代理文本：用**扁平表头**把每一行摊平成"列名=值"，供向量/BM25 召回。

    rows/header 传扁平化后的数据行与列名（见 layout.flat_header）；
    直接用原始合并表头会产出"列3=1.70"这种无意义字段名。
    """
    rows = [list(r) for r in (rows or []) if any((c or "").strip() for c in r)]
    if not rows:
        return title or ""
    header = [(c or "").replace("\n", " ").strip() for c in rows[0]]
    parts = [title.strip()] if title else []
    for row in rows[1:]:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if not cells or not cells[0]:
            continue
        lead = cells[0]
        # 转置型表：行首是符号（如 βH），表头各列才是它的取值条件（如 H≤8）
        # 例：表 5.3.2 "搭设高度H(m) | H≤8 | 8<H≤16 …" + 行 "βH | 1.00 | 1.05 …"
        if re.fullmatch(r"[A-Za-z\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff']*", lead):
            pairs = [f"{c}（{header[i]}）" for i, c in enumerate(cells[1:], start=1)
                     if c and i < len(header) and header[i]]
            cond = f"{header[0]}：" if header and header[0] else ""
            parts.append(f"{cond}{lead}=" + "；".join(pairs))
            continue
        kv = [f"{header[i] if i < len(header) and header[i] else '列' + str(i + 1)}={c}"
              for i, c in enumerate(cells) if c]
        if kv:
            parts.append("；".join(kv))
    if notes:
        joined = " ".join(n.strip() for n in notes if n)
        parts.append(joined if joined.startswith("注") else "注：" + joined)
    return "\n".join(parts)


class NodeBuilder:
    """
    节点工厂：绑定标准与文档元数据，按类型造节点。

        b = NodeBuilder("JGJ231", "JGJ/T 231-2021", "jgj231_2021_ch5_test", "2021")
        b.set_context(chapter="5 结构设计", section="5.1 一般规定")
        n = b.clause("5.1.1", text, pages=[13], bboxes=[[70.9, 153.2, 523.4, 246.0]])
    """

    def __init__(self, standard_id: str, standard_code: str, doc_id: str, version: str):
        self.standard_id = standard_id
        self.standard_code = standard_code
        self.doc_id = doc_id
        self.version = version
        self.chapter_title: "str | None" = None
        self.section_title: "str | None" = None
        self._seq = 0

    def set_context(self, chapter: "str | None" = None,
                    section: "str | None" = None) -> "NodeBuilder":
        if chapter is not None:
            self.chapter_title = chapter
        if section is not None:
            self.section_title = section
        return self

    def _next_fallback(self) -> str:
        self._seq += 1
        return f"seq{self._seq:04d}"

    def _base(self, node_type: str, num: "str | None", content: str,
              pages: Sequence[int], bboxes: Sequence, *, body: Any = None,
              refs: "list[dict] | None" = None, modality: "str | None" = None,
              auto_modality: bool = True, keywords: "Sequence[str] | None" = None,
              keywords_source: "str | None" = None,
              status: str = "ok", status_reason: "str | None" = None) -> dict:
        kws = list(keywords or [])
        return {
            "node_id": node_id_of(self.standard_id, node_type, num,
                                  fallback=None if num else self._next_fallback()),
            "standard_id": self.standard_id,
            "standard_code": self.standard_code,
            "doc_id": self.doc_id,
            "version": self.version,
            "type": node_type,
            "num": num,
            "level": TYPE_LEVEL[node_type],
            "chapter": self.chapter_title,
            "section": self.section_title,
            "path": " / ".join(x for x in (self.chapter_title, self.section_title) if x),
            "pages": [int(p) for p in pages],
            "bboxes": [list(map(float, b)) for b in bboxes],
            "content": content,
            "body": body,
            "parent_id": None,
            "child_ids": [],
            "refs": refs if refs is not None else extract_refs(content, self.standard_id),
            "referenced_by": [],
            "explains": [],
            "explained_by": [],
            "modality": modality if modality is not None else (
                detect_modality(content) if auto_modality else None),
            "keywords": kws,
            "keywords_source": (keywords_source or "structural") if kws else None,
            "status": status,
            "status_reason": status_reason,
        }

    def chapter_node(self, num: str, title: str, pages, bboxes, **kw) -> dict:
        return self._base("chapter", num, f"{num} {title}".strip(), pages, bboxes,
                          auto_modality=False, refs=[], **kw)

    def section(self, num: str, title: str, pages, bboxes, **kw) -> dict:
        return self._base("section", num, f"{num} {title}".strip(), pages, bboxes,
                          auto_modality=False, refs=[], **kw)

    def clause(self, num: str, content: str, pages, bboxes, **kw) -> dict:
        return self._base("clause", num, content, pages, bboxes, **kw)

    def item(self, clause_num: str, item_no: str, content: str, pages, bboxes, **kw) -> dict:
        return self._base("item", f"{clause_num}-{item_no}", content, pages, bboxes, **kw)

    def table(self, num: str, title: str, rows, pages, bboxes, *, notes=None,
              header_rows: int = 1, fill_down: bool = False, units=None,
              markdown=None, markdown_flat=None, header_flat=None,
              cross_page=None, agent_text: "str | None" = None,
              status: str = "ok", status_reason: "str | None" = None, **kw) -> dict:
        data_rows = [list(r) for r in rows[header_rows:]]
        if header_flat:
            content = agent_text or table_agent_text(title, [list(header_flat)] + data_rows, notes)
        else:
            content = agent_text or table_agent_text(title, rows, notes)
        body = {
            "table_id": num, "title": title,
            "rows": [["" if c is None else str(c) for c in r] for r in rows],
            "header_rows": header_rows, "fill_down": fill_down,
            "header_flat": list(header_flat or []),
            "notes": list(notes or []), "units": units,
            "markdown": markdown, "markdown_flat": markdown_flat,
            "cross_page": list(cross_page) if cross_page else None,
        }
        if cross_page and status == "ok":
            status, status_reason = "need_review", status_reason or "cross_page_table_header"
        return self._base("table", num, content, pages, bboxes, body=body,
                          auto_modality=False, refs=[],
                          status=status, status_reason=status_reason, **kw)

    def figure(self, num: str, title: str, pages, bboxes, *, legend=None,
               sub_labels=None, image_path=None, is_vector: bool = True, **kw) -> dict:
        content = " ".join(x for x in [title, *list(legend or [])] if x)
        body = {
            "figure_id": num, "title": title, "legend": list(legend or []),
            "sub_labels": list(sub_labels or []), "image_path": image_path,
            "is_vector": is_vector, "part_count": len(sub_labels or []) or 1,
        }
        reason = "figure_multi_part" if len(sub_labels or []) > 1 else None
        return self._base("figure", num, content, pages, bboxes, body=body,
                          auto_modality=False, refs=[],
                          status="need_review" if reason else "ok",
                          status_reason=reason, **kw)

    def formula(self, num: str, variables, pages, bboxes, *, latex=None,
                linear_text=None, image_path=None, caption=None, **kw) -> dict:
        variables = [dict(v) for v in (variables or [])]
        content = " ".join(x for x in [
            caption or f"式（{num}）",
            "；".join(f"{v.get('symbol')}={v.get('meaning')}" for v in variables),
        ] if x)
        body = {"equation_id": num, "latex": latex, "linear_text": linear_text,
                "variables": variables, "image_path": image_path}
        return self._base("formula", num, content, pages, bboxes, body=body,
                          auto_modality=False, refs=[], **kw)

    def explanation(self, num: str, content: str, pages, bboxes, *,
                    explains=None, **kw) -> dict:
        node = self._base("explanation", num, content, pages, bboxes,
                          auto_modality=False, refs=[], **kw)
        node["explains"] = list(explains or [])
        return node

    def preamble(self, content: str, pages, bboxes, **kw) -> dict:
        return self._base("preamble", None, content, pages, bboxes,
                          auto_modality=False, refs=[], **kw)

    def appendix(self, letter: str, title: str, pages, bboxes, **kw) -> dict:
        """附录标题（附录 A / B / C / D），与章同级但类型独立，便于 refs 的 in_scope 判定。"""
        return self._base("appendix", letter, f"附录{letter} {title}".strip(),
                          pages, bboxes, auto_modality=False, refs=[], **kw)


# --------------------------------------------------------------------- 校验

def _bbox_ok(b: Any) -> bool:
    """接受 [x0,top,x1,bottom]，或同页多区域 [[...],[...]]。"""
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(
            isinstance(v, (int, float)) for v in b):
        return True
    if isinstance(b, (list, tuple)) and len(b) > 0 and all(
            isinstance(x, (list, tuple)) and len(x) == 4 for x in b):
        return True
    return False


def validate_node(node: dict) -> list[str]:
    """校验单个节点，返回问题列表（空 = 通过）。"""
    problems = [f"缺字段: {f}" for f in NODE_FIELDS if f not in node]
    if problems:
        return problems
    t = node["type"]
    if t not in ALLOWED_TYPES:
        problems.append(f"type 非法: {t}")
    if node["level"] != TYPE_LEVEL.get(t):
        problems.append(f"level({node['level']}) 与 type({t}) 不匹配")
    expect = node_id_of(node["standard_id"], t, node["num"])
    if node["num"] and node["node_id"] != expect:
        problems.append(f"node_id 不自洽: {node['node_id']} != {expect}")
    if not isinstance(node["content"], str) or not node["content"].strip():
        problems.append("content 为空")
    elif t == "clause" and len(node["content"]) < 12:
        problems.append(f"content 过短({len(node['content'])}字)，疑似被切碎")
    elif t == "item" and len(node["content"]) < 6:
        problems.append(f"content 过短({len(node['content'])}字)，疑似被切碎")
    if not node["pages"]:
        problems.append("pages 为空")
    if len(node["bboxes"]) != len(node["pages"]):
        problems.append(f"pages({len(node['pages'])}) 与 bboxes({len(node['bboxes'])}) 不等长")
    else:
        for i, b in enumerate(node["bboxes"]):
            if not _bbox_ok(b):
                problems.append(f"bboxes[{i}] 形态非法: {b!r}")
                break
    for r in node["refs"]:
        if not isinstance(r, dict) or tuple(r) != REF_FIELDS:
            problems.append(f"ref 字段不合法: {r!r}")
            break
    if node["modality"] is not None and node["modality"] not in MODALITY_LEVEL:
        problems.append(f"modality 非法: {node['modality']}")
    if node["status"] not in STATUSES:
        problems.append(f"status 非法: {node['status']}")
    if node["status"] == "need_review" and not node["status_reason"]:
        problems.append("need_review 但缺 status_reason")
    if node["status"] == "ok" and node["status_reason"]:
        problems.append("status=ok 却有 status_reason")
    if node["keywords"] and not node["keywords_source"]:
        problems.append("有关键词但无 keywords_source")
    if not node["keywords"] and node["keywords_source"]:
        problems.append("无关键词却有 keywords_source")
    for f in ("child_ids", "referenced_by", "explains", "explained_by", "keywords"):
        if not isinstance(node[f], list):
            problems.append(f"{f} 不是列表")
    return problems


def validate_all(nodes: Iterable[dict]) -> dict:
    bad, n = [], 0
    for node in nodes:
        n += 1
        p = validate_node(node)
        if p:
            bad.append((node.get("node_id", "?"), p))
    return {"total": n, "bad_count": len(bad), "bad": bad}


# --------------------------------------------------------------------- 关联

def link_structure(nodes: Sequence[dict]) -> list[dict]:
    """
    一次性建立：父子链、双向引用边、条文说明双向关联、in_scope 标记。
    表/图/公式按传入顺序挂到其上方最近的"条"；调用前请把节点按版面顺序排好。
    原地修改并返回同一列表。
    """
    by_id = {n["node_id"]: n for n in nodes}
    stack: dict[int, str] = {}
    last_clause = None
    for node in nodes:
        if node["type"] in ("table", "figure", "formula"):
            node["parent_id"] = last_clause
            continue
        lv = node["level"]
        stack = {k: v for k, v in stack.items() if k < lv}
        # 取栈里"层级小于自己"的最近一个，而不是严格的 lv-1：
        # 附录是 附录(1) → 条(3)，中间没有"节"这一层，严格按 lv-1 会挂不上父节点。
        node["parent_id"] = None
        for k in sorted(stack, reverse=True):
            if k < lv:
                node["parent_id"] = stack[k]
                break
        # appendix 与 chapter 同级，也要压栈，否则附录条文挂不到附录节点上
        if node["type"] in ("chapter", "section", "clause", "appendix"):
            stack[lv] = node["node_id"]
        if node["type"] == "clause":
            last_clause = node["node_id"]

    for node in nodes:
        pid = node["parent_id"]
        if pid and pid in by_id:
            parent = by_id[pid]
            if node["node_id"] not in parent["child_ids"]:
                parent["child_ids"].append(node["node_id"])

    for node in nodes:
        for ref in node["refs"]:
            tid = ref["target_node_id"]
            ref["in_scope"] = bool(tid and tid in by_id)
            if ref["in_scope"]:
                by_id[tid]["referenced_by"].append(node["node_id"])

    for node in nodes:
        if node["type"] != "explanation":
            continue
        for cid in node["explains"]:
            target = by_id.get(cid)
            if target is not None and node["node_id"] not in target["explained_by"]:
                target["explained_by"].append(node["node_id"])
    return list(nodes)


# --------------------------------------------------------------------- IO

def to_jsonl(nodes: Iterable[dict], path: "str | Path") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
            n += 1
    return n


def from_jsonl(path: "str | Path") -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(nodes: Sequence[dict]) -> dict:
    """产物质量总览：类型分布、待复核清单、引用边内外。"""
    by_type, review, refs_total, refs_in = {}, [], 0, 0
    for n in nodes:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
        if n["status"] == "need_review":
            review.append((n["node_id"], n["status_reason"]))
        for r in n["refs"]:
            refs_total += 1
            refs_in += 1 if r["in_scope"] else 0
    return {
        "节点总数": len(nodes),
        "类型分布": by_type,
        "待复核": len(review),
        "待复核清单": review,
        "引用边总数": refs_total,
        "其中范围内": refs_in,
        "引用范围外": refs_total - refs_in,
    }
