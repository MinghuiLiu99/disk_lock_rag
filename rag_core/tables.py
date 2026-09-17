# -*- coding: utf-8 -*-
"""
附表工具化：把"查值表"和"表单模板"从文本块改造成可查询结构。

为什么要做这件事——三类表混在一起当文本块会有三个问题：

1. **吃上下文**。JGJ 表 C.0.1（稳定系数）代理文本 2213 字、DB11 表 A.0.1
   8496 字，而上下文预算大约 9000 字，一张表就能占掉大半。
2. **代理文本没有语义**。表 C.0.1 的表头是 λ 的个位数字，展平成
   "λ=100；0=0.588；1=0.580…"——那个 `0=` 是"λ 的个位"，不是列名。
3. **本质是查值不是检索**。问"λ=100 的 φ 是多少"应该像查字典一样按参数取值。

于是把表分成三类：
  · lookup   —— 值查表。展开成 (键 → 值) 记录，提供 lookup()；
                 nodes 的 content 换成"表头 + 参数 + 示例值"的短代理。
  · template —— 表单模板（验收记录表）。不是知识，不进检索，只作模板提供。
  · grid     —— 普通表，保持原样。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

__all__ = ["classify_table", "build_tool", "brief_content", "TableTools"]


# 原文用这些符号表示"本格无值"（如稳定系数表超出 λ 范围的那几格）
_PLACEHOLDERS = {"—", "–", "-", "－", "／", "/", "…", "···", ""}


def _clean(cell: Any) -> str:
    v = ("" if cell is None else str(cell)).strip()
    return "" if v in _PLACEHOLDERS else v


def _drop_constant_columns(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[int]:
    """
    找出"整列恒定的废列"——原表为了表现合并表头，常把同一个词组重复到整列。
    实测 JGJ 表 A.0.1 的第 2 列整列都是"地面粗糙度 μz 类别"，它没有任何信息量，
    留着还会让代理文本变成"地面粗糙度…=地面粗糙度…"这种废话。
    """
    drop = []
    ncol = max((len(r) for r in body), default=0)
    for c in range(ncol):
        name = _clean(header[c]) if c < len(header) else ""
        vals = {_clean(r[c]) for r in body if c < len(r)}
        vals.discard("")
        # 三种情况都是废列：
        #   ① 原表这一列没有表头 —— pdfplumber 多切出来的列。实测 DB11 表 A.0.1：
        #      6 个数据列被抽成 11 列，不剔除会让属性错位（型号 B-LG-3000 混进别人的单重）
        #   ② 整列全空   ③ 整列同值 —— 原表用重复词组表现合并表头
        if not name or len(vals) <= 1:
            drop.append(c)
    return drop


# 这些列名天然是行的标识符，优先拿来当键。
# 注意**不含"名称/类别"**——它们是分组列（"立杆"跨十几行），当键会让行互相覆盖。
_KEY_HINTS = ("型号", "编号", "序号", "代码", "标识")


def _pick_key_column(header: Sequence[str], body: Sequence[Sequence[str]]) -> int:
    """
    选一列作行键。

    默认用第 1 列，但它常常是"分组列"而不是标识列——DB11 表 A.0.1 的第 1 列是
    "名称"（立杆/水平杆…，113 行里只有 13 个唯一值），拿它当键会让后面的行互相覆盖。
    实际能唯一标识构配件的是"型号"列（Z-LG-500…，74 个唯一值）。

    规则：列名命中 _KEY_HINTS 的优先；否则取唯一值最多的列。
    """
    ncol = max((len(r) for r in body), default=0)
    if ncol == 0:
        return 0

    def uniq(c: int) -> int:
        return len({_clean(r[c]) for r in body if c < len(r) and _clean(r[c])})

    for c, name in enumerate(header):
        if any(h in (name or "") for h in _KEY_HINTS) and uniq(c) >= 2:
            return c
    return max(range(ncol), key=uniq)


# ------------------------------------------------------------------ 分类

def classify_table(title: str, rows: Sequence[Sequence[str]]) -> str:
    """判断一张表的类型。只看结构特征，不看格数——格数会把小表误判成大表。"""
    t = title or ""
    if any(k in t for k in ("记录表", "验收记录", "验收表")):
        return "template"
    rows = [list(r) for r in (rows or []) if any((c or "").strip() for c in r)]
    if len(rows) < 3:
        return "grid"
    header = [(c or "").strip() for c in rows[0]]
    body = rows[1:]
    # 表头第一格是参数名（λ / 离地高度 / 外径…），后面挂着一串可用作键的值
    if not header or not header[0]:
        return "grid"
    keys = [(r[0] or "").strip() for r in body if r and (r[0] or "").strip()]
    if len(keys) < 3:
        return "grid"
    # 行键可解析成数值，或行键是高重复度的名称（DB11 构配件规格表的"立杆/水平杆"）
    numeric = sum(1 for k in keys if re.fullmatch(r"\d+(?:\.\d+)?", k))
    if numeric >= len(keys) * 0.8:
        return "lookup"
    if len(rows) >= 20:                      # 大规格表也算查值表（按键找行）
        return "lookup"
    return "grid"


# ------------------------------------------------------------------ 展开

def _digit_grid(header: Sequence[str], body: Sequence[Sequence[str]]) -> "dict | None":
    """
    「十位为行、个位为列」的数字矩阵 —— JGJ 表 C.0.1/C.0.2 的稳定系数表。
    必须拼回完整的键（λ=120 行 + 个位 5 → λ=125），否则永远只能查到整十数。
    """
    cols = [str(c).strip() for c in header[1:]]
    if not cols or not all(re.fullmatch(r"\d", c) for c in cols):
        return None
    keys = [str(r[0]).strip() for r in body if r and str(r[0]).strip()]
    if not keys or not all(re.fullmatch(r"\d+", k) for k in keys):
        return None
    records: dict[str, str] = {}
    for row in body:
        base = str(row[0]).strip()
        if not base:
            continue
        for j, cell in enumerate(row[1:]):
            value = _clean(cell) if j + 1 < len(cols) + 1 else ""
            if not value:
                continue
            records[str(int(base) + int(cols[j]))] = value
    if len(records) < 20:
        return None
    return {
        "kind": "lookup",
        "layout": "digit_grid",
        "key_name": str(header[0]).strip() or "λ",
        "value_name": "φ",
        "records": records,
        "range": [min(int(k) for k in records), max(int(k) for k in records)],
    }


def build_tool(title: str, rows: Sequence[Sequence[str]], header_rows: int = 1,
               min_chars: int = 800, orig_chars: int = 0) -> "dict | None":
    """
    把一张表转成可查询结构；不是查值表、或本来就不长的话返回 None。

    min_chars / orig_chars 是门槛：原代理文本不到这个长度的表（如 JGJ 表 B.0.1
    那种 129 字的小常量表）保持原样更好——工具化反而丢掉了"一眼看全"的优势。
    门槛用**原代理文本长度**而不是估算的记录大小，因为它才是真正要省的量。
    """
    if orig_chars and orig_chars < min_chars:
        return None
    kind = classify_table(title, rows)
    if kind == "template":
        return {"kind": "template", "note": "表单模板，不进检索"}
    if kind != "lookup":
        return None
    rows = [list(r) for r in rows if any((c or "").strip() for c in r)]
    header = [(c or "").strip() for c in rows[0]]
    body = rows[1:]
    # 剔除整列恒定的废列
    drop = set(_drop_constant_columns(header, body))
    if drop:
        keep = [i for i in range(len(header)) if i not in drop]
        header = [header[i] for i in keep]
        body = [[(r[i] if i < len(r) else "") for i in keep] for r in body]
    digit = _digit_grid(header, body)
    if digit:
        return digit
    # 通用行查值表：选一列作键（不一定是第 1 列），其余列是属性
    kc = _pick_key_column(header, body)
    cols = [h or f"列{i+1}" for i, h in enumerate(header) if i != kc] or ["值"]
    records: dict[str, dict] = {}
    for row in body:
        key = _clean(row[kc]) if kc < len(row) else ""
        if not key:
            continue
        item = {}
        for name, c in zip(cols, [i for i in range(len(header)) if i != kc]):
            value = _clean(row[c]) if c < len(row) else ""
            if value:
                item[name] = value
        if item:
            records.setdefault(key, {}).update(item)   # 同名键（纵向合并）合并属性
    if not records:
        return None
    return {"kind": "lookup", "layout": "row",
            "key_name": header[kc] or "键",
            "columns": cols, "records": records}


# ------------------------------------------------------------- 短代理文本

def brief_content(title: str, tool: dict, sample: int = 4) -> str:
    """
    查值表的检索代理文本：表头 + 参数 + 示例值。
    示例值不只是"好看"——常见取值直接出现在材料里，模型不查表也能答对；
    精确值再走查表工具。
    """
    if tool["kind"] == "template":
        return f"{title}\n（表单模板，实际使用时应按附录提供的记录表逐项填写）"
    key, val = tool["key_name"], tool.get("value_name", "值")
    records = tool["records"]
    if tool["layout"] == "digit_grid":
        nums = sorted(int(k) for k in records)
        picks = [nums[0], nums[len(nums) // 4], nums[len(nums) // 2], nums[-1]][:sample]
        lines = [f"{key}={n} → {val}={records[str(n)]}" for n in picks]
        rng = tool["range"]
        return (f"{title}\n按{key}取值（{key} 范围 {rng[0]}~{rng[1]}，共 {len(records)} 个值）\n"
                + "；".join(lines) + "\n（全表已转为可查询结构，非示例值需按精确参数查表）")
    keys = list(records)[:sample]
    lines = []
    for k in keys:
        attrs = "；".join(f"{c}={v}" for c, v in list(records[k].items())[:3])
        lines.append(f"{key}={k} → {attrs}")
    return (f"{title}\n按{key}取值（共 {len(records)} 行）\n"
            + "\n".join(lines) + "\n（全表已转为可查询结构，可按参数精确查表）")


# ------------------------------------------------------------------ 查询

class TableTools:
    """查值表的运行时容器。由 build_index 时从 nodes 构建，供生成层调用。"""

    def __init__(self, tools: "dict[str, dict] | None" = None):
        self.tools = tools or {}

    @classmethod
    def from_nodes(cls, nodes: Sequence[dict]) -> "TableTools":
        tools = {}
        for n in nodes:
            if n["type"] != "table" or not n.get("num"):
                continue
            body = n.get("body") or {}
            tool = body.get("tool")
            if tool:
                tools[f"{n['standard_id']}:{n['num']}"] = {**tool, "title": body.get("title", ""),
                                                            "node_id": n["node_id"]}
        return cls(tools)

    def lookup(self, standard_id: str, table_num: str, key: Any) -> "dict | None":
        """精确查值；数字键会先尝试精确匹配，再退到最接近的键。"""
        tool = self.tools.get(f"{standard_id}:{table_num}")
        if not tool or tool["kind"] != "lookup":
            return None
        recs = tool["records"]
        k = str(key).strip()
        if isinstance(key, float) and key.is_integer():
            k = str(int(key))
        hit = recs.get(k)
        if hit is None and tool["layout"] == "digit_grid":
            nums = sorted(int(x) for x in recs)
            try:
                target = int(float(k))
            except (TypeError, ValueError):
                return None
            # 超出表范围太多就不认。否则型号里的数字会被当成长细比——
            # 实测 "Z-LG-3000" 里的 3000 会就近命中 λ=250，给出一条无关的查表结果。
            if target < nums[0] - 5 or target > nums[-1] + 5:
                return None
            near = min(nums, key=lambda x: abs(x - target))
            return {"key": near, "value": recs[str(near)], "exact": near == target,
                    "title": tool["title"], "key_name": tool["key_name"],
                    "value_name": tool.get("value_name", "值"), "node_id": tool["node_id"]}
        if hit is None:
            return None
        return {"key": k, "value": hit, "exact": True, "title": tool["title"],
                "key_name": tool["key_name"],
                "value_name": tool.get("value_name", ""),
                "columns": tool.get("columns", []),
                "node_id": tool["node_id"]}

    def describe(self) -> str:
        lookups = {k: v for k, v in self.tools.items() if v.get("kind") == "lookup"}
        if not lookups:
            return "（无查值表）"
        return "；".join(f"{k.split(':')[-1]}（{v.get('key_name')}）" for k, v in lookups.items())
