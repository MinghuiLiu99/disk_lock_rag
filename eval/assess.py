# -*- coding: utf-8 -*-
"""
解析质量评估：把「这份 nodes.jsonl 到底解析得怎么样」变成一个函数。

notebook 里：
    from eval.assess import assess, render

    r = assess("output/JGJ_T_231-2021/nodes.jsonl", pdf="规范/jgj 231-2021.pdf")
    print(render(r))          # 可读报告
    r["score"]                # 0–100 粗分
    r["numbering"]["gaps"]    # 条号跳号（漏切的主要信号）

命令行（不带 --nodes 就读 output/manifest.json，把所有规范都评一遍）：
    python eval/assess.py
    python eval/assess.py --nodes output/JGJ_T_231-2021/nodes.jsonl --pdf "规范/jgj 231-2021.pdf"
    python eval/assess.py --json          # 输出 JSON，便于二次处理

为什么要传 pdf：
    只看 nodes.jsonl，能查的只有「内部自洽」——字段齐不齐、type/level 配不配、
    引用闭不闭合。这些**发现不了漏切**：漏掉一整条，剩下的节点自己照样自洽。
    唯一不依赖人工标准答案的完整性检查，是拿 PDF 原文对账——从每页文本里正则捞
    行首条号，和节点里的条号取差集。差集非空就说明有东西没被切出来。
    扫描件没有文字层，这一步会如实标注「无法对账」，不会假装通过。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_core import from_jsonl, validate_all          # noqa: E402

# Windows 控制台默认 GBK，打印符号会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# PDF 文本里位于行首的条号，两种形态都要认：
#   4.2.1 / 8.0.4  →  数字.数字.数字
#   A.0.1 / D.0.2  →  字母.数字.数字（附录条文）
# 只写第一种会把整本附录误报成"节点多出来的条号"。
CLAUSE_RE = re.compile(r"^\s*((?:[A-Z]|\d{1,2})(?:\.\d{1,2}){2})\s*\S", re.M)

# 引用 target_type 属于「本规范内部」的那些；指向它们却找不到节点 = 悬空引用
SCOPE_TYPES = {"clause", "item", "table", "figure", "formula",
               "appendix", "section", "chapter", "explanation"}

SHORT_CLAUSE = 25          # 少于这么多字，判为疑似被切碎


# ----------------------------------------------------------------- 各分项检查

def _load(source) -> list[dict]:
    """source 可以是 jsonl 路径、路径列表，或者已经是节点列表。"""
    if isinstance(source, (str, Path)):
        return from_jsonl(source)
    if isinstance(source, dict):
        return [source]
    out: list[dict] = []
    for s in source:
        out += _load(s)
    return out


def _numbering(nodes: list[dict]) -> dict:
    """
    条号连续性：按「章.节」分组，看最后一段有没有跳号。

    跳号是漏切最直接的信号（比如 4.2.3 之后直接 4.2.5）。
    但要注意：跳号只是**提示**不是**错误**——规范里偶尔本来就有编号不连续的情况，
    看到跳号要回原文确认，不能直接判解析器有罪。
    """
    groups: dict[str, set[int]] = collections.defaultdict(set)
    seen = collections.Counter()
    for n in nodes:
        num = n.get("num")
        if n["type"] != "clause" or not num:
            continue
        seen[num] += 1
        parts = num.split(".")
        if len(parts) == 3:
            tail = re.sub(r"\D", "", parts[2])
            if tail:
                groups[".".join(parts[:2])].add(int(tail))

    gaps = []
    for head in sorted(groups):
        tails = groups[head]
        missing = [i for i in range(1, max(tails) + 1) if i not in tails]
        if missing:
            gaps.append({"group": head, "missing": missing, "found": sorted(tails)})
    return {"groups": len(groups), "gaps": gaps,
            "duplicates": [k for k, v in seen.items() if v > 1]}


def _review(nodes: list[dict]) -> dict:
    """待复核节点：解析器自己标记「我不确定」的那些（status != ok）。"""
    items = [n for n in nodes if n.get("status") != "ok"]
    reasons = collections.Counter(n.get("status_reason") or "(未填原因)" for n in items)
    return {"count": len(items),
            "rate": round(len(items) / max(1, len(nodes)), 4),
            "reasons": dict(reasons.most_common()),
            "items": [(n["node_id"], n.get("status_reason")) for n in items]}


def _text(nodes: list[dict]) -> dict:
    """
    条文长度分布 + 疑似被切碎的短条文。

    长度统计看 clause+item，但「过短」只对 **clause** 报警：
    款/项（item）本来就经常只有几个字（"1 位移监测；"），拿它当异常纯属制造噪声。
    """
    cl = [n for n in nodes if n["type"] in ("clause", "item")]
    if not cl:
        return {"n": 0, "too_short": []}
    lens = sorted(len(n.get("content") or "") for n in cl)
    only_clause = [n for n in cl if n["type"] == "clause"]
    short = sorted((n for n in only_clause if len(n.get("content") or "") < SHORT_CLAUSE),
                   key=lambda n: len(n["content"]))
    return {"n": len(cl), "min": lens[0], "median": statistics.median(lens),
            "mean": round(statistics.mean(lens), 1), "max": lens[-1],
            "clause_min": min((len(n["content"]) for n in only_clause), default=0),
            "too_short": [(n["node_id"], len(n["content"]), n["content"][:40])
                          for n in short]}


def _refs(nodes: list[dict]) -> dict:
    """引用边：指向本规范内的 / 指向别的规范（正常）的 / 悬空的（异常）。"""
    by_id = {n["node_id"] for n in nodes}
    refs = [r for n in nodes for r in n.get("refs") or []]
    inside = [r for r in refs if r.get("in_scope")]
    outside = [r for r in refs if not r.get("in_scope")]
    dangling = [r for r in outside if r.get("target_type") in SCOPE_TYPES
                and r.get("target_node_id") not in by_id]
    return {"total": len(refs), "inside": len(inside), "outside": len(outside),
            "dangling": len(dangling),
            "dangling_items": [(r.get("target_type"), r.get("target_num"),
                                r.get("target_node_id")) for r in dangling]}


def _tables(nodes: list[dict]) -> dict:
    """表格：有没有转成结构化文本、有没有被工具化、缺不缺标题、是不是空表。"""
    tb = [n for n in nodes if n["type"] == "table"]
    shaped = tooled = 0
    no_title, empty = [], []
    for n in tb:
        body = n.get("body") or {}
        flat = body.get("markdown_flat") or body.get("markdown") or ""
        rows = [x for x in flat.split("\n")
                if x.strip() and not re.match(r"^\|\s*-+", x.strip())]
        if flat:
            shaped += 1
        if body.get("tool"):
            tooled += 1
        if not (body.get("title") or "").strip():
            no_title.append(n["node_id"])
        if len(rows) <= 1:                     # 只剩表头 = 空表
            empty.append(n["node_id"])
    return {"count": len(tb), "shaped": shaped, "tooled": tooled,
            "no_title": no_title, "empty": empty}


def _figures(nodes: list[dict], root: Path) -> dict:
    """图：有没有裁图路径、文件在不在磁盘上。"""
    figs = [n for n in nodes if n["type"] == "figure"]
    no_path, missing = [], []
    for n in figs:
        p = (n.get("body") or {}).get("image_path")
        if not p:
            no_path.append(n["node_id"])
        elif not (root / p).exists():
            missing.append((n["node_id"], p))
    return {"count": len(figs), "no_path": no_path, "missing_file": missing}


def _formulas(nodes: list[dict]) -> dict:
    fm = [n for n in nodes if n["type"] == "formula"]
    with_v = sum(1 for n in fm if (n.get("body") or {}).get("variables"))
    with_ltx = sum(1 for n in fm if (n.get("body") or {}).get("latex"))
    return {"count": len(fm), "with_variables": with_v, "with_latex": with_ltx}


def _tree(nodes: list[dict]) -> dict:
    """目录树：章/节数量，以及挂不上父节点的孤儿。"""
    kinds = collections.Counter(n["type"] for n in nodes)
    orphan = [n["node_id"] for n in nodes
              if n["level"] > 1 and not n.get("parent_id")]
    return {"chapters": kinds.get("chapter", 0), "sections": kinds.get("section", 0),
            "orphans": orphan}


def _numkey(num: str):
    return [int(x) if x.isdigit() else 999 for x in num.split(".")]


def _coverage(pdf, nodes: list[dict]) -> dict:
    """
    跟原文对账——唯一不依赖人工 gold 的完整性检查。

    PDF 里行首出现的条号 减去 节点里的条号，再把差集分两类：
      missing_suspicious → 疑似漏切，带页码和上下文，要人工确认
      missing_quoted     → 原文里是「6.1.7 条规定」这种**引用别的规范**的句子，
                           正则会把它的条号捞进来，属正常的假阳性
      extra_in_nodes     → 条号在原文不在行首（常被表格/图打断），多为提示
    """
    import pdfplumber

    with pdfplumber.open(str(pdf)) as doc:
        page_count = len(doc.pages)
        texts = [(p.extract_text() or "") for p in doc.pages]

    chars = sum(len(t) for t in texts)
    if chars == 0:
        return {"pdf": str(pdf), "pages": page_count, "text_layer": False,
                "note": "扫描件：无文字层，无法与原文对账，需先 OCR"}

    hits: dict[str, list] = collections.defaultdict(list)     # 条号 -> [(页, 上下文)]
    for i, t in enumerate(texts, 1):
        for m in CLAUSE_RE.finditer(t):
            ctx = re.sub(r"\s+", " ", t[m.start():m.start() + 34]).strip()
            hits[m.group(1)].append((i, ctx))

    node_nums = {n["num"] for n in nodes
                 if n.get("num") and n["type"] in ("clause", "explanation")}

    missing = sorted((num for num in hits if num not in node_nums), key=_numkey)
    suspicious, quoted = [], []
    for num in missing:
        occurrences = hits[num]
        is_quote_only = all(re.match(rf"^{re.escape(num)}\s*条", ctx)
                            for _, ctx in occurrences)
        (quoted if is_quote_only else suspicious).append(
            (num, occurrences[0][0], occurrences[0][1]))

    covered = {p for n in nodes for p in n.get("pages") or []}
    return {"pdf": str(pdf), "pages": page_count, "text_layer": True, "chars": chars,
            "pdf_clauses": len(hits), "node_clauses": len(node_nums),
            "missing_suspicious": suspicious, "missing_quoted": quoted,
            "extra_in_nodes": sorted(set(node_nums) - set(hits), key=_numkey),
            "pages_with_nodes": len(covered),
            "pages_without_nodes": sorted(set(range(1, page_count + 1)) - covered)}


# --------------------------------------------------------------------- 综合分

def _score(r: dict) -> float:
    """
    0–100 的粗分，四项扣分。权重是主观定的，**别当成绝对指标**——
    它的用途只有一个：横向比两本规范，或者改了解析器之后看有没有变差。
    够不够写进论文，不靠这个数说话。
    """
    n = max(1, r["nodes"])
    s = 100.0
    s -= 40 * (r["structure"]["bad_count"] / n)     # 结构校验不通过，最重
    s -= 25 * r["review"]["rate"]                   # 待复核比例
    if r["refs"]["total"]:
        s -= 15 * (r["refs"]["dangling"] / r["refs"]["total"])
    cov = r.get("coverage")
    if cov and cov.get("text_layer"):
        s -= 20 * (len(cov["missing_suspicious"]) / (cov["pdf_clauses"] or 1))
    return round(max(0.0, min(100.0, s)), 1)


# ----------------------------------------------------------------------- 入口

def assess(source, pdf=None, root=None) -> dict:
    """
    评估一份（或多份合并的）解析结果，返回指标字典。

    source : nodes.jsonl 路径 / 路径列表 / 节点列表
    pdf    : 可选，原 PDF 路径。给了就做原文条号对账（强烈建议给）
    root   : 可选，解析 image_path 用的根目录，默认项目根
    """
    root = Path(root) if root else ROOT
    nodes = _load(source)
    report = {
        "source": str(source) if isinstance(source, (str, Path)) else f"内存 {len(nodes)} 节点",
        "standard_ids": sorted({n.get("standard_id", "?") for n in nodes}),
        "nodes": len(nodes),
        "by_type": dict(collections.Counter(n["type"] for n in nodes).most_common()),
        "structure": (lambda c: {"total": c["total"], "bad_count": c["bad_count"],
                                 "bad": c["bad"]})(validate_all(nodes)),
        "review": _review(nodes),
        "numbering": _numbering(nodes),
        "text": _text(nodes),
        "refs": _refs(nodes),
        "tables": _tables(nodes),
        "figures": _figures(nodes, root),
        "formulas": _formulas(nodes),
        "tree": _tree(nodes),
        "coverage": _coverage(pdf, nodes) if pdf else None,
    }
    report["score"] = _score(report)
    return report


def render(r: dict) -> str:
    """把 assess() 的结果渲染成可读报告（Markdown 风格，print 或写文件都行）。"""
    o = [f"## 解析质量 · {r['source']}", ""]
    if len(r["standard_ids"]) > 1:
        o += [f"含规范：{'、'.join(r['standard_ids'])}", ""]
    o += [f"**综合分 {r['score']} / 100**　节点 {r['nodes']} 个", ""]

    st = r["structure"]
    if st["bad_count"] == 0:
        o.append(f"- 结构校验：✅ {st['total']} 个节点全部通过")
    else:
        o.append(f"- 结构校验：❌ {st['bad_count']}/{st['total']} 个不通过")
        for nid, probs in st["bad"][:10]:
            o.append(f"    - `{nid}`：{'；'.join(probs)}")

    rv = r["review"]
    if rv["count"] == 0:
        o.append("- 待复核：✅ 0 个")
    else:
        o.append(f"- 待复核：⚠ {rv['count']} 个（{rv['rate'] * 100:.1f}%）")
        for reason, c in rv["reasons"].items():
            o.append(f"    - {reason}：{c}")

    nb = r["numbering"]
    if nb["gaps"]:
        o.append(f"- 条号连续性：⚠ {nb['groups']} 个节，跳号 {len(nb['gaps'])} 处")
        for g in nb["gaps"][:10]:
            o.append(f"    - {g['group']}：缺 {g['missing']}（已有 {g['found']}）")
    else:
        o.append(f"- 条号连续性：✅ {nb['groups']} 个节无跳号")
    if nb["duplicates"]:
        o.append(f"    - 重复条号：{nb['duplicates']}")

    tx = r["text"]
    if tx["n"]:
        o.append(f"- 条文长度：{tx['n']} 条　最短 {tx['min']} / 中位 {tx['median']:.0f} / "
                 f"均值 {tx['mean']:.0f} / 最长 {tx['max']} 字")
    if tx.get("too_short"):
        o.append(f"    - ⚠ 短于 {SHORT_CLAUSE} 字的**条文** {len(tx['too_short'])} 条（款/项不参与，"
                 f"它们本来就短），抽查：")
        for nid, ln, txt in tx["too_short"][:5]:
            o.append(f"        - `{nid}`（{ln} 字）{txt}")

    rf = r["refs"]
    o.append(f"- 引用：{rf['total']} 边　本规范内 {rf['inside']} / 规范外 {rf['outside']}"
             + (f"　⚠ 悬空 {rf['dangling']}" if rf["dangling"] else "　✅ 无悬空"))
    for t, num, tid in rf["dangling_items"][:5]:
        o.append(f"    - {t} {num} → 期望 `{tid}`")

    tb = r["tables"]
    o.append(f"- 表格：{tb['count']} 张　有结构 {tb['shaped']}　工具化 {tb['tooled']}"
             f"　缺标题 {len(tb['no_title'])}　空表 {len(tb['empty'])}")
    for nid in (tb["no_title"] + tb["empty"])[:5]:
        o.append(f"    - 待看 `{nid}`")

    fg = r["figures"]
    o.append(f"- 图：{fg['count']} 张　缺路径 {len(fg['no_path'])}　"
             f"文件缺失 {len(fg['missing_file'])}")
    for nid, p in fg["missing_file"][:5]:
        o.append(f"    - `{nid}` → {p}")

    fm = r["formulas"]
    o.append(f"- 公式：{fm['count']} 个　有变量表 {fm['with_variables']}　"
             f"有 LaTeX {fm['with_latex']}")

    tr = r["tree"]
    o.append(f"- 目录树：章 {tr['chapters']} / 节 {tr['sections']}　"
             + ("✅ 无孤儿节点" if not tr["orphans"] else f"⚠ 孤儿 {len(tr['orphans'])}"))
    for nid in tr["orphans"][:5]:
        o.append(f"    - `{nid}`")

    o.append("")
    cv = r.get("coverage")
    if cv is None:
        o.append("- 原文对账：未做（没传 pdf）")
    elif not cv.get("text_layer"):
        o.append(f"- 原文对账：⛔ {cv['pages']} 页无文字层——{cv['note']}")
    else:
        sus, quoted = cv["missing_suspicious"], cv["missing_quoted"]
        if sus:
            o.append(f"- 原文对账：⚠ PDF {cv['pages']} 页，正文条号 {cv['pdf_clauses']} 个，"
                     f"**疑似漏切 {len(sus)} 个**（需人工确认）")
            for num, page, ctx in sus[:12]:
                o.append(f"    - {num}（p{page}）{ctx}")
        else:
            o.append(f"- 原文对账：✅ PDF {cv['pages']} 页，正文条号 {cv['pdf_clauses']} 个全部命中"
                     + (f"（另有 {len(quoted)} 个是引用别规范的条号，已排除）" if quoted else ""))
        if cv["extra_in_nodes"]:
            o.append(f"    - （提示）{len(cv['extra_in_nodes'])} 个节点条号在原文行首没出现，"
                     f"多为被表格/图打断：{cv['extra_in_nodes'][:8]}")
        if cv["pages_without_nodes"]:
            o.append(f"    - （提示）{len(cv['pages_without_nodes'])} 页没有任何节点："
                     f"{cv['pages_without_nodes'][:15]}")
    return "\n".join(o)


def _targets(args) -> list[tuple[str, "str | None"]]:
    """返回 [(nodes 路径, pdf 路径或 None)]。不给 --nodes 就读 manifest.json。"""
    if args.nodes:
        return [(args.nodes, args.pdf)]
    manifest = ROOT / "output/manifest.json"
    if not manifest.exists():
        raise SystemExit("既没给 --nodes，也找不到 output/manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out = []
    for s in data.get("standards", []):
        pdf = ROOT / s["source_pdf"] if s.get("source_pdf") else None
        out.append((str(ROOT / s["nodes"]),
                    str(pdf) if pdf and pdf.exists() else None))
    return out


def main():
    ap = argparse.ArgumentParser(description="解析质量评估（默认评估 manifest 里的全部规范）")
    ap.add_argument("--nodes", help="nodes.jsonl 路径；不传则读 output/manifest.json")
    ap.add_argument("--pdf", help="原 PDF 路径，给了就做原文条号对账")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是可读报告")
    ap.add_argument("--out", help="把报告写到文件")
    args = ap.parse_args()

    chunks = []
    for nodes_path, pdf_path in _targets(args):
        r = assess(nodes_path, pdf=pdf_path)
        chunks.append(json.dumps(r, ensure_ascii=False, indent=2) if args.json
                      else render(r))
    text = "\n\n---\n\n".join(chunks)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
