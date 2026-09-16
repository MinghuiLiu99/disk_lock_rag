# -*- coding: utf-8 -*-
"""
跑一遍解析器：测试件 → 节点 + 裁图 + 抽检报告。

    python run_parse.py                # 解析测试件
    python run_parse.py --pdf 盘扣规范/jgj 231-2021.pdf --offset 0 --doc jgj231_2021_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core import DocumentParser, validate_all, to_jsonl, summary


def ascii_table(rows: list[list[str]]) -> str:
    """
    把表格渲染成终端可读的网格（中文按 2 列宽计算），方便肉眼看行列对不对。
    只在抽检报告里用——检索和喂 LLM 仍然用 Markdown 版。
    """
    import unicodedata

    def width(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

    if not rows:
        return ""
    n = max(len(r) for r in rows)
    grid = [[("" if c is None else str(c)).replace("\n", " ") for c in r] + [""] * (n - len(r))
            for r in rows]
    w = [max(width(r[i]) for r in grid) for i in range(n)]
    line = "+" + "+".join("-" * (w[i] + 2) for i in range(n)) + "+"
    out = [line]
    for r in grid:
        out.append("| " + " | ".join(
            cell + " " * (w[i] - width(cell)) for i, cell in enumerate(r)) + " |")
        out.append(line)
    return "\n".join(out)


def render_report(nodes: list[dict], check: dict) -> str:
    out = ["# 解析抽检报告（字段规范 v1.1）", ""]
    out += ["## 产物总览", "", "```json",
            json.dumps(summary(nodes), ensure_ascii=False, indent=2), "```", ""]
    out += ["## 校验结果", ""]
    if check["bad_count"] == 0:
        out.append(f"✅ {check['total']} 个节点全部通过校验")
    else:
        out.append(f"❌ {check['bad_count']}/{check['total']} 个节点有问题：")
        for nid, probs in check["bad"]:
            out.append(f"- `{nid}`: {'; '.join(probs)}")
    out.append("")

    out += ["## 表格抽检（请重点核对这 4 张）", ""]
    for n in nodes:
        if n["type"] != "table":
            continue
        b = n["body"]
        out += [f"### {b['title']}",
                f"- node_id: `{n['node_id']}`　页码: {n['pages']}　status: {n['status']}",
                f"- 表头行数: {b['header_rows']}　纵向合并已填充: {b['fill_down']}　单位: {b['units']}",
                f"- 扁平表头: {b.get('header_flat') or '（单层，与原表一致）'}",
                f"- 表注: {b['notes'] or '（无）'}", "",
                "**原表（保真）**", "", b["markdown"], "",
                "**扁平表头版（喂 LLM / 建行级索引用这个）**", "",
                b.get("markdown_flat") or "（同原表）", "",
                "**终端可读版（肉眼核对行列用）**", "", "```",
                ascii_table(b["rows"]), "```", "",
                "**检索代理文本**", "", "```", n["content"], "```", ""]

    out += ["## 图抽检", ""]
    for n in nodes:
        if n["type"] != "figure":
            continue
        b = n["body"]
        out += [f"- `{n['node_id']}`　{b['title']}",
                f"  - 页码 {n['pages']}　子图 {b['part_count']} 个　矢量图: {b['is_vector']}",
                f"  - 图注: {b['legend']}",
                f"  - 裁图: `{b['image_path']}`", ""]

    out += ["## 公式清单（20 个）", ""]
    for n in nodes:
        if n["type"] != "formula":
            continue
        b = n["body"]
        vars_txt = "；".join(f"{v['symbol']}={v['meaning'][:24]}" for v in b["variables"])
        out += [f"- `{n['node_id']}`（p{n['pages'][0]}）变量 {len(b['variables'])} 个",
                f"  - 线性文本: `{(b['linear_text'] or '').replace(chr(10), ' ')[:90]}`",
                f"  - {vars_txt or '（未抽到变量）'}", ""]

    out += ["## 条文与款（供边界核对）", ""]
    for n in nodes:
        if n["type"] not in ("chapter", "section", "clause", "item"):
            continue
        head = n["content"].split("\n")[0][:70]
        out.append(f"- `{n['node_id']}` level={n['level']} p{n['pages']} "
                   f"modality={n['modality']} refs={len(n['refs'])} | {head}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="jgj 231-2021_test.pdf")
    ap.add_argument("--out", default="out")
    ap.add_argument("--standard-id", default="JGJ231")
    ap.add_argument("--standard-code", default="JGJ/T 231-2021")
    ap.add_argument("--doc-id", default="jgj231_2021_ch5_test")
    ap.add_argument("--version", default="2021")
    ap.add_argument("--offset", type=int, default=12,
                    help="文件第 1 页对应的原书页码偏移（测试件首页码为 13 → 12）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    parser = DocumentParser(args.pdf, args.standard_id, args.standard_code,
                            args.doc_id, args.version, page_offset=args.offset,
                            figure_dir=fig_dir, verbose=True)
    nodes = parser.parse()
    check = validate_all(nodes)
    n = to_jsonl(nodes, out_dir / "nodes.jsonl")
    report = render_report(nodes, check)
    (out_dir / "check_report.md").write_text(report, encoding="utf-8")

    print(f"节点 {n} 个 → {out_dir / 'nodes.jsonl'}")
    print(f"校验: {check['bad_count']} 个问题")
    for nid, probs in check["bad"][:10]:
        print(f"  - {nid}: {'; '.join(probs)}")
    print(json.dumps(summary(nodes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
