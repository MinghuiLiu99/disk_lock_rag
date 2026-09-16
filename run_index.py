# -*- coding: utf-8 -*-
"""
构建索引并跑检索验收。

    python run_index.py                 # 用 LM Studio 的 bge-m3
    python run_index.py --hash-embed    # 不依赖本地模型，验证管道
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.index import Embedder, HashingEmbedder, build_index
from rag_core import from_jsonl

# 验收查询：覆盖 数值取值 / 表结构 / 条号精确匹配 / 公式 / 图 / 跨文档引用
QUERIES = [
    ("数值·查表", "双排架2步3跨布置时立杆计算长度系数是多少"),
    ("数值·转置表", "搭设高度24m时支撑架高度调整系数取多少"),
    ("数值·取值", "可调托撑的承载力设计值是多少"),
    ("数值·取值", "受弯构件的容许挠度是多少"),
    ("语义·条文", "立杆稳定性应该怎么验算"),
    ("语义·条文", "连墙件的稳定性怎么计算"),
    ("编号·精确", "5.4.2 说了什么"),
    ("编号·精确", "表5.1.9"),
    ("图形", "图5.1.4 是什么"),
    ("跨文档引用", "钢材的强度设计值去哪里查"),
]


def show(title: str, bundle, query: str, top_k: int = 3) -> None:
    print("=" * 96)
    print(f"[{title}] {query}")
    bm = bundle.bm25.search(query, top_k=top_k)
    qv = bundle.retriever.embedder.embed([query])[0]
    ve = bundle.vector.search(qv, top_k=top_k)
    hy = bundle.retriever.search(query, top_k=top_k, alpha=0.5)
    def fmt(rows, kind):
        out = []
        for nid, sc in rows:
            n = {x["node_id"]: x for x in bundle.nodes}[nid]
            out.append(f"{n['num'] or n['type']}({sc:.2f})")
        return " ".join(out) or "（无）"
    print(f"  BM25 : {fmt(bm, 'b')}")
    print(f"  向量 : {fmt(ve, 'v')}")
    print("  混合 : " + " | ".join(
        f"{r['type'][:4]}:{r['num']}" for r in hy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="out/nodes.jsonl")
    ap.add_argument("--out", default="out/index")
    ap.add_argument("--hash-embed", action="store_true")
    args = ap.parse_args()

    nodes = from_jsonl(args.nodes)
    print(f"载入 {len(nodes)} 个节点")
    if args.hash_embed:
        embedder = HashingEmbedder()
    else:
        embedder = Embedder(cache_path=Path(args.out) / "embeddings.json")
        if not embedder.available():
            print("⚠ LM Studio 不可用，改用哈希向量")
            embedder = HashingEmbedder()

    bundle = build_index(nodes, args.out, embedder=embedder)
    print(f"索引完成 | embedding={bundle.embedder_name} | 引用图 {bundle.graph.stats()}")
    print(f"向量库节点数 {len(bundle.vector.ids)}（款/章标题不进向量）")
    print()

    for title, q in QUERIES:
        show(title, bundle, q)
    print()
    print("=" * 96)
    print("[引用闭包扩展] 立杆稳定性应该怎么验算  +1 跳")
    for r in bundle.retriever.search("立杆稳定性应该怎么验算", top_k=3, expand_hops=1):
        via = f"  ← 由 {r['via_graph']} 带出" if r.get("via_graph") else ""
        print(f"   {r['type']:9s} {str(r['num']):10s} {r['node_id']}{via}")
    print()
    print("[表结构化查询] 2步3跨 双排架")
    for row in bundle.tables.find_rows("2步3跨 双排架"):
        print(f"   {row['title']} 第{row['row_index']+1}行: {row['row_text']}")


if __name__ == "__main__":
    main()
