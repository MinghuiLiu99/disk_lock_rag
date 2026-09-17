# -*- coding: utf-8 -*-
"""
文献核查（第 2 轮，精确检索）：查"盘扣 + RAG + 多模态文档解析"有没有人做。

第 1 轮用 OpenAlex 的 search= 做全文字匹配，命中数虚高（几十万条不相关）。
第 2 轮改用 filter=title_and_abstract.search: 做标题+摘要级精确匹配，并限定 2023 年以后。

数据源：OpenAlex、arXiv（CS 预印本）、Crossref

用途：写论文的 related work / 研究空白论证。跑一次就能拿到"这个方向有没有人做、
做到什么程度"的可复现证据，比手写一段"相关研究较少"要硬得多。

    python run_lit_check.py

OpenAlex 不需要任何配置。arXiv 那几组查询依赖本机的 paper-lookup skill 里的
Atom→JSON 小工具；找不到时只跳过 arXiv，OpenAlex 部分照常输出。
"""
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAILTO = "lit-check@example.org"
ARXIV_PARSER = Path(r"C:\Users\Aixko\.codex\skills\paper-lookup\scripts\arxiv_atom.py")
UA = {"User-Agent": "lit-check/1.0 (mailto:lit-check@example.org)"}
SELECT = "id,doi,title,publication_year,cited_by_count,primary_location,type"


def get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def openalex(q, n=8, since="2023-01-01"):
    """title+abstract 级精确检索；q 里的空格会被保留，用 + 连接多词短语。"""
    filt = f'title_and_abstract.search:{q}'
    if since:
        filt += f",from_publication_date:{since}"
    url = ("https://api.openalex.org/works?filter=" + urllib.parse.quote(filt, safe=":.,\"")
           + f"&per_page={n}&sort=relevance_score:desc&select={SELECT}&mailto={MAILTO}")
    d = json.loads(get(url))
    rows = []
    for w in d.get("results", []):
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "-"
        rows.append({"title": (w.get("title") or "").strip(), "year": w.get("publication_year"),
                     "venue": src, "cited": w.get("cited_by_count"), "type": w.get("type"),
                     "doi": (w.get("doi") or "").replace("https://doi.org/", "")})
    return d["meta"]["count"], rows


def arxiv(q, n=10):
    if not ARXIV_PARSER.exists():
        raise RuntimeError(f"缺少 arXiv 解析脚本 {ARXIV_PARSER}——跳过 arXiv，OpenAlex 不受影响")
    url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + f"&start=0&max_results={n}&sortBy=relevance")
    xml = get(url, timeout=60)
    p = subprocess.run([sys.executable, ARXIV_PARSER, "-"], input=xml,
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"arxiv_atom.py exit={p.returncode}: {(p.stderr or p.stdout)[:200]}")
    d = json.loads(p.stdout)
    rows = [{"title": (e.get("title") or "").replace("\n", " ").strip(),
             "year": (e.get("published") or "")[:4],
             "arxiv_id": e.get("arxiv_id"),
             "cats": ",".join(e.get("categories") or [])[:40]}
            for e in d.get("entries", [])]
    return d.get("total_results"), rows


def show(tag, query, total, rows, source, note=""):
    print("=" * 100)
    print(f"[{tag}] ({source}) {query}")
    print(f"    命中总数: {total}  {note}")
    for r in rows:
        extra = ""
        if r.get("cited") is not None:
            extra += f" | 被引{r['cited']}"
        if r.get("type"):
            extra += f" | {r['type']}"
        if r.get("doi"):
            extra += f" | doi:{r['doi']}"
        if r.get("arxiv_id"):
            extra += f" | arXiv:{r['arxiv_id']} | {r.get('cats','')}"
        print(f"    - [{r.get('year')}] {r.get('title','')[:115]}")
        print(f"      {(r.get('venue') or '')[:72]}{extra}")
    print()


OPENALEX_QUERIES = [
    ("B1", '"building code" "large language model"'),
    ("B2", '"code compliance checking" "large language model"'),
    ("B3", '"retrieval augmented generation" "construction"'),
    ("B4", '"retrieval augmented generation" "engineering standard"'),
    ("B5", '"construction safety" "question answering"'),
    ("B6", '"scaffold" "large language model"'),
    ("B7", '"table structure recognition"'),
    ("B8", '"cross-page table"'),
    ("B9", '"PDF parsing" "retrieval augmented generation"'),
    ("B10", '"document structure" "chunking"'),
    ("B11", '"knowledge graph" "construction specification"'),
    ("B12", '"multi-hop retrieval" "regulation"'),
    ("B13", '"Chinese standard" "large language model"'),
    ("B14", '"hierarchical structure" "retrieval augmented generation"'),
    ("B15", '"provision" "segmentation" "standard" "retrieval"'),
]

ARXIV_QUERIES = [
    ("Y1", 'ti:"document parsing" AND abs:"benchmark"'),
    ("Y2", 'abs:"chunking" AND abs:"retrieval-augmented generation"'),
    ("Y3", 'abs:"table" AND abs:"retrieval-augmented generation" AND abs:"parsing"'),
    ("Y4", 'abs:"citation" AND abs:"retrieval-augmented generation"'),
    ("Y5", 'abs:"building code" OR abs:"engineering standard"'),
    ("Y6", 'abs:"multi-hop" AND abs:"retrieval" AND abs:"graph"'),
]


def main():
    print("#" * 100)
    print("# OpenAlex —— title+abstract 精确匹配，限定 2023-01-01 之后")
    print("#" * 100)
    for tag, q in OPENALEX_QUERIES:
        try:
            total, rows = openalex(q)
            show(tag, q, total, rows, "OpenAlex")
        except Exception as e:
            print(f"[{tag}] {q}\n    OpenAlex 失败: {type(e).__name__}: {e}\n")
        time.sleep(0.4)

    print("#" * 100)
    print("# arXiv（限速 1 请求 / 3 秒）")
    print("#" * 100)
    for i, (tag, q) in enumerate(ARXIV_QUERIES):
        try:
            if i:
                time.sleep(3.3)
            total, rows = arxiv(q)
            show(tag, q, total, rows, "arXiv")
        except Exception as e:
            print(f"[{tag}] {q}\n    arXiv 失败: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
