# -*- coding: utf-8 -*-
"""
索引层：关键词、BM25、向量、表格库、引用图、混合检索。

设计取舍（演示阶段）：
  · 向量用 numpy 暴力余弦——72 个节点下比向量库更快、更透明、可调试；
    语料上千条后再换 chromadb（接口已隔离在 VectorIndex 内）。
  · 中文分词用"结构标识符 + 汉字二元组"混合切分，零依赖；
    装了 jieba 会自动改用它（见 tokenize 的 USE_JIEBA 开关）。
  · embedding 走 LM Studio 的 OpenAI 兼容接口（bge-m3），带磁盘缓存；
    接口不可用时用确定性哈希向量降级，仅用于验证管道，不用于评估。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schema import node_id_of

try:      # jieba 可选：装了就用，没装不影响
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


# --------------------------------------------------------------- 分词

# 结构标识符优先抽出：条号 / 表号 / 图号 / 式号 / 数值+单位 / 英文与希腊符号
_STRUCT_PATTERNS = (
    re.compile(r"\d+\.\d+\.\d+(?:-\d+)?"),                     # 5.3.3-1
    re.compile(r"\d+\.\d+(?!\d)"),                             # 5.3
    re.compile(r"[表图]\s?[A-Z]?\.?\d+(?:\.\d+)*(-\d+)?"),      # 表5.1.9 / 图B.2
    re.compile(r"[A-Za-z]+"),                                   # Ledger / kN
    re.compile(r"[\u0370-\u03ff][A-Za-z0-9\u0370-\u03ff]*"),    # φ βH γG
    re.compile(r"\d+(?:\.\d+)?\s?(?:mm|cm|m2|m3|kN|kPa|MPa|kg|%|°)"),  # 550mm
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """
    混合分词：先把结构标识符整段抽出（条号、表号、数值+单位、英文、希腊字母），
    剩下的中文按二元组切。

    这么做是因为本语料里最关键的检索信号恰恰是标识符和数值——"650mm""表5.1.9""5.3.3"
    被分词器切碎就再也精确匹配不上了。二元组对中文短词的匹配效果接近分词器。
    """
    if not text:
        return []
    text = text.strip()
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for pat in _STRUCT_PATTERNS:
        for m in pat.finditer(text):
            tokens.append(re.sub(r"\s+", "", m.group(0)).lower())
            spans.append((m.start(), m.end()))
    masked = list(text)
    for a, b in spans:                 # 遮掉已抽出的部分，避免重复切
        for i in range(a, b):
            masked[i] = " "
    rest = "".join(masked)
    for seg in _CJK_RE.findall(rest):
        if _HAS_JIEBA:
            tokens += [w for w in jieba.lcut(seg) if w.strip()]
        elif len(seg) == 1:
            tokens.append(seg)
        else:
            tokens += [seg[i:i + 2] for i in range(len(seg) - 1)]
    return tokens


# --------------------------------------------------------------- BM25

class BM25Index:
    """
    BM25（k1=1.5, b=0.75）。语料小，直接内存里算。
    keywords 会被加权重复，因为条号/表号这类结构关键词是精确匹配的主力。
    """

    def __init__(self, docs: Sequence[tuple[str, str, str]], k1: float = 1.5, b: float = 0.75):
        """docs = [(doc_id, content, keywords_text)]"""
        self.ids = [d[0] for d in docs]
        self.tokens = []
        for _, content, keywords in docs:
            bag = tokenize(keywords) * 3 + tokenize(content)
            self.tokens.append(bag)
        self.freqs = [Counter(t) for t in self.tokens]
        self.lens = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter()
        for f in self.freqs:
            df.update(f.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = []
        for i, f in enumerate(self.freqs):
            dl = self.lens[i] or 1
            s = 0.0
            for t in q:
                tf = f.get(t)
                if not tf:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# --------------------------------------------------------------- 向量

class Embedder:
    """LM Studio 的 OpenAI 兼容 embedding，带磁盘缓存（按文本哈希去重）。"""

    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "text-embedding-bge-m3",
                 cache_path: "str | Path | None" = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=5):
                return True
        except Exception:
            return False

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in texts if self._key(t) not in self.cache]
        for i in range(0, len(todo), 16):          # 分批，避免单次请求过大
            batch = todo[i:i + 16]
            body = json.dumps({"model": self.model, "input": list(batch)}).encode()
            req = urllib.request.Request(self.base_url + "/embeddings", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"embedding 服务不可用（{self.base_url}）：{e}。"
                    "请启动 LM Studio 并加载 bge-m3，或使用 HashingEmbedder 做管道验证。") from e
            for text, item in zip(batch, data["data"]):
                self.cache[self._key(text)] = item["embedding"]
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False),
                                       encoding="utf-8")
        return np.array([self.cache[self._key(t)] for t in texts], dtype=np.float32)


class HashingEmbedder:
    """
    确定性哈希向量（字符二元组 → 512 维）。**只用于验证管道连通性**，不能用于评估——
    它没有语义能力，只保证"字面相似 → 向量相近"。
    """

    dim = 512

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    def available(self) -> bool:
        return True


class VectorIndex:
    """余弦相似度暴力检索。语料小的时候比向量库更快、更透明。"""

    def __init__(self, ids: Sequence[str], matrix: np.ndarray):
        self.ids = list(ids)
        m = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m / np.where(norms == 0, 1.0, norms)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        # 注意：本机 numpy 2.5.3 的 BLAS 在 2D@1D 上会原生崩溃（0xc06d007f），
        # 逐元素乘 + 求和是等价写法且稳定。语料上千条前性能差异可忽略。
        sims = (self.matrix * q).sum(axis=1)
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]


# --------------------------------------------------------------- 表格库

class TableStore:
    """
    表格结构化库（SQLite）。让"2步3跨布置时双排架系数是多少"这类问题走精确取值，
    而不是靠向量碰运气。每行存成 列名=值 的 JSON，便于按单元格过滤。
    """

    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        # check_same_thread=False：建索引在 FastAPI 的同步线程池里执行，
        # 而查询发生在异步事件循环线程——不关掉线程检查会抛
        # "SQLite objects created in a thread can only be used in that same thread"。
        # 本类只做"建完后只读查询"，且单进程单请求，关掉是安全的。
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE table_rows (
                node_id TEXT, table_id TEXT, title TEXT, page INTEGER,
                row_index INTEGER, row_text TEXT, cells TEXT)""")
        self.conn.execute("""CREATE TABLE tables (
                node_id TEXT PRIMARY KEY, table_id TEXT, title TEXT,
                header_flat TEXT, notes TEXT, pages TEXT, markdown_flat TEXT)""")

    def add_table(self, node: dict) -> None:
        b = node["body"]
        rows = b["rows"]
        header = b.get("header_flat") or (
            [(c or "").strip() for c in rows[0]] if rows else [])
        data = rows[b["header_rows"]:]
        self.conn.execute("INSERT OR REPLACE INTO tables VALUES (?,?,?,?,?,?,?)", (
            node["node_id"], b["table_id"], b["title"],
            json.dumps(header, ensure_ascii=False), json.dumps(b["notes"], ensure_ascii=False),
            json.dumps(node["pages"]), b.get("markdown_flat") or b.get("markdown")))
        for i, row in enumerate(data):
            cells = {header[j]: row[j] for j in range(min(len(header), len(row))) if row[j]}
            text = "；".join(f"{k}={v}" for k, v in cells.items())
            self.conn.execute("INSERT INTO table_rows VALUES (?,?,?,?,?,?,?)", (
                node["node_id"], b["table_id"], b["title"], node["pages"][0], i,
                text, json.dumps(cells, ensure_ascii=False)))
        self.conn.commit()

    def find_rows(self, phrase: str, top_k: int = 5) -> list[dict]:
        """按行文本做朴素匹配（演示够用；换成 FTS5 或接 BM25 都很容易）。"""
        toks = [t for t in tokenize(phrase) if len(t) > 1]
        cur = self.conn.execute(
            "SELECT node_id, table_id, title, row_index, row_text FROM table_rows")
        scored = []
        for nid, tid, title, idx, text in cur.fetchall():
            s = sum(text.count(t) for t in toks)
            if s:
                scored.append((s, {"node_id": nid, "table_id": tid, "title": title,
                                   "row_index": idx, "row_text": text}))
        scored.sort(key=lambda x: -x[0])
        return [x[1] for x in scored[:top_k]]


# --------------------------------------------------------------- 引用图

class RefGraph:
    """
    关系图：节点=node_id。两种边——
      · refs 出边（条文 → 它引用的表/图/式/附录），类型为目标类型
      · contains 边（父子：条 → 它自己的表/图/公式/款）
    闭包扩展必须两种都走：问"立杆稳定性怎么验算"，需要的不只是该条引用的附录，
    还有它自带的公式（那是父子关系，不是引用关系）。
    """

    def __init__(self, nodes: Sequence[dict]):
        import networkx as nx
        self.nodes = {n["node_id"]: n for n in nodes}
        self.g = nx.DiGraph()
        self.g.add_nodes_from(self.nodes)
        for n in nodes:
            for r in n["refs"]:
                if r["in_scope"] and r["target_node_id"] in self.nodes:
                    self.g.add_edge(n["node_id"], r["target_node_id"], type=r["target_type"])
            if n["parent_id"] in self.nodes:
                self.g.add_edge(n["parent_id"], n["node_id"], type="contains")

    def neighbors(self, node_id: str, hops: int = 1, successors_only: bool = False,
                  kinds: "set[str] | None" = None) -> list[str]:
        """
        successors_only=True 时只向下走（引用边 + 父子边）。
        闭包扩展必须这样——否则每条条文都会把它的"节""章"父节点拉进来，
        上下文预算全被目录型节点吃掉，对回答毫无帮助。
        kinds 可限定只走某类边（如 {"contains"} 只走父子、排除 "contains" 只走引用）。
        """
        import networkx as nx
        if node_id not in self.g:
            return []
        seen = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for _, tgt, data in self.g.out_edges(n, data=True):
                    if kinds and data.get("type") not in kinds:
                        continue
                    nxt.add(tgt)
                if not successors_only:
                    for src, _, data in self.g.in_edges(n, data=True):
                        if kinds and data.get("type") not in kinds:
                            continue
                        nxt.add(src)
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return [n for n in seen if n != node_id]

    def stats(self) -> dict:
        return {"节点": self.g.number_of_nodes(), "边": self.g.number_of_edges()}


# --------------------------------------------------------------- 混合检索

# 进向量库的类型：款(item) 的文本与所属条文重复，进向量只会挤占召回位，故排除
VECTOR_TYPES = ("chapter", "section", "clause", "table", "figure", "formula")

# 图里除 "contains"（父子）之外的所有边类型，即"引用型"边。
# 闭包扩展要先走这一组：跨章依赖（5.3.3 → 附录C）才是闭包的价值所在。
REF_KINDS = {"clause", "table", "figure", "formula", "appendix", "external_standard"}

# 查询里的显式标识符：命中即直接加权。编号类查询靠向量必翻车（"5.4.2 说了什么"
# 会被语义相近的 5.4.3 抢走），所以这一路必须是硬规则，不能交给相似度。
_TYPED_MENTIONS = (
    (re.compile(r"表\s?([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)"), "table"),
    (re.compile(r"图\s?(\d+(?:\.\d+)*(?:-\d+)?)"), "figure"),
    (re.compile(r"式\s?[（(]\s*([\d.]+(?:-\d+)?)\s*[）)]"), "formula"),
    (re.compile(r"附录\s?([A-Z])"), "appendix"),
)
_BARE_CLAUSE = re.compile(r"(?<![\d.])(\d+\.\d+\.\d+(?:-\d+)?)")


def mentioned_ids(query: str, standard_id: str) -> set[str]:
    """
    从查询里抽出被显式提到的节点 id（"表5.1.9""5.4.2""图5.1.4"）。

    注意先带类型再裸编号，且带类型的匹配要被遮掉：
    "表5.1.9" 里的 5.1.9 是表号，不是条号——不遮的话条文 5.1.9 也会被加权，
    结果条文把表挤到第二（实测踩过）。
    """
    out: set[str] = set()
    claimed: list[tuple[int, int]] = []
    for pattern, node_type in _TYPED_MENTIONS:
        for m in pattern.finditer(query):
            out.add(node_id_of(standard_id, node_type, re.sub(r"\s+", "", m.group(1))))
            claimed.append((m.start(), m.end()))
    for m in _BARE_CLAUSE.finditer(query):
        if any(a <= m.start() < b for a, b in claimed):
            continue
        out.add(node_id_of(standard_id, "clause", m.group(1)))
    return out


# 领域同义词表：用户口语 ↔ 规范用词。
# 只收"零字面重叠"的词对——有重叠的靠汉字二元组本来就能匹配
# （"底座"能命中"可调底座"），收进来只是噪声。
# 实测触发点："盘扣架的斜撑怎么布置" 因"斜撑"与"竖向斜杆"无共同字符而完全召回失败。
DOMAIN_SYNONYMS = {
    "斜撑": "斜杆 竖向斜杆",
    "剪刀撑": "斜杆 竖向斜杆 水平剪刀撑",
    "架子": "脚手架",
    "外架": "作业架",
    "立柱": "立杆",
    "横杆": "水平杆",
    "圆盘": "连接盘",
    "扣盘": "连接盘",
    "顶托": "可调托撑",
    "顶丝": "可调托撑",
    "细长比": "长细比",
    "承载力": "承载力设计值",
    "挠度": "容许挠度",
}


def expand_query(query: str) -> str:
    """查询侧同义词扩展：把口语词追加成规范用词，让两路召回都能命中。"""
    extra = [standard for k, standard in DOMAIN_SYNONYMS.items() if k in query]
    if not extra:
        return query
    merged = []
    for chunk in extra:                       # 去重，避免同一个词重复加权
        for w in chunk.split():
            if w not in merged:
                merged.append(w)
    return query + " " + " ".join(merged)


class HybridRetriever:
    """
    BM25 + 向量，用 RRF（Reciprocal Rank Fusion）融合后再可选做引用图扩展。

    为什么用 RRF 而不是加权分数：两路分数量纲不同（BM25 无上界、余弦在 0~1），
    归一化会随语料变化；RRF 只看排名，稳定且不需要调参。
    两路的权重通过 alpha 调：alpha=BM25 权重，1-alpha=向量权重。
    """

    def __init__(self, nodes: Sequence[dict], bm25: BM25Index, vec: VectorIndex,
                 embedder, graph: RefGraph):
        self.nodes = {n["node_id"]: n for n in nodes}
        self.bm25 = bm25
        self.vec = vec
        self.embedder = embedder
        self.graph = graph

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5,
               pool: int = 30, expand_hops: int = 0,
               mention_boost: float = 0.05) -> list[dict]:
        # 查询侧做同义词扩展；检索结果里仍然保留原始 query 供展示
        effective = expand_query(query)
        bm = self.bm25.search(effective, top_k=pool)
        qv = self.embedder.embed([effective])[0]
        ve = self.vec.search(qv, top_k=pool)
        k = 60.0
        fused: dict[str, dict] = {}
        for rank, (nid, score) in enumerate(bm, start=1):
            fused.setdefault(nid, {"node_id": nid, "score": 0.0, "bm25_rank": None,
                                   "vec_rank": None, "bm25_score": None, "vec_score": None})
            fused[nid]["score"] += alpha / (k + rank)
            fused[nid]["bm25_rank"], fused[nid]["bm25_score"] = rank, round(score, 3)
        for rank, (nid, score) in enumerate(ve, start=1):
            fused.setdefault(nid, {"node_id": nid, "score": 0.0, "bm25_rank": None,
                                   "vec_rank": None, "bm25_score": None, "vec_score": None})
            fused[nid]["score"] += (1 - alpha) / (k + rank)
            fused[nid]["vec_rank"], fused[nid]["vec_score"] = rank, round(score, 3)

        # 显式标识符加权：查询里点名了表5.1.9 / 5.4.2 / 图5.1.4，就直接把那几个节点顶上来。
        # 这一步放在 RRF 之后，等于"硬规则优先于相似度"。
        mentioned = mentioned_ids(query, self.nodes[next(iter(self.nodes))]["standard_id"]) \
            if self.nodes else set()
        for nid in mentioned:
            if nid not in self.nodes:
                continue
            rec = fused.setdefault(nid, {"node_id": nid, "score": 0.0, "bm25_rank": None,
                                         "vec_rank": None, "bm25_score": None, "vec_score": None})
            rec["score"] += mention_boost
            rec["mentioned"] = True

        ranked = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
        for r in ranked:
            r["type"] = self.nodes[r["node_id"]]["type"]
            r["num"] = self.nodes[r["node_id"]]["num"]
            if effective != query:
                r["expanded_query"] = effective

        if expand_hops:                      # 引用闭包扩展：把命中的表/图/公式/附录带回来
            have = {r["node_id"] for r in ranked}
            extra: dict[str, list[str]] = {}
            # 节/章是枢纽节点，从它们扩展会把整节条文全拉进来，上下文瞬间被淹。
            # 只从"内容型"节点出发：条 / 款 / 表 / 图 / 公式。
            seeds = [r["node_id"] for r in ranked
                     if self.nodes[r["node_id"]]["type"] not in ("section", "chapter")]

            def take(kinds, budget: int) -> None:
                """
                逐种子轮转取（每轮每个种子最多取 1 个），避免单个枢纽把预算吃光。
                实测踩过：5.4.1 一条就带 4 款 + 3 公式 + 1 表，顺序取会把 6 个预算占满，
                真正需要的附录 C（由 5.3.3 引用）反而拿不回来。
                """
                added = 0
                while added < budget:
                    progress = False
                    for s in seeds:
                        if added >= budget:
                            break
                        for nb in self.graph.neighbors(s, hops=expand_hops,
                                                       successors_only=True, kinds=kinds):
                            if nb in have or nb in extra:
                                continue
                            if self.nodes[nb]["type"] in ("section", "chapter"):
                                continue
                            extra[nb] = [s]
                            added += 1
                            progress = True
                            break                      # 每个种子每轮只取一个
                    if not progress:
                        break

            # 第一轮走**跨章引用**（闭包的价值所在），第二轮才补父子（子表/子公式/款）
            take(REF_KINDS, 5)
            take({"contains"}, 4)
            for nid, via in extra.items():
                n = self.nodes[nid]
                ranked.append({"node_id": nid, "type": n["type"], "num": n["num"],
                               "title": (n.get("body") or {}).get("title") if n["body"] else None,
                               "score": 0.0,
                               "bm25_rank": None, "vec_rank": None,
                               "bm25_score": None, "vec_score": None,
                               "via_graph": via})
        return ranked


# --------------------------------------------------------------- 组装

class IndexBundle:
    """一次构建、多处使用：BM25 / 向量 / 表格库 / 引用图 / 混合检索。"""

    def __init__(self, nodes: Sequence[dict], bm25: BM25Index, vector: VectorIndex,
                 tables: TableStore, graph: RefGraph, retriever: HybridRetriever,
                 embedder_name: str):
        self.nodes = list(nodes)
        self.bm25 = bm25
        self.vector = vector
        self.tables = tables
        self.graph = graph
        self.retriever = retriever
        self.embedder_name = embedder_name


def _index_text(node: dict) -> str:
    """进索引的正文。表格用检索代理文本；条文用原文。"""
    return node["content"]


def build_index(nodes: Sequence[dict], out_dir: "str | Path",
                embedder=None) -> IndexBundle:
    """构建全部索引。embedder 传 None 时自动尝试 LM Studio，失败则退化为哈希向量。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        emb = Embedder(cache_path=out_dir / "embeddings.json")
        if emb.available():
            embedder = emb
        else:
            print("⚠ LM Studio embedding 不可用，降级为哈希向量（仅验证管道，不可用于评估）")
            embedder = HashingEmbedder()

    docs = [(n["node_id"], _index_text(n), " ".join(n["keywords"])) for n in nodes]
    bm25 = BM25Index(docs)

    vec_nodes = [n for n in nodes if n["type"] in VECTOR_TYPES]
    matrix = embedder.embed([_index_text(n) for n in vec_nodes])
    vector = VectorIndex([n["node_id"] for n in vec_nodes], matrix)

    tables = TableStore(out_dir / "tables.db")
    for n in nodes:
        if n["type"] == "table":
            tables.add_table(n)

    graph = RefGraph(nodes)
    retriever = HybridRetriever(nodes, bm25, vector, embedder, graph)
    name = getattr(embedder, "model", type(embedder).__name__)
    return IndexBundle(nodes, bm25, vector, tables, graph, retriever, name)
