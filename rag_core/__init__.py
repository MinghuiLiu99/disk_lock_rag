# -*- coding: utf-8 -*-
"""盘扣架规范 RAG —— 结构感知解析器（字段规范 v1.1）。"""
from .schema import (
    NODE_FIELDS, TYPE_LEVEL, REF_TYPES, MODALITY_LEVEL, STATUS_REASONS,
    NodeBuilder, validate_node, validate_all, extract_refs, detect_modality,
    link_structure, to_jsonl, from_jsonl, summary,
)
from .parser import DocumentParser
from .index import (BM25Index, Embedder, HashingEmbedder, HybridRetriever,
                    IndexBundle, RefGraph, TableStore, VectorIndex, build_index,
                    tokenize)

__all__ = [
    "DocumentParser", "NodeBuilder", "NODE_FIELDS", "TYPE_LEVEL", "REF_TYPES",
    "MODALITY_LEVEL", "STATUS_REASONS", "validate_node", "validate_all",
    "extract_refs", "detect_modality", "link_structure",
    "to_jsonl", "from_jsonl", "summary",
    "BM25Index", "Embedder", "HashingEmbedder", "HybridRetriever", "IndexBundle",
    "RefGraph", "TableStore", "VectorIndex", "build_index", "tokenize",
]
