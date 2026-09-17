# -*- coding: utf-8 -*-
"""
盘扣架规范问答 Demo 前端。

    python app.py                                   # 默认：7 页测试件
    python app.py --profile full                    # 全量 57 页
    python app.py --nodes out_full/nodes.jsonl --pdf "盘扣规范/jgj 231-2021.pdf" --offset 0

三个接口：
    POST /api/ask        流式问答（SSE：meta → delta… → done）
    GET  /api/page/{n}   渲染 PDF 页并高亮指定坐标（book 页码）
    GET  /api/health     健康检查
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from rag_core import from_jsonl
from rag_core.generate import Answerer
from rag_core.index import Embedder, build_index

ROOT = Path(__file__).resolve().parent

# 两套可切换的数据：测试件（7 页，跑得快）与全量（57 页）
PROFILES = {
    "test": {"nodes": "out/nodes.jsonl", "index": "out/index",
             "pdf": "jgj 231-2021_test.pdf", "offset": 12,
             "title": "JGJ/T 231-2021《第 5 章 结构设计》",
             "examples": ["双排架2步3跨布置时立杆计算长度系数是多少",
                          "可调托撑的承载力设计值是多少",
                          "搭设高度24m时支撑架高度调整系数取多少",
                          "立杆稳定性应该怎么验算",
                          "连墙件的稳定性怎么计算",
                          "钢材的强度设计值去哪里查",
                          "5.4.2 说了什么",
                          "图5.1.4 是什么",
                          "盘扣架立杆的颜色有什么要求"]},
    "full": {"nodes": "out_full/nodes.jsonl", "index": "out_full/index",
             "pdf": "盘扣规范/jgj 231-2021.pdf", "offset": 0,
             "title": "JGJ/T 231-2021《全本 57 页》",
             "examples": ["插销安装后下沉量不应大于多少",
                          "扫地杆距可调底座底板不应大于多少",
                          "立杆稳定性应该怎么验算",
                          "长细比100的Q235钢管，稳定系数φ取多少",
                          "可调托撑的承载力设计值是多少",
                          "脚手架搭设高度超过24m时有哪些要求",
                          "作业架连墙件的设置间距有什么要求",
                          "拆除脚手架时应注意哪些安全要求",
                          "2步3跨布置时双排架的计算长度系数",
                          "盘扣架立杆的颜色有什么要求"]},
}

# 运行时配置（由 main 按 profile 填充）
CFG = {"nodes": ROOT / "out/nodes.jsonl", "index": ROOT / "out/index",
       "pdf": ROOT / "jgj 231-2021_test.pdf", "offset": 12,
       "title": PROFILES["test"]["title"], "examples": PROFILES["test"]["examples"]}
PDF_PAGE_OFFSET = 12

STATE: dict = {}
app = FastAPI(title="盘扣架规范问答 Demo")


def boot() -> None:
    if STATE:
        return
    nodes = from_jsonl(CFG["nodes"])
    bundle = build_index(nodes, CFG["index"],
                         embedder=Embedder(cache_path=CFG["index"] / "embeddings.json"))
    STATE["bundle"] = bundle
    STATE["answerer"] = Answerer(bundle)
    STATE["nodes"] = {n["node_id"]: n for n in nodes}
    STATE["pdf"] = None


def _pdf():
    """惰性打开 PDF（pypdfium2），用于高亮渲染。"""
    if STATE.get("pdf") is None:
        import pypdfium2 as pdfium
        STATE["pdf"] = pdfium.PdfDocument(str(CFG["pdf"]))
    return STATE["pdf"]


_COUNT: dict = {}


def _node_count() -> int:
    """页面上显示节点数。直接数 JSONL 行数，避免为了显示一个数字去建索引。"""
    key = str(CFG["nodes"])
    if key not in _COUNT:
        _COUNT[key] = sum(1 for line in CFG["nodes"].open(encoding="utf-8") if line.strip())
    return _COUNT[key]


def render_page(book_page: int, boxes: list[list[float]], dpi: int = 150,
                band_pad: float = 90.0) -> bytes:
    """
    渲染原书某页并把给定坐标框画成高亮。坐标是 pdfplumber 约定（top 距上沿）。

    有坐标时**只裁高亮附近的一条横带**（上下各留 90pt 上下文）：
    整页图在 150dpi 下有 ~1750px 高，而证据卡片只有 ~300px 可视高度，
    直接放整页会导致高亮位置在可视区外、看起来是空白（实测踩过）。
    """
    from PIL import ImageDraw
    doc = _pdf()
    idx = book_page - PDF_PAGE_OFFSET - 1
    if idx < 0 or idx >= len(doc):
        raise ValueError(f"页码 {book_page} 超出测试件范围")
    img = doc[idx].render(scale=dpi / 72).to_pil().convert("RGB")
    scale = dpi / 72.0
    draw = ImageDraw.Draw(img, "RGBA")
    for b in boxes:
        x0, top, x1, bottom = [float(v) * scale for v in b]
        draw.rectangle([x0, top, x1, bottom], fill=(255, 214, 0, 70),
                       outline=(230, 120, 0, 255), width=3)
    if boxes:
        y0 = max(0.0, min(float(b[1]) for b in boxes) * scale - band_pad * scale)
        y1 = min(float(img.height), max(float(b[3]) for b in boxes) * scale + band_pad * scale)
        if y1 - y0 > 40:
            img = img.crop((0, int(y0), img.width, int(y1)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/api/health")
def health():
    boot()
    b = STATE["bundle"]
    return {"nodes": len(b.nodes), "graph": b.graph.stats(),
            "embedder": b.embedder_name, "model": STATE["answerer"].model,
            "profile": CFG["title"], "pdf": str(CFG["pdf"].relative_to(ROOT))}


@app.get("/api/page/{book_page}")
def page_image(book_page: int, boxes: str = ""):
    boot()
    parsed: list[list[float]] = []
    if boxes:
        for chunk in boxes.split(";"):
            if chunk.strip():
                parsed.append([float(v) for v in chunk.split(",")])
    try:
        return Response(render_page(book_page, parsed), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/node/{node_id:path}")
def node_detail(node_id: str):
    """节点详情：正文材料块 + 可高亮的页码坐标 + 图/表附件。"""
    boot()
    n = STATE["nodes"].get(node_id)
    if not n:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"node_id": n["node_id"], "type": n["type"], "num": n["num"],
            "path": n["path"], "pages": n["pages"], "bboxes": n["bboxes"],
            "content": n["content"], "body": n["body"],
            "modality": n["modality"], "refs": n["refs"],
            "explained_by": n["explained_by"], "status": n["status"]}


@app.get("/api/trace")
def trace(q: str = "", top_k: int = 6):
    """只做检索、不生成：用来演示混合检索与闭包扩展的过程。"""
    boot()
    if not q:
        return {"error": "empty query"}
    hits = STATE["bundle"].retriever.search(q, top_k=top_k, expand_hops=1)
    return {"question": q, "hits": [
        {"node_id": h["node_id"], "type": h["type"], "num": h["num"],
         "score": round(h.get("score", 0.0), 4),
         "via_graph": h.get("via_graph")} for h in hits]}


@app.post("/api/ask")
async def ask(payload: dict):
    boot()
    question = (payload or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    ans = STATE["answerer"]
    # 依据材料数量可由前端调（默认 6）：材料越多答案越全，但首字越慢——
    # 本地 27B 要先吞完所有材料才吐第一个字。
    prepared = ans.prepare(question,
                           top_k=int((payload or {}).get("top_k", 10)),
                           max_blocks=int((payload or {}).get("max_blocks", 10)))

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        yield sse("meta", {"question": question, "model": prepared["model"],
                           "contexts": [{k: c[k] for k in
                                         ("node_id", "type", "num", "label", "pages", "bboxes", "image_path")}
                                        for c in prepared["contexts"]],
                           "hits": prepared["hits"],
                           "table_rows": prepared["table_rows"]})
        buf = []
        reasoning = 0
        last_err = None
        for attempt in (1, 2):          # LM Studio 偶发 400/断流（实测一次），重试一次即可
            buf.clear()
            reasoning = 0
            try:
                for kind, delta in ans.stream(prepared["messages"]):
                    if kind == "reasoning":
                        reasoning += len(delta)
                        yield sse("reasoning", {"text": delta})
                        continue
                    buf.append(delta)
                    yield sse("delta", {"text": delta})
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt == 1:
                    yield sse("retry", {"message": f"生成中断，正在重试：{e}"})
                    continue
        if last_err is not None:
            yield sse("error", {"message": str(last_err)})
            return
        final = ans.finalize(prepared, "".join(buf))
        yield sse("done", {
            "answer": final["answer"],
            "reasoning_chars": reasoning,
            "citations": final["citations"],
            "secondary_citations": final["secondary_citations"],
            "invalid_citations": final["invalid_citations"],
            "cited_nodes": [c["node_id"] for c in prepared["contexts"]
                            if any(x["type"] == c["type"] and str(x["num"]) == str(c["num"])
                                   for x in final["citations"] + final["secondary_citations"])],
            "refused": final["refused"],
            "citation_ok": final["citation_ok"],
            "has_citation": final["has_citation"],
        })

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def index():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    return (html.replace("<!--PROFILE-->", CFG["title"])
                .replace("<!--NODES-->", str(_node_count()))
                .replace("<!--EXAMPLES-->", json.dumps(CFG["examples"], ensure_ascii=False)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="test")
    ap.add_argument("--nodes")
    ap.add_argument("--pdf")
    ap.add_argument("--offset", type=int)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    CFG.update(nodes=ROOT / (args.nodes or prof["nodes"]),
               index=ROOT / (prof["index"] if args.nodes is None else Path(args.nodes).parent),
               pdf=ROOT / (args.pdf or prof["pdf"]),
               offset=args.offset if args.offset is not None else prof["offset"],
               title=prof["title"], examples=prof["examples"])
    PDF_PAGE_OFFSET = CFG["offset"]
    print(f"数据源: {CFG['nodes'].relative_to(ROOT)} | PDF: {CFG['pdf'].relative_to(ROOT)} "
          f"| 页码偏移 {CFG['offset']} | {CFG['title']}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
