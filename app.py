# -*- coding: utf-8 -*-
"""
盘扣架规范问答 Demo 前端。

    python app.py          然后打开 http://127.0.0.1:8000

三个接口：
    POST /api/ask        流式问答（SSE：meta → delta… → done）
    GET  /api/page/{n}   渲染 PDF 页并高亮指定坐标（book 页码）
    GET  /api/health     健康检查
"""
from __future__ import annotations

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
PDF = ROOT / "jgj 231-2021_test.pdf"
PDF_PAGE_OFFSET = 12          # 文件第 1 页 = 原书第 13 页

STATE: dict = {}
app = FastAPI(title="盘扣架规范问答 Demo")


def boot() -> None:
    if STATE:
        return
    nodes = from_jsonl(ROOT / "out" / "nodes.jsonl")
    bundle = build_index(nodes, ROOT / "out" / "index",
                         embedder=Embedder(cache_path=ROOT / "out" / "index" / "embeddings.json"))
    STATE["bundle"] = bundle
    STATE["answerer"] = Answerer(bundle)
    STATE["nodes"] = {n["node_id"]: n for n in nodes}
    STATE["pdf"] = None


def _pdf():
    """惰性打开 PDF（pypdfium2），用于高亮渲染。"""
    if STATE.get("pdf") is None:
        import pypdfium2 as pdfium
        STATE["pdf"] = pdfium.PdfDocument(str(PDF))
    return STATE["pdf"]


def render_page(book_page: int, boxes: list[list[float]], dpi: int = 140) -> bytes:
    """渲染原书某页，并把给定坐标框画成高亮。坐标是 pdfplumber 约定（top 距上沿）。"""
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/api/health")
def health():
    boot()
    b = STATE["bundle"]
    return {"nodes": len(b.nodes), "graph": b.graph.stats(),
            "embedder": b.embedder_name, "model": STATE["answerer"].model}


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
    prepared = ans.prepare(question, top_k=int((payload or {}).get("top_k", 6)))

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
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
