# -*- coding: utf-8 -*-
"""
生成层：上下文组装 → 带引用约束的生成 → 引用回查 → 拒答判定。

三条约束决定了这里的设计：
  1. 施工规范问答必须给法源——所以提示词强制标注条号，生成后还要回查
  2. 材料里没有的不能编——所以要显式拒答，而不是"尽力回答"
  3. 命中款/公式时用户要看到完整条文——所以要做父子回填
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Sequence

from .schema import MODALITY_LEVEL
from .tables import TableTools

# 答案里出现的引用形态（与 schema.extract_refs 保持一致）
CITE_PATTERNS = (
    # 条号。负向后顾排除"条文说明第4.4.4条"——那是说明节点、不是条文节点，
    # 被抢过来会拿 (clause, 4.4.4) 去回查，而材料里只有 explanation 节点，
    # 结果把一条**正确的**引用误判成幻觉。
    ("clause", re.compile(r"(?<!说明)第\s*(\d+(?:\.\d+){1,2})\s*条")),
    ("table", re.compile(r"表\s*([A-Z]?\.?\d+(?:\.\d+)*(?:-\d+)?)")),
    ("figure", re.compile(r"图\s*(\d+(?:\.\d+)*(?:-\d+)?)")),
    ("formula", re.compile(r"式\s*[（(]\s*([\d.]+(?:-\d+)?)\s*[）)]")),
    # 公式号在本规范正文里是独立成行的、不带"式"字前缀，如 (5.3.3-2)。
    # 不带这条规则的话，模型引用这类公式会被误判成"材料里没有"的幻觉。
    ("formula", re.compile(r"[（(]\s*(\d+\.\d+\.\d+-\d+)\s*[）)]")),
    # 附录条文写成"附录D.0.1"。必须单独认，否则会被下面的"附录D"规则吞掉、丢掉小节号，
    # 回查时定位到的就是附录本体而不是那条条文。
    ("clause", re.compile(r"附录\s*([A-Z]\.\d+(?:\.\d+)?)")),
    # 附录。以前没有这条：模型引用附录时既不算有效也不算幻觉，直接静默消失，
    # 引用回查的统计里凭空少一类。
    # 两个负向断言把"附录D.0.1"这类带条号的写法排除在外：(?!\s*[A-Z]) 挡"附录DD"，
    # (?!\.\d) 挡"附录D.0.1"。
    ("appendix", re.compile(r"附录\s*([A-Z])(?!\s*[A-Z])(?!\.\d)")),
    # 条文说明。材料块的标签就是"条文说明 4.4.4"，模型会照这个形态抄。
    # 缺这条规则会造成两种坏结果：写成"条文说明 4.4.4"时引用**完全静默**
    # （不算有效也不算幻觉），写成"条文说明第4.4.4条"时被 clause 规则抢走、
    # 误判成幻觉。两种都测过。
    ("explanation", re.compile(r"条文说明\s*第?\s*(\d+(?:\.\d+){1,2})\s*条?")),
)
REFUSAL_MARKERS = ("未规定", "未涉及", "没有规定", "无法回答", "材料中未", "本章未")

# 答案里"来源标记"的形态，提示词要求写成 [来源：标准号 条号]
SOURCE_TAG_RE = re.compile(r"[\[【]\s*来源\s*[：:]\s*([^\]】]+)[\]】]")

# 任意标准号形态，用来判断模型引用的标准**在不在本次材料里**：
#   JGJ/T 231-2021 / DB11/T 2100-2023 / GB 51210-2016 / CJJ 166-2011
STANDARD_CODE_RE = re.compile(r"[A-Z]{2,4}\d*\s*/?\s*T?\s*\d+(?:\.\d+)?\s*-\s*\d{4}")

# 从问题里抽"可能是表格键"的候选值：
#   · 数字（λ=100 / 长细比 100 / 高度 20）
#   · 型号串（DB11 的 Z-LG-500 / B-LG-3000）
_NUM_CAND = re.compile(r"\d+(?:\.\d+)?")
_MODEL_CAND = re.compile(r"[A-Z]{1,3}-[A-Z]{1,4}-\d+(?:\.\d+)?")


def _title_overlap(title: str, question: str) -> int:
    """
    表标题与问题的二元组重叠数，用来判断"这张表跟问题有没有关系"。

    没有这道闸会出现"一个数字匹配所有表"：问"长细比是100的Q235钢管稳定系数"，
    数字 100 既能当 λ 又能当离地高度，于是风压系数表也被查了一遍，
    材料里多出两条无关的查表结果。
    """
    def grams(s: str) -> set:
        s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s or "")
        return {s[i:i + 2] for i in range(len(s) - 1)}

    return len(grams(title) & grams(question))

def build_system_prompt(standards: Sequence[dict], multi: bool,
                        detail: str = "rich") -> str:
    """
    系统提示词按知识库里的规范动态生成。
    单规范时引用只写条号；多规范时必须带标准号——否则"第5.1.2条"指哪一本说不清。

    detail 控制答案篇幅：
      "brief" —— 250 字以内、只给结论（首字和总时延都短）
      "rich"  —— 充分展开，带上适用条件与例外（默认）
    篇幅和 max_tokens 必须一起调，只改一边会出现"还没说完就被截断"。
    """
    names = "；".join(f"《{s['name']}》{s['code']}" for s in standards) or "（空）"
    cite_rule = (
        "格式严格写成：[来源：标准号 条号]，例如 [来源：JGJ/T 231-2021 第5.3.3条]、"
        "[来源：DB11/T 2100-2023 表3.0.6]。**标准号不能省略**——知识库里有多个标准，"
        "同一个条号在不同标准下含义不同。"
        if multi else
        "格式严格写成：[来源：第5.3.3条] 或 [来源：表5.1.9] 或 [来源：图5.1.4] 或 [来源：式（5.3.3-1）]"
    )
    rules = [
        f"1. 每个结论后面必须标注来源，{cite_rule}",
        '2. 材料中没有的内容，直接回答"提供的材料中未规定"，不要编造',
        "3. 涉及数值、公式、表格取值时原样引用，不要换算、不要四舍五入",
        "4. 用书面语分点陈述，不要大段照抄材料原文",
    ]
    rules += (
        ["5. 回答控制在 250 字以内，分点不超过 6 条",
         "6. 不要展开分析过程，直接给结论"]
        if detail == "brief" else
        ["5. 回答要充分展开，篇幅 500~1200 字：先给结论，再说明适用条件、边界与例外",
         "6. 材料里有多条相关规定的，逐条列出并分别标注来源，不要合并成一句话"]
    )
    if multi:
        rules.append("7. 不同标准对同一问题规定不一致时，必须分别列出并说明各自出处，"
                     "不要合并成一个结论，也不要替读者取舍")
    rules.append(f"{len(rules) + 1}. 若材料里出现\"用词：严禁/必须/应/宜/可\"，"
                 "引用时要保留该用词，不要把\"宜\"说成\"应\"")
    return ("你是工程建设标准的检索助手。\n"
            f"本知识库包含：{names}\n"
            "只依据【材料】回答，禁止使用材料之外的任何知识，禁止推测。\n\n"
            "规则：\n" + "\n".join(rules))


def _label(node: dict) -> str:
    """给材料块一个可被引用的编号标签。"""
    t, num = node["type"], node["num"]
    if t == "table":
        return f"表{num}"
    if t == "figure":
        return f"图{num}"
    if t == "formula":
        return f"式（{num}）"
    if t in ("clause", "item"):
        # 附录条文用 A.0.1 这种编号，标签也该是"附录A.0.1"而不是"第A.0.1条"
        return f"附录{num}" if re.match(r"^[A-Z]\.", str(num)) else f"第{num}条"
    if t == "section":
        return f"第{num}节"
    if t == "appendix":
        return f"附录{num}"
    if t == "explanation":
        return f"条文说明 {num}"
    return f"第{num}节"


def _standard_near(answer: str, pos: int, code_map: dict) -> tuple[str, bool]:
    """
    把一处引用归属到某本规范：从引用位置**往前**找最近出现的标准号。

    返回 (standard_id, 是否可信)：
      ("", True)   —— 引用前面没有标准号。单规范模式下提示词允许省略，返回空串，
                      由调用方按编号反查兜底
      (sid, True)  —— 定位到了本次材料里的某一本
      ("", False)  —— 前面有标准号，但它**不在本次材料里**，直接判为幻觉。
                      例如材料只给了 JGJ 的内容，模型却写"[来源：DB11/T 2100-2023 第8.0.7条]"——
                      这两本的 8.0.7 内容完全不同，不能算数
    """
    found, code = -1, ""
    for m in STANDARD_CODE_RE.finditer(answer, 0, pos):
        if m.start() > found:
            found, code = m.start(), m.group(0)
    if found < 0:
        return "", True
    key = re.sub(r"\s+", "", code)
    for known, sid in code_map.items():
        if re.sub(r"\s+", "", known) == key:
            return sid, True
    return "", False


def _render_template_table(b: dict, fallback: str = "") -> str:
    """
    表单模板的渲染（表 D.0.1 / D.0.2 这类施工验收记录表）。

    不能套用普通表格的 markdown_flat：这类表大量使用合并单元格，`fill_down` 之后
    整列都是重复值——实测表 D.0.1 的表头被 pdfplumber 抽成 22 列全是"项目名称"，
    渲染出来是一片乱格子，信息量为零。

    真正有用的是**第一列去重后的清单**：它是这张表的填写项与检查项
    （项目名称、搭设部位、立杆垂直度≤L/500且±50、水平杆水平度…）。
    """
    rows = b.get("rows") or []
    items, seen = [], set()
    for row in rows:
        v = re.sub(r"\s+", " ", (row[0] if row else "") or "").strip()
        v = v.replace(" / ", "/").strip("/")
        if v and v not in seen and len(v) <= 40:
            seen.add(v)
            items.append(v)
    if not items:
        return fallback
    return "（表单模板，原表为空白记录表）\n表中栏目与检查项：" + "、".join(items)


def render_block(node: dict, with_standard: bool = False) -> str:
    """把节点渲染成给 LLM 的材料块。表格用扁平表头版，公式带变量表。"""
    label = _label(node)
    # 多规范并存时，每块材料都必须标明来自哪本——否则模型无法在答案里说清出处
    head = f"【{node['standard_code']} {label}】" if with_standard else f"【{label}】"
    if node["section"]:
        head += f"（{node['section']}）"
    if node["modality"]:
        head += f"［用词：{node['modality']}，严格程度 {MODALITY_LEVEL[node['modality']]}/4］"
    body = node["content"]
    if node["type"] == "table":
        b = node["body"]
        if (b.get("tool") or {}).get("kind") == "template":
            body = _render_template_table(b, body)
        else:
            body = b.get("markdown_flat") or b.get("markdown") or body
        if b.get("notes"):
            body += "\n注：" + " ".join(b["notes"])
    elif node["type"] == "figure":
        b = node["body"]
        body = b["title"] + "\n图注：" + "；".join(b["legend"])
    elif node["type"] == "formula":
        b = node["body"]
        body = f"式（{b['equation_id']}）线性形式：{b.get('linear_text') or '（未识别）'}"
        if b["variables"]:
            body += "\n式中：" + "；".join(
                f"{v['symbol']}——{v['meaning']}" for v in b["variables"])
    out_of_scope = [r["target_display"] for r in node["refs"] if r["in_scope"] is False]
    if out_of_scope:
        # 必须显式告知：条文里写"按附录B采用"，但附录B不在本次材料里。
        # 不标注的话模型会把它当成自己的引用来源，回查时报假阳性。
        body += "\n（注：本条引用的 " + "、".join(out_of_scope) + " 不在本次材料范围内）"
    return f"{head}\n{body}"


class Answerer:
    """
    参数
      bundle     IndexBundle（BM25 + 向量 + 表格库 + 引用图）
      llm_base   LM Studio 的 OpenAI 兼容地址
      model      生成模型名
    """

    def __init__(self, bundle, llm_base: str = "http://localhost:1234/v1",
                 model: str = "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp",
                 temperature: float = 0.1, timeout: int = 900, max_tokens: "int | None" = None,
                 reasoning_effort: str = "none", detail: str = "rich"):
        self.bundle = bundle
        self.llm_base = llm_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # 篇幅与 token 预算必须配套。中文约 1.5 token/字，"rich" 模式正文上限
        # 1200 字 ≈ 1800 token，给到 2500 留余量；"brief" 沿用原来的 1200。
        # ⚠ 本地 27B 实测约 15 token/s，跑满 2500 上限约 2.5 分钟——
        #   输出越丰富，等待越久，这是本地部署绕不开的代价。
        self.detail = detail
        if max_tokens is None:
            max_tokens = 1200 if detail == "brief" else 2500
        self.max_tokens = max_tokens
        # 这两个本地模型默认"永远思考"：先把 token 预算烧在 reasoning_content 上，
        # 预算不够就只思考不出正文（空答案），且首字要等 15~26 秒。
        # LM Studio 支持 reasoning_effort="none" 直接关闭思考：首字降到 2~3 秒，答案反而更直接。
        # 注意取值是 "none"，"off" 会返回 HTTP 400。
        self.reasoning_effort = reasoning_effort
        self.by_id = {n["node_id"]: n for n in bundle.nodes}
        # 知识库里有哪些规范：决定提示词怎么写、材料块要不要带标准号
        self.standards = sorted(
            {(n.get("standard_id"), n.get("standard_code"), n.get("standard_name") or "")
             for n in bundle.nodes},
            key=lambda x: x[1] or "")
        self.multi_standard = len(self.standards) > 1
        self._prompts: dict[str, str] = {}
        self.system_prompt = self._prompt_for(detail)
        # 附表工具：查值表（稳定系数/风压系数/构配件规格…）的运行时查询入口
        self.table_tools = TableTools.from_nodes(bundle.nodes)

    def _prompt_for(self, detail: str) -> str:
        """按篇幅档位取系统提示词（缓存，避免每次请求重建）。"""
        if detail not in self._prompts:
            self._prompts[detail] = build_system_prompt(
                [{"id": s[0], "code": s[1], "name": s[2]} for s in self.standards],
                self.multi_standard, detail)
        return self._prompts[detail]

    # ---------------------------------------------------------- 附表查值
    @staticmethod
    def _key_candidates(question: str) -> list[str]:
        """从问题里抽出可能的表格键：先试型号串（更具体），再试数字（长的优先）。"""
        models = _MODEL_CAND.findall(question)
        nums = _NUM_CAND.findall(question)
        # 去掉年份、规范号里的数字（如 GB 50009、2021）
        nums = [n for n in nums if not (len(n) == 4 and n.startswith(("19", "20")))]
        nums.sort(key=len, reverse=True)
        out = []
        for c in models + nums:
            if c not in out:
                out.append(c)
        return out

    def _probe_table(self, node: dict, question: str) -> "str | None":
        """
        命中的表若是查值表，就替模型把值查好，作为一行短材料附在表后面。

        这样做而不是让模型自己读整张表：一张稳定系数表有 251 个值、2000+ 字，
        模型在数字流里挑对的概率远低于按参数精确查表。
        """
        body = node.get("body") or {}
        tool = body.get("tool")
        if not tool or tool.get("kind") != "lookup":
            return None
        # 表标题里的钢材牌号要与问题一致：问 Q235 时别把 Q355 的表也查出来，
        # 否则两张表的值一起进材料，模型有拿错的风险。
        title = body.get("title", "")
        grades = [g for g in ("Q355", "Q235", "Q195") if g in title]
        asked = [g for g in ("Q355", "Q235", "Q195") if g in question]
        if grades and asked and not set(grades) & set(asked):
            return None
        # 标题与问题的语义重叠不足就跳过，避免"一个数字匹配所有表"
        if _title_overlap(title, question) < 2:
            return None
        for cand in self._key_candidates(question):
            hit = self.table_tools.lookup(node["standard_id"], node["num"], cand)
            if not hit:
                continue
            if isinstance(hit["value"], dict):
                attrs = "；".join(f"{k}={v}" for k, v in hit["value"].items())
                text = f"{hit['key_name']}={hit['key']} → {attrs}"
            else:
                text = f"{hit['key_name']}={hit['key']} → {hit.get('value_name', '值')}={hit['value']}"
            mark = "" if hit["exact"] else "（表中无此精确值，取最接近的一档）"
            return f"【查表结果】{hit['title']}：{text}{mark}"
        return None

    def _direct_lookup(self, question: str, exclude: Sequence[str] = ()) -> list[dict]:
        """
        直接对所有查值表试键，**不依赖检索命中**。

        为什么需要这一步：型号（Z-LG-3000）、参数（λ=100）这类串对向量检索来说
        信息量太低，表往往不会被召回——但它恰恰是唯一能回答问题的依据。
        查值是精确匹配，误命中风险远低于向量相似度，所以可以全表扫一遍。
        """
        cands = self._key_candidates(question)
        if not cands:
            return []
        out, seen = [], set(exclude)
        for key, tool in self.table_tools.tools.items():
            if tool.get("kind") != "lookup" or tool["node_id"] in seen:
                continue
            sid, num = key.split(":", 1)
            node = self.by_id.get(tool["node_id"])
            if not node:
                continue
            probe = self._probe_table(node, question)
            if not probe:
                continue
            seen.add(tool["node_id"])
            out.append({"node_id": node["node_id"], "type": "table", "num": node["num"],
                        "label": _label(node), "pages": node["pages"],
                        "bboxes": node["bboxes"], "image_path": None,
                        "standard_id": node.get("standard_id", ""),
                        "standard_code": node.get("standard_code", ""),
                        "standard_name": node.get("standard_name", ""),
                        "text": f"{_label(node)}\n" + probe, "direct_lookup": True})
        return out

    # ---------------------------------------------------------- 上下文组装
    # 上下文预算：本地 27B 要先吞完材料才吐第一个字，但**实测代价很低**——
    # 关掉思考模式后，6 块→2.4s、10 块→3.6s、20 块→4.2s，几乎不影响首字。
    # （早期"10 块材料首字 26.7s"是在 reasoning 开着的情况下测的，前提已不成立。）
    # 材料越多，闭包带回的公式/附表越全，所以默认给 10 块。
    # 能进模型上下文的类型。两个容易被漏掉的：
    #   explanation —— 条文说明是"这条为什么这么规定"的唯一来源，128 个节点。
    #                  之前不在白名单，导致 _link_explanations 建好的双向关联在生成阶段被切断。
    #   appendix    —— 附录节点本身只是"指路牌"（内容就是"附录D 脚手架施工验收记录"），
    #                  但它同时是父子回填的上游：命中 D.0.1 时要把附录D 带进来，
    #                  否则用户看到的材料里没有"这条在附录D 里"。回填也走这个白名单。
    CONTEXT_TYPES = ("clause", "item", "table", "figure", "formula",
                     "explanation", "appendix")

    def build_context(self, hits: Sequence[dict], max_chars: int = 9000,
                      max_blocks: int = 10, question: str = "") -> list[dict]:
        """
        父子回填 + 去重 + 排序。
        命中款/表/图/公式时把所属条文一并带上（用户要看完整条文）；
        排序：条文 → 款 → 节 → 表 → 公式 → 图。
        """
        picked: dict[str, dict] = {}
        for r in hits:
            # 节/章只是导航节点，正文是空的，喂进去只会拖慢首字，不提供任何依据
            if r["type"] not in self.CONTEXT_TYPES:
                continue
            picked.setdefault(r["node_id"], dict(r))
        for r in list(picked.values()):
            n = self.by_id.get(r["node_id"])
            if not n:
                continue
            pid = n["parent_id"]
            if pid in self.by_id and pid not in picked \
                    and self.by_id[pid]["type"] in self.CONTEXT_TYPES:
                p = self.by_id[pid]
                picked[pid] = {"node_id": pid, "type": p["type"], "num": p["num"],
                               "score": 0.0, "via_parent": r["node_id"]}

        # 内容型的排前面，导航型的排后面。
        # 这张表决定"扩展带出的节点"谁抢得到剩下的名额（默认 10 块里留 2 个位置）。
        # 之前漏了 appendix / explanation，它们会拿到默认值 9 掉到队尾——
        # 结果就是检索列表里能看到附录D，材料里却永远没有它。
        order = {"clause": 0, "item": 1, "explanation": 2,
                 "table": 3, "formula": 4, "figure": 5,
                 "appendix": 6, "section": 7, "chapter": 8}
        # 两段填充：先把"直接命中"（score>0）按类型排进去，再补"扩展带出"的。
        # 否则零分的扩展节点会靠类型顺序把高分的表/图挤出名额——实测踩过：
        # 问"可调托撑承载力"，表5.1.9 被 5 条零分扩展条文挤掉，模型只能回答"材料中未规定"。
        direct = sorted([r for r in picked.values() if r.get("score", 0) > 0],
                        key=lambda r: (order.get(r["type"], 9), -r["score"]))
        expanded = sorted([r for r in picked.values() if not r.get("score", 0) > 0],
                          key=lambda r: order.get(r["type"], 9))
        # 分数比例过滤：只保留与最高分同量级的直接命中。
        # 查值类问题只要 2~3 块材料，多喂的每一块都在按秒计费（提示词处理 ~100 token/s）。
        if direct:
            top = direct[0]["score"]
            # 上限跟着 max_blocks 走，给扩展命中留 2 个位置
            core = [r for r in direct if r["score"] >= 0.5 * top][:max(2, max_blocks - 2)]
        else:
            core = []
        core = core or direct[:2]
        core_ids = {c["node_id"] for c in core}
        # 父条文已入选的"款"要提前剔掉，不能等到渲染循环里才跳过它。
        # 渲染循环的跳过是对的（父条文已收录，再单独渲染"款"就是同一段话出现两次），
        # 但那是**切片之后**才做的——被跳过的款已经白占了 expanded 的名额。
        # 实测踩过：8.0.5 / 8.0.4 入选 core 后，它们的 4 个款占满了 expanded 的 2 个位置，
        # 引用图带出来的附录D 被挤在外面，材料里就永远看不到"这条规定在附录D 里"。
        core_clauses = {c["node_id"] for c in core if c["type"] == "clause"}
        expanded = [r for r in expanded
                    if r["node_id"] not in core_ids
                    and not (r["type"] == "item"
                             and self.by_id[r["node_id"]]["parent_id"] in core_clauses)]
        room = max(1, max_blocks - len(core))
        expanded = expanded[:room]
        items = (core + expanded)[:max_blocks]
        out, used, seen_clause = [], 0, set()
        for r in items:
            if len(out) >= max_blocks:
                break
            n = self.by_id[r["node_id"]]
            if n["type"] == "clause":
                seen_clause.add(n["node_id"])
            # 父条文已收录时不再单独渲染"款"，否则材料里同一段话出现两次
            if n["type"] == "item" and n["parent_id"] in seen_clause:
                continue
            text = render_block(n, with_standard=self.multi_standard)
            if question and n["type"] == "table":
                probe = self._probe_table(n, question)
                if probe:
                    text += "\n" + probe
            if used + len(text) > max_chars:
                continue
            used += len(text)
            out.append({"node_id": n["node_id"], "type": n["type"], "num": n["num"],
                        "label": _label(n), "pages": n["pages"], "bboxes": n["bboxes"],
                        "standard_id": n.get("standard_id", ""),
                        "standard_code": n.get("standard_code", ""),
                        "standard_name": n.get("standard_name", ""),
                        "text": text,
                        "image_path": (n["body"] or {}).get("image_path") if n["body"] else None})
        return out

    # ---------------------------------------------------------- 生成
    def _chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "stream": False}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(self.llm_base + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"生成模型调用失败（{self.llm_base}, model={self.model}）：{e}") from e

    def stream(self, messages: list[dict]):
        """
        流式生成，产出 (kind, text)，kind ∈ {"reasoning", "content"}。

        这个模型是推理型：先吐几十段 reasoning_content 再吐正文。
        必须把思考流也接出来——一是让前端立刻有反馈，二是 max_tokens 要同时覆盖
        思考与正文，Token 预算给少了会出现"思考完就没正文"的空答案（踩过）。
        """
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "stream": True}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(self.llm_base + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except Exception:
                    continue
                if delta.get("reasoning_content"):
                    yield "reasoning", delta["reasoning_content"]
                if delta.get("content"):
                    yield "content", delta["content"]

    def prepare(self, question: str, top_k: int = 6, expand_hops: int = 1,
                alpha: float = 0.5, max_blocks: int = 10,
                detail: "str | None" = None) -> dict:
        """先做检索与上下文组装（快），把 messages 交给生成阶段（慢）。"""
        hits = self.bundle.retriever.search(question, top_k=top_k, alpha=alpha,
                                            expand_hops=expand_hops)
        contexts = self.build_context(hits, max_blocks=max_blocks, question=question)
        # 查值表补一轮"直接匹配"：型号/参数精确串往往召回不到表，但它是唯一依据
        direct = self._direct_lookup(question, exclude=[c["node_id"] for c in contexts])
        if direct:
            contexts = contexts + direct[:3]
        material = "\n\n".join(c["text"] for c in contexts) or "（无）"
        return {
            "question": question,
            "messages": [
                {"role": "system", "content": self._prompt_for(detail or self.detail)},
                {"role": "user", "content": f"【材料】\n{material}\n\n【问题】\n{question}"},
            ],
            "detail": detail or self.detail,
            "standards": [{"code": s[1], "name": s[2]} for s in self.standards],
            "contexts": contexts,
            "table_rows": self.bundle.tables.find_rows(question, top_k=3),
            "hits": [{"node_id": h["node_id"], "type": h["type"], "num": h["num"],
                      "score": round(h.get("score", 0.0), 4),
                      "expanded_query": h.get("expanded_query")} for h in hits],
            "model": self.model,
        }

    def finalize(self, prepared: dict, answer: str) -> dict:
        check = self.verify_citations(answer, prepared["contexts"])
        return {**prepared, "answer": answer.strip(),
                "citations": check["cited"], "secondary_citations": check["secondary"],
                "invalid_citations": check["invalid"], "citation_ok": check["all_valid"],
                "has_citation": check["has_citation"], "refused": self.is_refusal(answer),
                "messages": None}

    # ---------------------------------------------------------- 引用回查
    @staticmethod
    def verify_citations(answer: str, contexts: Sequence[dict]) -> dict:
        """
        三分类回查，比"对/错"更准确：
          · cited      —— 材料里确实有这个节点（硬依据）
          · secondary  —— 编号只出现在材料正文的转述里（如条文写"按附录B表B.0.2采用"），
                          模型引用它不算幻觉，但也不是直接依据
          · invalid    —— 材料里根本找不到，判为幻觉

        **引用必须定位到具体标准。** 两本规范的 (类型, 编号) 有 135 个重叠
        ——所有章节号，加上大批条文号。例如 JGJ 和 DB11 都有 8.0.7，内容完全不同：
        JGJ 8.0.7 是"验收后应形成记录"，DB11 8.0.7 是"发生下列情况时应重新检查验收"。
        只比 (类型, 编号) 的话，模型把 DB11 的结论标成 JGJ 的条号也会判为有效，
        混库模式的防幻觉机制等于失效。所以比对键统一成三元组（标准, 类型, 编号）。
        """
        allowed = {(c.get("standard_id", ""), c["type"], str(c["num"])) for c in contexts}
        # 反查：同一个 (类型, 编号) 出现在哪几本里——答案省略标准号时用它兜底
        owners: dict[tuple[str, str], set[str]] = {}
        for sid, node_type, num in allowed:
            owners.setdefault((node_type, num), set()).add(sid)
        # 标准号 / 标准标识 → standard_id，用来把答案里的引用归到某一本
        code_map: dict[str, str] = {}
        for c in contexts:
            for key in (c.get("standard_code"), c.get("standard_id")):
                if key:
                    code_map[key] = c.get("standard_id", "")

        mentioned: set[tuple[str, str, str]] = set()
        for c in contexts:
            sid = c.get("standard_id", "")
            for node_type, pattern in CITE_PATTERNS:
                for raw in pattern.findall(c["text"]):
                    mentioned.add((sid, node_type, re.sub(r"\s+", "", raw)))

        cited, secondary, bad = [], [], []
        # 只在**来源标记内部**抽引用。模型在正文里复述"应符合附录D的要求"是正常表述，
        # 不该被当成一次引用声明（否则每句转述都会拿去回查，凭空报出一堆幻觉）。
        # 完全没有来源标记时退回全篇扫描——那是模型没按格式写，但仍要能查出问题。
        segments = [(m.start(1), m.group(1)) for m in SOURCE_TAG_RE.finditer(answer)]
        if not segments:
            segments = [(0, answer)]
        for base, seg in segments:
            for node_type, pattern in CITE_PATTERNS:
                for m in pattern.finditer(seg):
                    num = re.sub(r"\s+", "", m.group(1))
                    sid, known = _standard_near(answer, base + m.start(), code_map)
                    if known and sid:
                        cands = [(sid, node_type, num)]
                    elif known:
                        cands = [(s, node_type, num)
                                 for s in owners.get((node_type, num), ())]
                    else:
                        cands = []          # 引用了不在材料里的标准
                    item = {"type": node_type, "num": num, "standard_id": sid}
                    if any(c in allowed for c in cands):
                        if item not in cited:
                            cited.append(item)
                    elif any(c in mentioned for c in cands):
                        if item not in secondary:
                            secondary.append(item)
                    elif item not in bad:
                        bad.append(item)
        return {"cited": cited, "secondary": secondary, "invalid": bad,
                "has_citation": bool(cited or secondary), "all_valid": not bad,
                "allowed": sorted(f"{c['label']}" for c in contexts)}

    @staticmethod
    def is_refusal(answer: str) -> bool:
        return any(m in answer for m in REFUSAL_MARKERS)

    # ---------------------------------------------------------- 入口
    def answer(self, question: str, top_k: int = 6, expand_hops: int = 1,
               alpha: float = 0.5, max_blocks: int = 10,
               detail: "str | None" = None) -> dict:
        prepared = self.prepare(question, top_k=top_k, expand_hops=expand_hops,
                                alpha=alpha, max_blocks=max_blocks, detail=detail)
        return self.finalize(prepared, self._chat(prepared["messages"]))
