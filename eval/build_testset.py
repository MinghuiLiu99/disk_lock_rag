# -*- coding: utf-8 -*-
"""
构建测试集并自动校验。

与旧测试集（rag_test_set.csv）的区别：
  · 每题绑定 **证据节点**（node_id），这样才能算召回率——旧集只有
    "relevant_clauses" 这种模糊字段
  · 答案写成**原文可摘录**的形式，并由 check() 自动回查原文——
    旧集里 Q04/Q06/Q29/Q31 的答案在规范里根本找不到，就是缺这道校验
  · 题型覆盖 11 类（含数值陷阱、查表、闭包、跨规范冲突、不可回答）

    python eval/build_testset.py            # 校验并输出
    python eval/build_testset.py --out eval
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，直接 print 下面那些 ✅/❌ 会抛 UnicodeEncodeError，
# 校验结果反而看不全。强制 stdout 走 UTF-8，控制台渲染不了也不会崩。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 题型：fact 事实 / cond 条件 / trap 数值陷阱 / lookup 查表 / table 表结构 /
#       formula 公式 / figure 图 / closure 多跳闭包 / compare 对比 /
#       conflict 跨规范冲突 / explain 条文说明 / unanswerable 不可回答 / enum 枚举
#
# 字段：id, type, difficulty, question, answer_gold, evidence（node_id 列表）, note
CASES: list[dict] = [
    # ---------- JGJ/T 231-2021 事实型 ----------
    dict(id="J01", type="fact", diff="简单", q="支撑架立杆的几何长细比不得大于多少？",
         a="150", ev=["JGJ231:clause:5.1.6"]),
    dict(id="J02", type="fact", diff="简单", q="作业架立杆的几何长细比不得大于多少？",
         a="210", ev=["JGJ231:clause:5.1.6"]),
    dict(id="J03", type="fact", diff="简单", q="受拉杆件的几何长细比不得大于多少？",
         a="350", ev=["JGJ231:clause:5.1.6"]),
    dict(id="J04", type="fact", diff="简单", q="盘扣架搭设步距不应超过多少？",
         a="2m", ev=["JGJ231:clause:6.1.3"]),
    dict(id="J05", type="fact", diff="简单", q="作为扫地杆的最底层水平杆中心线离可调底座底板的高度不应大于多少？",
         a="550mm", ev=["JGJ231:clause:6.2.5"]),
    dict(id="J06", type="fact", diff="中等", q="盘扣架搭设完成后，立杆的垂直偏差限值是多少？",
         a="不应大于支撑架总高度的1/500，且不得大于50mm", ev=["JGJ231:clause:7.4.10"]),
    dict(id="J07", type="fact", diff="中等", q="土层地基上的立杆，垫板的长度不宜少于多少跨？",
         a="2跨", ev=["JGJ231:clause:7.3.2"]),
    dict(id="J08", type="fact", diff="中等", q="双排外作业架一次搭设高度不应超过最上层连墙件几步？自由高度不应大于多少？",
         a="不应超过两步，自由高度不应大于4m", ev=["JGJ231:clause:7.5.1"]),
    dict(id="J09", type="fact", diff="简单", q="水平杆及斜杆插销安装完成后，连续下沉量不应大于多少？",
         a="3mm", ev=["JGJ231:clause:7.4.7"]),

    # ---------- JGJ 数值陷阱型（同一话题多个数值，最易混） ----------
    dict(id="J10", type="trap", diff="中等", q="支撑架可调托撑伸出顶层水平杆或双槽托梁中心线的悬臂长度不应超过多少？",
         a="650mm", ev=["JGJ231:clause:6.2.4"], note="与可调底座的300mm、扫地杆的550mm易混"),
    dict(id="J11", type="trap", diff="中等", q="支撑架可调托撑的丝杆外露长度不应超过多少？",
         a="400mm", ev=["JGJ231:clause:6.2.4"]),
    dict(id="J12", type="trap", diff="中等", q="支撑架可调托撑插入立杆或双槽托梁的长度不得小于多少？",
         a="150mm", ev=["JGJ231:clause:6.2.4"]),
    dict(id="J13", type="trap", diff="中等", q="支撑架可调底座的丝杆外露长度不宜大于多少？",
         a="300mm", ev=["JGJ231:clause:6.2.5"], note="注意这是可调底座不是可调托撑"),
    dict(id="J14", type="trap", diff="困难", q="可调底座和可调托撑安装完成后，立杆外径与螺母台阶内径差不应大于多少？",
         a="2mm", ev=["JGJ231:clause:7.4.6"]),

    # ---------- JGJ 查表型（附表工具化后应能答） ----------
    dict(id="J15", type="lookup", diff="中等", q="长细比为100的Q235钢管轴心受压构件，稳定系数φ取多少？",
         a="0.588", ev=["JGJ231:table:C.0.1"]),
    dict(id="J16", type="lookup", diff="中等", q="λ=125时Q355钢管轴心受压构件的稳定系数是多少？",
         a="0.322", ev=["JGJ231:table:C.0.2"]),
    dict(id="J17", type="lookup", diff="中等", q="30m高度、C类地面粗糙度下，风压高度变化系数取多少？",
         a="0.88", ev=["JGJ231:table:A.0.1"]),
    dict(id="J18", type="lookup", diff="简单", q="标准型（B型）脚手架可调底座的承载力设计值是多少？",
         a="100kN", ev=["JGJ231:table:5.1.9"]),
    dict(id="J19", type="lookup", diff="中等", q="搭设高度20m时，支撑架搭设高度调整系数βH取多少？",
         a="1.10", ev=["JGJ231:table:5.3.2"]),

    # ---------- JGJ 公式 / 图 ----------
    dict(id="J20", type="formula", diff="中等", q="不组合风荷载时，立杆稳定性应按什么公式验算？",
         a="N/(φA) ≤ f", ev=["JGJ231:formula:5.3.3-1", "JGJ231:clause:5.3.3"]),
    dict(id="J21", type="formula", diff="中等", q="立杆计算长度 l0 怎么计算？",
         a="βH", ev=["JGJ231:formula:5.3.2-1", "JGJ231:formula:5.3.2-2"]),
    dict(id="J22", type="figure", diff="简单", q="图6.2.2-3 表示的是哪种斜杆布置型式？",
         a="间隔2跨型式支撑架斜杆设置图", ev=["JGJ231:figure:6.2.2-3"]),

    # ---------- JGJ 多跳闭包 / 对比 ----------
    dict(id="J23", type="closure", diff="困难", q="立杆稳定性验算需要用到哪些参数？分别去哪里取值？",
         a="φ按附录C；W、A按附录B表B.0.2",
         ev=["JGJ231:clause:5.3.3", "JGJ231:appendix:B", "JGJ231:appendix:C",
             "JGJ231:table:B.0.2"], note="必须跨章带回附录B/C才算召回完整"),
    dict(id="J24", type="compare", diff="中等", q="支撑架和作业架的设计计算内容有什么区别？",
         a="支撑架含抗倾覆验算；作业架含连墙件计算",
         ev=["JGJ231:clause:5.1.2", "JGJ231:clause:5.1.3"]),
    dict(id="J25", type="explain", diff="中等", q="密目式安全立网全封闭脚手架的挡风系数φ不宜小于多少？",
         a="0.8", ev=["JGJ231:explanation:4.2.4"],
         note="该值只在条文说明里，正文4.2.4只给计算公式——考查说明区是否被索引"),

    # ---------- DB11/T 2100-2023 事实型 ----------
    dict(id="D01", type="fact", diff="简单", q="承插型盘扣式脚手架能否用扣件式钢管剪刀撑代替斜杆？",
         a="严禁", ev=["DB11T2100:clause:3.0.8"]),
    dict(id="D02", type="fact", diff="简单", q="双排作业脚手架搭设高度不宜大于多少？",
         a="24m", ev=["DB11T2100:clause:5.2.3"]),
    dict(id="D03", type="fact", diff="中等", q="满堂作业脚手架搭设高度不应大于多少？施工总荷载不应大于多少？",
         a="15m；2kN/m²", ev=["DB11T2100:clause:5.2.4"]),
    dict(id="D04", type="fact", diff="简单", q="双排作业脚手架的立杆横距宜选用多少？立杆纵距不宜大于多少？",
         a="横距宜选用0.9m或1.2m，纵距不宜大于1.8m", ev=["DB11T2100:clause:5.2.3"]),
    dict(id="D05", type="fact", diff="中等", q="标准型（B型）立杆的荷载设计值不应大于多少？重型（Z型）呢？",
         a="标准型40kN，重型60kN", ev=["DB11T2100:clause:6.2.1"]),
    dict(id="D06", type="fact", diff="中等", q="可调托撑的U形顶托板厚度不应小于多少？可调底座的垫座板厚度呢？",
         a="顶托板5mm，垫座板6mm", ev=["DB11T2100:clause:4.2.4"]),
    dict(id="D07", type="fact", diff="中等", q="可调托撑托板的边长不宜大于多少？",
         a="120mm", ev=["DB11T2100:clause:6.2.5"]),
    dict(id="D08", type="fact", diff="简单", q="DB11规程中，扫地杆的水平杆中心线距可调底座底板不应大于多少？",
         a="550mm", ev=["DB11T2100:clause:6.2.8"]),

    # ---------- DB11 查表型 ----------
    dict(id="D09", type="lookup", diff="中等", q="构配件型号Z-LG-3000的规格和单重参考值是多少？",
         a="Φ60.3×3.2×3000，18.40kg", ev=["DB11T2100:table:A.0.1"]),

    # ---------- 跨规范冲突型 ----------
    dict(id="X01", type="conflict", diff="困难",
         q="可调托撑伸出顶层水平杆的悬臂长度不应大于多少？",
         a="JGJ/T 231-2021为650mm，DB11/T 2100-2023为500mm，两本标准规定不一致",
         ev=["JGJ231:clause:6.2.4", "DB11T2100:clause:6.2.6"],
         note="必须分列两个标准，不能合并成一个结论"),
    dict(id="X02", type="conflict", diff="中等", q="脚手架的步距不应超过多少？",
         a="两本标准均为2m", ev=["JGJ231:clause:6.1.3", "DB11T2100:clause:3.0.8"],
         note="规定一致时也应分别标注出处"),

    # ---------- 不可回答型 ----------
    dict(id="U01", type="unanswerable", diff="简单", q="盘扣架立杆的颜色有什么要求？", a="", ev=[]),
    dict(id="U02", type="unanswerable", diff="简单", q="盘扣架需要刷什么防火涂料？", a="", ev=[]),
    dict(id="U03", type="unanswerable", diff="中等", q="盘扣架在核电站建设中有什么特殊要求？", a="", ev=[]),

    # ---------- 已知弱点：枚举型 ----------
    dict(id="E01", type="enum", diff="困难", q="两本标准中哪些条文提到了连墙件？",
         a="", ev=[], note="top-k 检索的固有弱点，预期漏答——保留作为对照"),
]


def load_nodes() -> dict:
    nodes = {}
    for p in (ROOT / "output/JGJ_T_231-2021/nodes.jsonl",
              ROOT / "output/DB11_T_2100-2023/nodes.jsonl"):
        for line in p.open(encoding="utf-8"):
            n = json.loads(line)
            nodes[n["node_id"]] = n
    return nodes


def _tokens(text: str) -> list[str]:
    """从答案里抽"必须能在原文找到"的片段：数字带单位、纯数字、引号内内容。"""
    out = []
    out += re.findall(r"\d+(?:\.\d+)?\s*(?:mm|m|kN|kN/m²|kN/m2|kg|%)", text)
    out += [q for q in re.findall(r"[“\"]([^”\"]+)[”\"]", text)]
    out += re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text)
    # 去重保序
    seen, res = set(), []
    for t in out:
        t = t.strip()
        if t and t not in seen:
            seen.add(t); res.append(t)
    return res


def check(cases: list[dict], nodes: dict) -> list[tuple]:
    """校验：证据节点存在 + 答案片段能在证据原文里找到。返回问题列表。"""
    problems = []
    for c in cases:
        if c["type"] == "unanswerable":
            if c["ev"]:
                problems.append((c["id"], "不可回答题不该有证据节点"))
            continue
        if not c["ev"]:
            problems.append((c["id"], "缺少证据节点"))
            continue
        missing = [e for e in c["ev"] if e not in nodes]
        if missing:
            problems.append((c["id"], f"证据节点不存在: {missing}"))
            continue
        if c["type"] == "enum":
            continue
        blob = "\n".join(nodes[e]["content"] for e in c["ev"])
        blob_flat = re.sub(r"\s+", "", blob)
        miss = [t for t in _tokens(c["a"])
                if re.sub(r"\s+", "", t) not in blob_flat]
        if miss:
            problems.append((c["id"], f"答案片段在原文中找不到: {miss}"))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes()
    problems = check(CASES, nodes)

    # 输出
    fields = ["id", "type", "difficulty", "question", "answer_gold",
              "evidence_nodes", "evidence_quote", "answerable", "note"]
    rows = []
    for c in CASES:
        quote = ""
        for e in c["ev"][:1]:
            n = nodes.get(e)
            if n:
                quote = re.sub(r"\s+", " ", n["content"])[:120]
                break
        rows.append({"id": c["id"], "type": c["type"], "difficulty": c["diff"],
                     "question": c["q"], "answer_gold": c["a"],
                     "evidence_nodes": ";".join(c["ev"]), "evidence_quote": quote,
                     "answerable": "否" if c["type"] == "unanswerable" else "是",
                     "note": c.get("note", "")})
    with (out_dir / "testset.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "testset.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    import collections
    print(f"题目 {len(CASES)} 道 → {out_dir/'testset.csv'} / testset.jsonl")
    print("题型分布:", dict(collections.Counter(c["type"] for c in CASES)))
    print("难度分布:", dict(collections.Counter(c["diff"] for c in CASES)))
    print(f"可回答 {sum(1 for c in CASES if c['type']!='unanswerable')} / "
          f"不可回答 {sum(1 for c in CASES if c['type']=='unanswerable')}")
    print()
    if problems:
        print(f"❌ 校验发现 {len(problems)} 个问题：")
        for cid, p in problems:
            print(f"   {cid}: {p}")
    else:
        print("✅ 校验通过：所有答案片段都能在证据节点原文里找到")


if __name__ == "__main__":
    main()
