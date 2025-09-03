#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Evolution Sankey (edges.csv + optional nodes.csv)
======================================================
- 读取 edges.csv 并可选读取 nodes.csv，用于更友好的标签与悬浮提示。
- 若 nodes_enriched.csv 中包含 label_llm/summary_llm，将优先使用。

用法示例：
    python plot_sankey_plus.py \
      --edges ./ten_out/edges.csv \
      --nodes ./ten_out/nodes_enriched.csv \
      --out sankey_plus.html \
      --label-mode auto --label-topk 3 \
      --min-weight 0.7 \
      --year-range 2010:2018

参数：
--edges         (必需) edges.csv，包含 source,target,weight[,kind]
--nodes         (可选) nodes.csv 或 nodes_enriched.csv（含 id,year,topic,top_terms,count 等）
--out           输出 HTML，默认 sankey.html
--label-mode    auto/keywords/year_topic/id（auto：优先 label_llm，否则 keywords）
--label-topk    关键词显示个数（keywords 模式）
--min-weight    过滤弱边（例如 0.7）
--year-range    仅显示该年份区间的节点/边（格式 2010:2018，需要 nodes.year）
--width/--height 图尺寸
"""
import argparse, pandas as pd, numpy as np, re
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


def short_terms(s: str, k: int = 3) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("、", ";").replace(",", ";").replace("|", ";")
    toks = [t.strip() for t in s.split(";") if t.strip()]
    return ", ".join(toks[:k])


def parse_year_range(s: str | None):
    if not s:
        return None
    m = re.match(r"^\s*(\d{3,4})\s*:\s*(\d{3,4})\s*$", s)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--out", default="sankey_plus.html")
    ap.add_argument("--label-mode", default="auto", choices=["auto", "keywords", "year_topic", "id"]) 
    ap.add_argument("--label-topk", type=int, default=3)
    ap.add_argument("--min-weight", type=float, default=None)
    ap.add_argument("--year-range", default=None)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    # 读 edges
    e = pd.read_csv(args.edges)
    e.columns = [c.strip().lower() for c in e.columns]
    for req in ["source", "target", "weight"]:
        if req not in e.columns:
            raise ValueError(f"edges.csv 缺少列: {req}")

    if args.min_weight is not None:
        e = e[e["weight"].astype(float) >= args.min_weight]

    # 读 nodes
    node_df = None
    if args.nodes:
        node_df = pd.read_csv(args.nodes)
        node_df.columns = [c.strip().lower() for c in node_df.columns]
        if "year" in node_df.columns:
            node_df["year"] = pd.to_numeric(node_df["year"], errors="coerce").astype("Int64")
        # 过滤年份
        yr = parse_year_range(args.year_range)
        if yr and "year" in node_df.columns:
            a, b = yr
            keep_ids = set(node_df[(node_df["year"] >= a) & (node_df["year"] <= b)]["id"].astype(str))
            e = e[e["source"].astype(str).isin(keep_ids) & e["target"].astype(str).isin(keep_ids)]
        elif yr and "year" not in node_df.columns:
            print("[Warn] year-range 指定了，但 nodes.csv 没有 year 列，忽略。")

    # 构建节点集合（以边端点为准）
    node_ids = sorted(set(e["source"].astype(str)).union(set(e["target"].astype(str))))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    # 索引化边
    src_idx = e["source"].astype(str).map(idx_of).tolist()
    tgt_idx = e["target"].astype(str).map(idx_of).tolist()
    values = e["weight"].astype(float).tolist()

    # 组装节点显示信息
    labels, hover, colors = [], [], None

    counts = years = terms = topics = None
    label_llm = summary_llm = None

    if node_df is not None and not node_df.empty:
        meta = node_df.set_index(node_df["id"].astype(str))
        if "count" in meta.columns:
            counts = []
        if "year" in meta.columns:
            years = []
        if "top_terms" in meta.columns:
            terms = []
        if "topic" in meta.columns:
            topics = []
        if "label_llm" in meta.columns:
            label_llm = []
        if "summary_llm" in meta.columns:
            summary_llm = []

        for nid in node_ids:
            row = meta.loc[nid] if nid in meta.index else None
            y   = int(row["year"]) if (row is not None and "year" in row and pd.notna(row["year"])) else None
            tpc = int(row["topic"]) if (row is not None and "topic" in row and pd.notna(row["topic"])) else None
            tt  = str(row["top_terms"]) if (row is not None and "top_terms" in row and pd.notna(row["top_terms"])) else ""
            cnt = int(row["count"]) if (row is not None and "count" in row and pd.notna(row["count"])) else None
            lab = str(row["label_llm"]) if (row is not None and "label_llm" in row and pd.notna(row.get("label_llm"))) else ""
            summ= str(row["summary_llm"]) if (row is not None and "summary_llm" in row and pd.notna(row.get("summary_llm"))) else ""

            if years is not None:  years.append(y)
            if topics is not None: topics.append(tpc)
            if terms is not None:  terms.append(tt)
            if counts is not None: counts.append(cnt)
            if label_llm is not None: label_llm.append(lab)
            if summary_llm is not None: summary_llm.append(summ)

        # 标签策略
        mode = args.label_mode
        if mode == "auto" and label_llm is not None and any(lab.strip() for lab in label_llm):
            # 优先用 label_llm（为空时回退）
            for i, nid in enumerate(node_ids):
                lab = label_llm[i] if label_llm else ""
                if lab and lab.strip():
                    labels.append(lab.strip())
                else:
                    # 回退：year + keywords
                    y  = years[i] if years else None
                    tt = terms[i] if terms else ""
                    fallback = (f"{y} | {short_terms(tt, args.label_topk)}") if y is not None else short_terms(tt, args.label_topk)
                    labels.append(fallback if fallback else nid)
        elif mode == "keywords":
            for i, nid in enumerate(node_ids):
                y  = years[i] if years else None
                tt = terms[i] if terms else ""
                label = (f"{y} | {short_terms(tt, args.label_topk)}") if y is not None else short_terms(tt, args.label_topk)
                labels.append(label if label else nid)
        elif mode == "year_topic":
            for i, nid in enumerate(node_ids):
                y   = years[i] if years else None
                tpc = topics[i] if topics else None
                label = f"{y}:{tpc}" if (y is not None and tpc is not None) else nid
                labels.append(label)
        else:  # id
            labels = node_ids[:]

        # hover（优先展示 summary_llm）
        for i, nid in enumerate(node_ids):
            y   = years[i]  if years  else None
            tpc = topics[i] if topics else None
            tt  = terms[i]  if terms  else ""
            cnt = counts[i] if counts else None
            lab = label_llm[i] if label_llm else ""
            summ= summary_llm[i] if summary_llm else ""
            hover_text = f"<b>ID</b>: {nid}"
            if lab: hover_text += f"<br><b>LLM</b>: {lab}"
            if y   is not None:  hover_text += f"<br><b>Year</b>: {y}"
            if tpc is not None:  hover_text += f"<br><b>Topic</b>: {tpc}"
            if cnt is not None:  hover_text += f"<br><b>Count</b>: {cnt}"
            if summ:             hover_text += f"<br><b>Summary</b>: {summ}"
            elif tt:             hover_text += f"<br><b>Top terms</b>: {short_terms(tt, min(args.label_topk, 6))}"
            hover.append(hover_text)

        # 颜色：按 count 渐变（Viridis）
        if counts and any(c is not None for c in counts):
            cvals = np.array([c if c is not None else 0 for c in counts], dtype=float)
            # 用 5/95 分位裁剪，减少离群点影响
            cmin, cmax = np.percentile(cvals, 5), np.percentile(cvals, 95)
            if cmax <= cmin:
                cmin = float(cvals.min()); cmax = float(cvals.max() + 1e-6)
            norm = np.clip((cvals - cmin) / (cmax - cmin + 1e-9), 0, 1)
            colors = sample_colorscale("Viridis", norm.tolist())
        else:
            colors = None

    else:
        # 没有 nodes：用 id 作为标签
        labels = node_ids[:]
        hover  = [f"ID: {nid}" for nid in node_ids]
        colors = None

    # link hover
    link_hover = [f"{s} → {t}<br><b>weight</b>: {w:.3f}" for s, t, w in zip(e["source"].astype(str), e["target"].astype(str), e["weight"].astype(float))]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=14,
            line=dict(color="black", width=0.3),
            label=labels,
            color=colors if colors is not None else "rgba(120,120,200,0.85)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ),
        link=dict(
            source=src_idx,
            target=tgt_idx,
            value=values,
            color="rgba(180,180,180,0.4)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover,
        ),
    )])

    fig.update_layout(
        title="Topic Evolution Sankey",
        font=dict(size=12),
        width=args.width,
        height=args.height,
    )
    fig.write_html(args.out)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
