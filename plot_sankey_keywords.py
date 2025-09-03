#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Sankey from edges.csv (+ optional nodes.csv enrichment)

- 节点标签：支持 id / year_topic / keywords(来自 nodes.top_terms, 可 --label-topk)
- 节点 hover：显示 year/topic/count/top_terms（若提供 nodes）
- 节点配色：按 count 做连续色带（若提供 nodes），否则统一色
- 过滤：--min-weight；--year-range a:b（基于 nodes.year）
"""

import argparse, pandas as pd, numpy as np, re
import plotly.graph_objects as go

def short_terms(s, k=3):
    s = str(s) if s is not None else ""
    # 将常见分隔符统一为 ';'
    s = s.replace("、",";").replace(",",";").replace("|",";")
    toks = [t.strip() for t in s.split(";") if t.strip()]
    return ", ".join(toks[:k])

def parse_year_range(s):
    if not s: return None
    m = re.match(r"^\s*(\d{3,4})\s*:\s*(\d{3,4})\s*$", s)
    if not m: return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b: a, b = b, a
    return (a, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="edges.csv with columns: source,target,weight[,kind]")
    ap.add_argument("--nodes", default=None, help="nodes.csv with columns: id,year,topic,top_terms,count")
    ap.add_argument("--out", default="sankey_keywords.html")
    ap.add_argument("--label-mode", default="keywords", choices=["keywords","year_topic","id"])
    ap.add_argument("--label-topk", type=int, default=3)
    ap.add_argument("--min-weight", type=float, default=None)
    ap.add_argument("--year-range", default=None, help="like 2010:2018 (requires nodes.year)")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    e = pd.read_csv(args.edges)
    # 规范列名
    e.columns = [c.strip().lower() for c in e.columns]
    if "weight" not in e.columns:
        raise ValueError("edges.csv 需要包含 weight 列")
    if "source" not in e.columns or "target" not in e.columns:
        raise ValueError("edges.csv 需要包含 source/target 列")

    # 过滤弱边
    if args.min_weight is not None:
        e = e[e["weight"] >= args.min_weight]

    node_df = None
    if args.nodes:
        node_df = pd.read_csv(args.nodes)
        node_df.columns = [c.strip().lower() for c in node_df.columns]
        # year 统一为 int（无则设 NaN）
        if "year" in node_df.columns:
            with np.errstate(all="ignore"):
                node_df["year"] = pd.to_numeric(node_df["year"], errors="coerce").astype("Int64")
        # 过滤年份范围
        yr = parse_year_range(args.year_range)
        if yr and "year" in node_df.columns:
            a, b = yr
            keep_ids = set(node_df[(node_df["year"]>=a) & (node_df["year"]<=b)]["id"].astype(str))
            e = e[e["source"].astype(str).isin(keep_ids) & e["target"].astype(str).isin(keep_ids)]
        elif yr and "year" not in node_df.columns:
            print("[Warn] year-range 指定了，但 nodes.csv 没有 year 列，忽略。")

    # 构建节点列表：以边里出现过的节点为准
    node_ids = sorted(set(e["source"].astype(str)).union(set(e["target"].astype(str))))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    # 索引化边
    src_idx = e["source"].astype(str).map(idx_of).tolist()
    tgt_idx = e["target"].astype(str).map(idx_of).tolist()
    values  = e["weight"].astype(float).tolist()

    # 组装节点标签/hover/color
    labels = []
    hover  = []
    colors = None

    counts = None
    years  = None
    terms  = None
    topics = None

    if node_df is not None:
        meta = node_df.set_index(node_df["id"].astype(str))
        if "count" in meta.columns:
            counts = []
        if "year" in meta.columns:
            years = []
        if "top_terms" in meta.columns:
            terms = []
        if "topic" in meta.columns:
            topics = []

        for nid in node_ids:
            row = meta.loc[nid] if nid in meta.index else None
            y   = int(row["year"]) if (row is not None and "year" in row and pd.notna(row["year"])) else None
            tpc = int(row["topic"]) if (row is not None and "topic" in row and pd.notna(row["topic"])) else None
            tt  = str(row["top_terms"]) if (row is not None and "top_terms" in row and pd.notna(row["top_terms"])) else ""
            cnt = int(row["count"]) if (row is not None and "count" in row and pd.notna(row["count"])) else None

            if years is not None:  years.append(y)
            if topics is not None: topics.append(tpc)
            if terms is not None:  terms.append(tt)
            if counts is not None: counts.append(cnt)

        # 标签
        if args.label_mode == "keywords":
            for i, nid in enumerate(node_ids):
                y   = years[i]  if years  else None
                tt  = terms[i]  if terms  else ""
                label = (f"{y} | {short_terms(tt, args.label_topk)}") if y is not None else short_terms(tt, args.label_topk)
                labels.append(label if label else nid)
        elif args.label_mode == "year_topic":
            for i, nid in enumerate(node_ids):
                y   = years[i]  if years  else None
                tpc = topics[i] if topics else None
                label = f"{y}:{tpc}" if (y is not None and tpc is not None) else nid
                labels.append(label)
        else:  # id
            labels = node_ids[:]

        # hover
        for i, nid in enumerate(node_ids):
            y   = years[i]  if years  else None
            tpc = topics[i] if topics else None
            tt  = terms[i]  if terms  else ""
            cnt = counts[i] if counts else None
            hover_text = f"<b>ID</b>: {nid}"
            if y   is not None:  hover_text += f"<br><b>Year</b>: {y}"
            if tpc is not None:  hover_text += f"<br><b>Topic</b>: {tpc}"
            if cnt is not None:  hover_text += f"<br><b>Count</b>: {cnt}"
            if tt:               hover_text += f"<br><b>Top terms</b>: {short_terms(tt, min(args.label_topk, 6))}"
            labels.append("") if False else None  # 占位
            hover.append(hover_text)

        # 颜色：按 count 渐变
        if counts and any(c is not None for c in counts):
            cvals = np.array([c if c is not None else 0 for c in counts], dtype=float)
            # 归一化到 [0,1]，映射到 colorscale
            cmin, cmax = np.percentile(cvals, 5), np.percentile(cvals, 95)
            cmin = float(min(cmin, cmax-1e-6))
            cmax = float(cmax)
            norm = np.clip((cvals - cmin) / (cmax - cmin + 1e-9), 0, 1)
            # 用 Viridis colorscale 取色（Plotly 支持直接字符串，但这里我们构造 rgba）
            import matplotlib.cm as cm  # 若无 matplotlib，可改成固定色
            cmap = cm.get_cmap("viridis")
            colors = ["rgba(%d,%d,%d,0.9)" % tuple(int(255*x) for x in cmap(v)[:3]) for v in norm]
        else:
            colors = None

    else:
        # 没有 nodes.csv：简单 ID 标签 & 基础 hover
        labels = node_ids[:]
        hover  = [f"ID: {nid}" for nid in node_ids]
        colors = None

    # 链接 hover
    link_hover = []
    for s, t, w in zip(e["source"].astype(str), e["target"].astype(str), e["weight"].astype(float)):
        link_hover.append(f"{s} → {t}<br><b>weight</b>: {w:.3f}")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=14,
            line=dict(color="black", width=0.3),
            label=labels,
            color=colors if colors is not None else "rgba(120,120,200,0.85)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover
        ),
        link=dict(
            source=src_idx,
            target=tgt_idx,
            value=values,
            color="rgba(180,180,180,0.4)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover
        )
    )])

    fig.update_layout(
        title="Topic Evolution Sankey",
        font=dict(size=12),
        width=args.width,
        height=args.height
    )
    fig.write_html(args.out)
    print(f"[Saved] {args.out}")

if __name__ == "__main__":
    main()