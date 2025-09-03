#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Sankey Diagram from edges.csv (TEN output)
===============================================
使用 Plotly 将 `edges.csv` 可视化为桑基图。

输入: edges.csv (包含 source, target, weight 列)
输出: sankey_plot.html (交互式桑基图)

用法:
    python plot_sankey.py --edges ./ten_out/edges.csv --out sankey_plot.html --short-labels
"""
import argparse
import pandas as pd
import plotly.graph_objects as go

def shorten_label(label: str, max_terms: int = 3) -> str:
    """将节点标签缩短为 年份:主题号 + 前几个关键词"""
    if ":" in label:
        # 格式: year:topic
        return label
    return label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--out', default='sankey_year.html')
    ap.add_argument('--short-labels', action='store_true', help='只显示年份:topic 或 年份+前几个关键词')
    args = ap.parse_args()

    df = pd.read_csv(args.edges)
    if not {'source','target','weight'}.issubset(df.columns):
        raise ValueError("edges.csv 必须包含 source,target,weight 列")

    # 构建节点列表
    labels = pd.unique(df[['source','target']].values.ravel()).tolist()
    if args.short_labels:
        labels_display = [shorten_label(l) for l in labels]
    else:
        labels_display = labels
    label_to_id = {lab:i for i,lab in enumerate(labels)}

    # 构建 link 数据
    sources = df['source'].map(label_to_id)
    targets = df['target'].map(label_to_id)
    values = df['weight']

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels_display,
            color="rgba(31,119,180,0.8)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(200,200,200,0.4)"
        )
    )])

    fig.update_layout(title_text="Topic Evolution Network — Sankey", font_size=12)
    fig.write_html(args.out, include_plotlyjs='cdn')
    print(f"[Saved] {args.out}")

if __name__ == '__main__':
    main()
