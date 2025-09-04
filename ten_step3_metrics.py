#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEN Step 3 — 图指标 & 社群检测（可选 LLM 解读）
================================================
输入：
  - nodes_clustered.csv（Step 2 产物）：id, year, topic(=cluster_id), top_terms, count[, label_llm]
  - edges_clustered.csv（Step 2 产物）：source, target, weight

功能：
  1) 在聚合图上计算网络指标：degree / weighted_degree / betweenness / pagerank。
  2) 社群检测：优先 python-louvain（Leiden/Louvain 家族），若无则回落到 NetworkX 的 greedy_modularity。
  3) 将指标写回节点文件，输出：nodes_metrics.csv。
  4) 输出每个社群的概览：communities.csv（包含聚合关键词、年份范围等）。
  5) （可选）LLM 对每个社群生成 1 句说明 / 一个名称（--llm-explain）。

安装：
  pip install pandas networkx python-louvain openai

用法示例：
  python ten_step3_metrics.py \
    --nodes ./ten_out_step2/nodes_clustered.csv \
    --edges ./ten_out_step2/edges_clustered.csv \
    --outdir ./ten_out_step3

  # 可选：让 LLM 为社群命名/解释（OpenAI 兼容）
  python ten_step3_metrics.py \
    --nodes ./ten_out_step2/nodes_clustered.csv \
    --edges ./ten_out_step2/edges_clustered.csv \
    --outdir ./ten_out_step3 \
    --llm-explain --api openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY
"""
from __future__ import annotations
import os, sys, json, argparse, math
from typing import List, Dict
import pandas as pd
import networkx as nx

# Louvain（可选）
try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

# OpenAI 兼容 LLM（可选）
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def build_graph(nodes_csv: str, edges_csv: str) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    n = pd.read_csv(nodes_csv)
    e = pd.read_csv(edges_csv)
    # 标准化列名
    n.columns = [c.strip().lower() for c in n.columns]
    e.columns = [c.strip().lower() for c in e.columns]

    G = nx.Graph()
    for _, r in n.iterrows():
        G.add_node(str(r['id']))
    for _, r in e.iterrows():
        u, v = str(r['source']), str(r['target'])
        w = float(r.get('weight', 1.0))
        if G.has_edge(u, v):
            # 累加并行边
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G, n, e


def compute_metrics(G: nx.Graph) -> dict:
    # 无权度 & 加权度
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight='weight'))
    # 介数（带权，使用 weight 作为距离的反比：用 1/weight 作为长度）
    # 为避免 1/0，把权重最小值做下限
    import numpy as np
    W = [d.get('weight', 1.0) for _,_,d in G.edges(data=True)]
    wmin = max(1e-9, float(np.percentile(W, 5))) if W else 1.0
    length = {(u,v): 1.0/max(w, wmin) for u,v,w in [(u,v,d.get('weight',1.0)) for u,v,d in G.edges(data=True)]}
    nx.set_edge_attributes(G, {(u,v): {'length': L} for (u,v), L in length.items()})
    btw = nx.betweenness_centrality(G, weight='length', normalized=True)
    # PageRank（边权作转移概率权重）
    pr = nx.pagerank(G, weight='weight')
    return {'degree': deg, 'w_degree': wdeg, 'betweenness': btw, 'pagerank': pr}


def detect_communities(G: nx.Graph) -> dict:
    if community_louvain is not None:
        # 使用 Louvain（权重）
        parts = community_louvain.best_partition(G, weight='weight', resolution=1.0)
        return parts
    # 回退：NetworkX 贪婪模块度（不支持权重），对大图较慢
    comms = nx.algorithms.community.greedy_modularity_communities(G)
    parts = {}
    for cid, nodes in enumerate(comms):
        for u in nodes:
            parts[u] = cid
    return parts


def llm_explain_community(api: str, api_base: str|None, model: str, api_key_env: str,
                           rows: pd.DataFrame) -> tuple[str, str]:
    """返回 (label, summary)；若失败返回空串。"""
    if api != 'openai' or OpenAI is None:
        return '', ''
    key = os.getenv(api_key_env or 'OPENAI_API_KEY')
    if not key:
        return '', ''
    client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)
    # 聚合若干主题标签/关键词
    labs = [str(x) for x in rows.get('label_llm', []) if isinstance(x, str) and x.strip()]
    terms = [str(x) for x in rows.get('top_terms', []) if isinstance(x, str) and x.strip()]
    text = "\n".join((labs[:8] + terms[:12]))[:1500]
    prompt = (
        "你是文学主题分析助手。下面是一个社群中若干主题的名称与关键词。请：\n"
        "1) 给出该社群的一个‘母题名称’（≤6字），\n2) 用≤60字总结该社群的共同叙事或题材特征。\n"
        f"材料：\n{text}\n只返回 JSON：{{\"label\":\"…\", \"summary\":\"…\"}}"
    )
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2, max_tokens=200
        )
        ans = rsp.choices[0].message.content.strip().strip('`')
        if ans.lower().startswith('json'):
            ans = ans[4:].lstrip()
        data = json.loads(ans)
        return str(data.get('label','')).strip(), str(data.get('summary','')).strip()
    except Exception:
        return '', ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--edges', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--llm-explain', action='store_true')
    ap.add_argument('--api', default='openai')
    ap.add_argument('--api-base', default=None)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    G, n, e = build_graph(args.nodes, args.edges)
    metrics = compute_metrics(G)
    parts = detect_communities(G)

    # 写回节点
    n['degree'] = n['id'].map(metrics['degree']).fillna(0).astype(float)
    n['w_degree'] = n['id'].map(metrics['w_degree']).fillna(0).astype(float)
    n['betweenness'] = n['id'].map(metrics['betweenness']).fillna(0).astype(float)
    n['pagerank'] = n['id'].map(metrics['pagerank']).fillna(0).astype(float)
    n['community'] = n['id'].map(parts).fillna(-1).astype(int)

    out_nodes = os.path.join(args.outdir, 'nodes_metrics.csv')
    n.to_csv(out_nodes, index=False)

    # 社群概览
    recs = []
    for cid, grp in n.groupby('community'):
        if cid == -1:
            continue
        years = sorted(set(int(y) for y in grp['year'].dropna().astype(int).tolist()))
        label, summary = ('','')
        if args.llm_explain:
            label, summary = llm_explain_community(args.api, args.api_base, args.model, args.api_key_env, grp)
        # 关键词汇总（Top 10）
        toks = []
        for t in grp['top_terms'].fillna('').tolist():
            for x in str(t).replace('、',';').replace(',',';').split(';'):
                x = x.strip()
                if x:
                    toks.append(x)
        top_terms = ', '.join(pd.Series(toks).value_counts().head(10).index.tolist())
        recs.append({
            'community': cid,
            'size': len(grp),
            'year_min': min(years) if years else None,
            'year_max': max(years) if years else None,
            'label_llm_comm': label,
            'summary_llm_comm': summary,
            'top_terms_agg': top_terms
        })
    comm_df = pd.DataFrame.from_records(recs)
    out_comm = os.path.join(args.outdir, 'communities.csv')
    comm_df.to_csv(out_comm, index=False)

    print('[Saved]', out_nodes)
    print('[Saved]', out_comm)
    print('\n下一步建议：')
    print('1) 在 Gephi 里用 nodes_metrics.csv + edges_clustered.csv：节点大小=degree/pagerank，颜色=community。')
    print('2) 或者把 communities.csv 做成表格，挑选 size 大/ pagerank 高 的社群做质性解读。')

if __name__ == '__main__':
    main()
