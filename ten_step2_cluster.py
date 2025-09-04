#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEN Step 2 — 跨年主题归并（母题聚合）
=================================
输入：
  - nodes.csv（或 nodes_enriched.csv）：列需含 id, year, topic, top_terms[, count, label_llm, summary_llm]
  - edges.csv：source, target, weight[, kind]

做什么：
  1) 选定每个节点的“文本签名”（优先 label_llm，其次 top_terms），用句向量表征；
  2) 以相似度阈值构造“同母题”图并连通分量聚类 ⇒ 得到 cluster_id；
  3) 聚合输出：
     - id→cluster 映射：node_members.csv
     - 母题簇概览：clusters.csv（cluster_id, size, years, label, summary, keywords）
     - 聚合后的年度节点：nodes_clustered.csv（用于桑基，可与 edges_clustered.csv 搭配）
     - 聚合后的年度跨年边：edges_clustered.csv（sum 或 max 权重）

可选：
  - --llm-confirm：对临界相似度对（介于阈值±窗口）调用 LLM 二次判断同母题（需 openai 兼容接口），带缓存。

安装：
  pip install pandas numpy sentence-transformers scikit-learn networkx openai

用法示例：
  python ten_step2_cluster.py \
    --nodes ./ten_out/nodes_enriched.csv \
    --edges ./ten_out/edges.csv \
    --embed-model paraphrase-multilingual-MiniLM-L12-v2 \
    --sim-thresh 0.78 --min-size 2 --agg weight_sum \
    --outdir ./ten_out_step2

  # 开启 LLM 边界确认（可选）
  python ten_step2_cluster.py \
    --nodes ./ten_out/nodes_enriched.csv --edges ./ten_out/edges.csv \
    --embed-model paraphrase-multilingual-MiniLM-L12-v2 \
    --sim-thresh 0.78 --confirm-window 0.03 --llm-confirm \
    --api openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY --cache ./ten_out_step2/llm_pairs.json
"""
from __future__ import annotations
import os, sys, json, argparse, math
from typing import List, Dict, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ------------------------- Utils -------------------------
SEP = "[;,，、|\s]+"

def norm_terms(s: str) -> str:
    import re, unicodedata
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s.strip().lower())
    toks = [t for t in re.split(SEP, s) if t]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return ";".join(out)

# -------------------- LLM (optional) ---------------------
try:
    from openai import OpenAI  # >=1.0
except Exception:
    OpenAI = None

class LLM:
    def __init__(self, enabled: bool, api: str, model: str, api_base: str|None, api_key_env: str, cache_path: str|None):
        self.enabled = enabled
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, 'r', encoding='utf-8'))
            except Exception:
                self.cache = {}
        if enabled:
            if api != 'openai':
                raise ValueError('--api 仅支持 openai 兼容')
            if OpenAI is None:
                raise RuntimeError('openai 库未安装: pip install openai')
            kwargs = {}
            if api_base:
                kwargs['base_url'] = api_base
            key = os.getenv(api_key_env or 'OPENAI_API_KEY')
            if key:
                kwargs['api_key'] = key
            self.client = OpenAI(**kwargs)
            self.model = model

    def _save(self):
        if not self.cache_path: return
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)

    def same_cluster(self, text_a: str, text_b: str) -> bool:
        if not self.enabled:
            return False
        key = f"{text_a}\n###\n{text_b}"
        if key in self.cache:
            return bool(self.cache[key])
        prompt = (
            "你是文学主题归并助手。给定两个主题描述（关键词/标签），判断它们是否为同一母题。"
            "若基本等价或同一语义族，返回 JSON {\"same\": true}，否则 {\"same\": false}。\n"
            f"A: {text_a}\nB: {text_b}\n只返回 JSON。"
        )
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=10
            )
            txt = rsp.choices[0].message.content.strip().strip('`')
            if txt.lower().startswith('json'):
                txt = txt[4:].lstrip()
            data = json.loads(txt)
            same = bool(data.get('same', False))
        except Exception:
            same = False
        self.cache[key] = same
        self._save()
        return same

# ---------------- Sentence Embedding ---------------------
from sentence_transformers import SentenceTransformer

def embed_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

# --------------------- Core logic ------------------------

def choose_sig(row: pd.Series, prefer: str) -> str:
    if prefer == 'llm' and 'label_llm' in row and isinstance(row['label_llm'], str) and row['label_llm'].strip():
        return row['label_llm'].strip()
    return norm_terms(row.get('top_terms', ''))


def build_clusters(nodes: pd.DataFrame, sim_thresh: float, confirm_window: float, llm: LLM,
                   model_name: str, prefer: str = 'llm') -> Tuple[pd.Series, pd.DataFrame]:
    # 1) 为每个唯一签名建立索引
    sigs = nodes.apply(lambda r: choose_sig(r, prefer), axis=1)
    sigs = sigs.fillna('')
    uniq_sigs, inv = np.unique(sigs.values, return_inverse=True)

    # 2) 向量化唯一签名
    emb = embed_texts(list(uniq_sigs), model_name)

    # 3) ANN 近邻 + 阈值图
    nn = NearestNeighbors(metric='cosine', algorithm='auto')
    nn.fit(emb)
    k = min(30, len(uniq_sigs))
    dist, ind = nn.kneighbors(emb, n_neighbors=k)
    # 余弦相似度 = 1 - 距离
    sims = 1.0 - dist

    G = nx.Graph()
    G.add_nodes_from(range(len(uniq_sigs)))

    low = max(0.0, sim_thresh - confirm_window)
    high = min(1.0, sim_thresh + confirm_window) if confirm_window > 0 else sim_thresh

    for i in range(len(uniq_sigs)):
        for j_pos in range(1, k):  # 跳过自身 j_pos=0
            j = ind[i, j_pos]
            if j <= i:  # 无向图去重
                continue
            s = sims[i, j_pos]
            if s >= high:
                G.add_edge(i, j, w=float(s), why='sim')
            elif s >= low and confirm_window > 0 and llm.enabled:
                a, b = uniq_sigs[i], uniq_sigs[j]
                if llm.same_cluster(a, b):
                    G.add_edge(i, j, w=float(s), why='llm')

    # 4) 连通分量作为 cluster
    comp = list(nx.connected_components(G))
    cid_of = {}
    for cid, nodeset in enumerate(sorted(comp, key=lambda c: -len(c))):
        for u in nodeset:
            cid_of[u] = cid

    # 5) 把唯一签名映射回每个节点
    cluster_series = pd.Series([cid_of.get(inv[i], i) for i in range(len(inv))], index=nodes.index, name='cluster_id')

    # 6) 产出 clusters 概览表
    groups = nodes.assign(cluster_id=cluster_series).groupby('cluster_id')
    recs = []
    for cid, g in groups:
        years = sorted(set(int(y) for y in g['year'].dropna().astype(int).tolist())) if 'year' in g else []
        label = ''
        if 'label_llm' in g and g['label_llm'].notna().any():
            label = g['label_llm'].dropna().value_counts().index.tolist()[0]
        if not label:
            # 合并 top_terms 前 5 个词
            toks = []
            for t in g['top_terms'].fillna('').tolist():
                toks += norm_terms(t).split(';')
            label = ', '.join([t for t,_ in pd.Series(toks).value_counts().head(5).items()]) if toks else ''
        summary = ''
        if 'summary_llm' in g and g['summary_llm'].notna().any():
            summary = g['summary_llm'].dropna().value_counts().index.tolist()[0]
        recs.append({
            'cluster_id': cid,
            'size': len(g),
            'years': ';'.join(map(str, years)),
            'year_min': min(years) if years else None,
            'year_max': max(years) if years else None,
            'label': label,
            'summary': summary,
            'keywords': norm_terms(';'.join(g['top_terms'].fillna('').tolist()))[:300]
        })
    clusters_df = pd.DataFrame.from_records(recs)

    return cluster_series, clusters_df


def aggregate_for_sankey(nodes: pd.DataFrame, edges: pd.DataFrame, cluster_series: pd.Series,
                          agg: str = 'weight_sum') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把原 nodes/edges 聚合为 (year, cluster_id) 级别的节点/边。"""
    nodes2 = nodes.copy()
    nodes2['cluster_id'] = cluster_series

    # 年度 cluster 节点：id 格式 "{year}:C{cid}"
    def mk_id(y, c):
        return f"{int(y)}:C{int(c)}"

    g_nodes = nodes2.groupby(['year','cluster_id'], as_index=False).agg({
        'count': 'sum' if 'count' in nodes2.columns else 'size',
        'top_terms': lambda s: norm_terms(';'.join(map(str, s)))[:300],
        'label_llm': lambda s: s.dropna().value_counts().index.tolist()[0] if s.notna().any() else ''
    })
    g_nodes['id'] = [mk_id(y,c) for y,c in zip(g_nodes['year'], g_nodes['cluster_id'])]
    g_nodes['topic'] = g_nodes['cluster_id']  # 用 cluster 显示
    g_nodes = g_nodes[['id','year','topic','top_terms','count','label_llm']]

    # 聚合边：把原来的 (id_i -> id_j) 折叠为 (year_i:Ci -> year_j:Cj)
    edges2 = edges.copy()
    m = nodes2[['id','year','cluster_id']].set_index('id')
    def map_node(nid):
        try:
            row = m.loc[str(nid)]
            return mk_id(int(row['year']), int(row['cluster_id']))
        except Exception:
            return None
    edges2['S'] = edges2['source'].map(map_node)
    edges2['T'] = edges2['target'].map(map_node)
    edges2 = edges2.dropna(subset=['S','T'])

    if agg == 'weight_max':
        g_edges = edges2.groupby(['S','T'], as_index=False)['weight'].max()
    else:  # weight_sum
        g_edges = edges2.groupby(['S','T'], as_index=False)['weight'].sum()

    g_edges = g_edges.rename(columns={'S':'source','T':'target'})

    return g_nodes, g_edges

# --------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--edges', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--embed-model', default='paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--prefer', default='llm', choices=['llm','terms'], help='签名优先级：llm=label_llm，terms=top_terms')
    ap.add_argument('--sim-thresh', type=float, default=0.8)
    ap.add_argument('--confirm-window', type=float, default=0.0, help='阈值窗口；>0 时可配合 --llm-confirm')
    ap.add_argument('--llm-confirm', action='store_true')
    ap.add_argument('--api', default='openai')
    ap.add_argument('--api-base', default=None)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--agg', default='weight_sum', choices=['weight_sum','weight_max'])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    # 列名小写化
    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]

    llm = LLM(enabled=args.llm_confirm, api=args.api, model=args.model, api_base=args.api_base,
              api_key_env=args.api_key_env, cache_path=args.cache)

    cluster_series, clusters_df = build_clusters(nodes, args.sim_thresh, args.confirm_window, llm,
                                                 model_name=args.embed_model, prefer=('llm' if args.prefer=='llm' else 'terms'))

    # 映射表
    members = pd.DataFrame({'id': nodes['id'], 'cluster_id': cluster_series})
    members.to_csv(os.path.join(args.outdir, 'node_members.csv'), index=False)

    # 聚合输出	
    nodes_clustered, edges_clustered = aggregate_for_sankey(nodes, edges, cluster_series, agg=args.agg)
    nodes_clustered.to_csv(os.path.join(args.outdir, 'nodes_clustered.csv'), index=False)
    edges_clustered.to_csv(os.path.join(args.outdir, 'edges_clustered.csv'), index=False)

    clusters_df.to_csv(os.path.join(args.outdir, 'clusters.csv'), index=False)

    print('[Saved]', os.path.join(args.outdir, 'node_members.csv'))
    print('[Saved]', os.path.join(args.outdir, 'nodes_clustered.csv'))
    print('[Saved]', os.path.join(args.outdir, 'edges_clustered.csv'))
    print('[Saved]', os.path.join(args.outdir, 'clusters.csv'))
    print('\n下一步：可直接用 plot_sankey_plus.py 画聚合后的桑基图：')
    print(f"python plot_sankey_plus.py --edges {os.path.join(args.outdir,'edges_clustered.csv')} --nodes {os.path.join(args.outdir,'nodes_clustered.csv')} --out {os.path.join(args.outdir,'sankey_clustered.html')} --label-mode auto --label-topk 3 --min-weight 0.7")

if __name__ == '__main__':
    main()
