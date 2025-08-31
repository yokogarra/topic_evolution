#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Evolution Network (TEN) minimal, reproducible template
===========================================================

适配你的数据结构：
- title（书名）
- story（简介内容）
- Year（投稿年，int）
- Group（大类，str，可选）
- Genre（小类，str，可选）
- global_point（综合评分，float/int，可选）
- fav_novel_cnt（收藏量，int，可选）

功能：
1) 按时间片（Year 或滑动窗口）独立建模主题（嵌入→聚类）。
2) 为每个主题计算：质心、Top-N关键词（c-TF-IDF）、原型文档、规模、聚类内一致性。
3) 相邻时间片主题对齐：混合相似度（质心余弦 / 关键词 Jaccard / 原型文档相似度）。
4) 阈值自适应（随机对照95分位）+ 主干一对一匹配（Hungarian），再补充分裂/合并边。
5) 事件标注：延续/分裂/合并/新生/消亡 + 漂移强度。
6) 输出：
   - nodes.csv, edges.csv（可导入 Gephi/RAWGraphs/Plotly Sankey）
   - sankey.csv（source,target,value,time_from,time_to）
   - graph.graphml（networkx 导出）
   - 生成可选的 Neo4j 导入脚本 neo4j_import.cypher

依赖（建议）：
    pip install pandas numpy scikit-learn sentence-transformers hdbscan networkx scipy tqdm regex

可选：
- 模型：'paraphrase-multilingual-MiniLM-L12-v2' 或 'gte-multilingual-base'（需本地可用）
- 你也可以替换为自己的 finetuned 多语模型目录

用法：
    python topic_evolution_network_template.py \
        --input data.csv \
        --textcol story \
        --yearcol Year \
        --titlecol title \
        --groupcol Group \
        --genrecol Genre \
        --scorecol global_point \
        --favcol fav_novel_cnt \
        --outdir ./ten_out \
        --model paraphrase-multilingual-MiniLM-L12-v2 \
        --min-cluster-size 30 --min-samples 10 \
        --time-window 1 --time-step 1

如果 Year 稀疏，可以改用 --time-window 2 --time-step 1（两年窗，逐年滑动）。
"""

from __future__ import annotations
import os
import math
import json
import argparse
import regex as re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import hdbscan
import networkx as nx
from scipy.optimize import linear_sum_assignment

# ----------------------------
# 工具函数
# ----------------------------

@dataclass
class TopicNode:
    time_id: int              # 时间片编号（索引）
    time_from: int            # 原始起始年份（窗口左边）
    time_to: int              # 原始结束年份（窗口右边）
    tid: int                  # 该时间片内的主题簇 id
    size: int
    coherence: float
    top_terms: List[str]
    centroid: np.ndarray
    proto_idx: List[int]      # 该片内原始 df 的 index（绝对索引）
    label: str                # 可读标签（由 top_terms 拼接）
    
    # 可选的外生属性（加权/平均）
    mean_score: Optional[float] = None
    sum_fav: Optional[float] = None


def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def sliding_year_bins(years: np.ndarray, window: int = 1, step: int = 1) -> List[Tuple[int,int]]:
    """返回滑动窗口的 [start, end]（闭区间）列表。"""
    y_min, y_max = int(years.min()), int(years.max())
    bins = []
    start = y_min
    while start <= y_max:
        end = start + window - 1
        bins.append((start, end))
        start += step
    return bins


def subset_by_year(df: pd.DataFrame, yearcol: str, y_from: int, y_to: int) -> pd.DataFrame:
    m = (df[yearcol] >= y_from) & (df[yearcol] <= y_to)
    return df.loc[m].copy()


def embed_texts(texts: List[str], model: SentenceTransformer, batch: int = 64) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 30, min_samples: int = 10) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(X)
    return labels


def compute_c_tf_idf(docs_by_topic: List[List[str]], max_features: int = 5000,
                       analyzer: str = 'char', ngram_range=(2,4),
                       stop_words=None, topk: int = 15,
                       tokenizer=None) -> List[List[str]]:
    """简化版 c-TF-IDF：把每个主题的文档拼接成一个大文档做 TF-IDF，再取每类的 Top terms。"""
    print(len(docs_by_topic))
    corpus = [" ".join(map(sanitize_text, docs)) for docs in docs_by_topic]
    if tokenizer is not None:
        # word analyzer using custom tokenizer (e.g., MeCab)
        vectorizer = TfidfVectorizer(max_features=max_features,
                                     analyzer='word',
                                     tokenizer=tokenizer,
                                     token_pattern=None,
                                     stop_words=stop_words)
    else:
        # char n-gram analyzer (robust for CJK without pre-tokenization)
        vectorizer = TfidfVectorizer(max_features=max_features,
                                     analyzer=analyzer,
                                     ngram_range=ngram_range,
                                     stop_words=stop_words)
    X = vectorizer.fit_transform(corpus)  # shape: n_topics x vocab
    terms = np.array(vectorizer.get_feature_names_out())
    top_terms_list = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        idx = np.argsort(row.toarray()[0])[::-1][:topk]
        top_terms_list.append(terms[idx].tolist())
    return top_terms_list


def topic_internal_coherence(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """简单一致性 = 簇内平均余弦相似度。"""
    coh = {}
    for k in sorted(set(labels)):
        if k == -1:
            continue
        idx = np.where(labels == k)[0]
        if len(idx) < 3:
            coh[k] = float('nan')
            continue
        sub = embeddings[idx]
        # 质心相似
        centroid = sub.mean(axis=0, keepdims=True)
        sim = cosine_similarity(sub, centroid)
        coh[k] = float(sim.mean())
    return coh


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    if u == 0:
        return 0.0
    return len(sa & sb) / u


def avg_proto_sim(proto_a: np.ndarray, proto_b: np.ndarray) -> float:
    if proto_a.size == 0 or proto_b.size == 0:
        return 0.0
    S = cosine_similarity(proto_a, proto_b)
    return float(S.mean())


def robust_cluster_embeddings(X: np.ndarray,
                              min_cluster_size: int = 30,
                              min_samples: int = 10):
    """
    先用 HDBSCAN；若全部噪声则降参重试；仍不行则用 KMeans 兜底。
    返回: labels, method_str
    """
    import hdbscan
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # 1) 第一次 HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean',
                                cluster_selection_method='leaf')  # leaf 更细粒度
    labels = clusterer.fit_predict(X)
    if np.any(labels != -1):
        return labels, f"hdbscan({min_cluster_size},{min_samples})"

    # 2) 降参重试（更小的 min_cluster_size/min_samples）
    n = len(X)
    mcs2 = max(10, int(0.02 * n))      # 至少 10，或 2% 样本
    ms2  = max(1,  int(0.01 * n))      # 至少 1
    clusterer2 = hdbscan.HDBSCAN(min_cluster_size=mcs2,
                                 min_samples=ms2,
                                 metric='euclidean',
                                 cluster_selection_method='leaf')
    labels2 = clusterer2.fit_predict(X)
    if np.any(labels2 != -1):
        return labels2, f"hdbscan({mcs2},{ms2})"

    # 3) 仍失败 → KMeans 兜底
    # 经验性选 k：5 ~ 50 之间，取 sqrt(n) 附近
    k_lo, k_hi = 5, min(50, max(6, int(np.sqrt(n))))
    best_labels, best_k, best_sil = None, None, -1.0
    for k in range(k_lo, max(k_lo+1, k_hi+1)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labs = km.fit_predict(X)
            # 至少产生两个簇才能算 silhouette
            if len(set(labs)) > 1:
                sil = silhouette_score(X, labs, metric='euclidean')
            else:
                sil = -1.0
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, labs
        except Exception:
            continue
    if best_labels is not None:
        print(f"[Warn] HDBSCAN failed; fallback to KMeans(k={best_k}, sil={best_sil:.3f})")
        return best_labels, f"kmeans({best_k})"

    # 实在不行：全部标为一个簇（避免后续空列表），但会给出警告
    print("[Warn] Clustering failed; assigning all to one cluster as a last resort.")
    return np.zeros(n, dtype=int), "single-cluster"


def build_topics_for_slice(df_slice: pd.DataFrame, embeddings: np.ndarray, titlecol: str, textcol: str,
                           scorecol: Optional[str], favcol: Optional[str],
                           min_cluster_size: int, min_samples: int, topk_terms: int = 15, tokenizer=None) -> List[TopicNode]:
    # labels = cluster_hdbscan(embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels, method_used = robust_cluster_embeddings(embeddings,
                                                    min_cluster_size=min_cluster_size,
                                                    min_samples=min_samples)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[Debug] HDBSCAN labels (value counts): {dict(zip(unique.tolist(), counts.tolist()))}")

    nodes: List[TopicNode] = []

    coh = topic_internal_coherence(embeddings, labels)

    # 聚类索引
    unique_labels = [k for k in sorted(set(labels)) if k != -1]
    docs_by_topic = []
    topic_indices = []

    for k in unique_labels:
        idx_local = np.where(labels == k)[0]  # 相对该切片的行号
        idx_abs = df_slice.index.values[idx_local]  # 绝对索引
        topic_indices.append(idx_abs)
        docs_by_topic.append(df_slice.loc[idx_abs, textcol].fillna("").astype(str).tolist())

    top_terms_list = compute_c_tf_idf(docs_by_topic, topk=topk_terms, tokenizer=tokenizer)

    for k, idx_abs, top_terms in zip(unique_labels, topic_indices, top_terms_list):
        sub_emb = embeddings[(df_slice.index.get_indexer(idx_abs))]
        centroid = sub_emb.mean(axis=0)
        # 原型文档：与质心最相近的前5篇
        sims = cosine_similarity(sub_emb, centroid.reshape(1, -1)).ravel()
        order = np.argsort(sims)[-5:][::-1]
        proto_abs = idx_abs[order]

        # 外生属性（均值/求和）
        mean_score = None
        sum_fav = None
        if scorecol and scorecol in df_slice.columns:
            mean_score = float(pd.to_numeric(df_slice.loc[idx_abs, scorecol], errors='coerce').mean())
        if favcol and favcol in df_slice.columns:
            sum_fav = float(pd.to_numeric(df_slice.loc[idx_abs, favcol], errors='coerce').sum())

        label = ", ".join(top_terms[:6])

        node = TopicNode(
            time_id=-1,   # 之后填
            time_from=-1,
            time_to=-1,
            tid=int(k),
            size=int(len(idx_abs)),
            coherence=float(coh.get(k, float('nan'))),
            top_terms=top_terms,
            centroid=centroid,
            proto_idx=proto_abs.tolist(),
            label=label,
            mean_score=mean_score,
            sum_fav=sum_fav,
        )
        nodes.append(node)
    return nodes


def pick_threshold_from_null(sim_matrix: np.ndarray, n_null: int = 2000, seed: int = 42) -> float:
    """用打乱列/行的方式形成随机相似度，取 95% 分位做阈值。"""
    rng = np.random.default_rng(seed)
    m, n = sim_matrix.shape
    null_vals = []
    for _ in range(min(n_null, m*n)):
        i = rng.integers(0, m)
        j = rng.integers(0, n)
        null_vals.append(sim_matrix[i, j])
    if not null_vals:
        return 0.0
    return float(np.quantile(null_vals, 0.95))


def max_weight_bipartite(sim: np.ndarray, tau: float) -> List[Tuple[int,int,float]]:
    """Hungarian 求最大权一对一匹配（将权重转负作为费用）。过滤 < tau 的边。"""
    if sim.size == 0:
        return []
    # 线性分配最小化，故将 -sim
    cost = 1.0 - sim  # 在 [0,1] 内更稳定
    r, c = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(r, c):
        w = sim[i, j]
        if w >= tau:
            pairs.append((int(i), int(j), float(w)))
    return pairs


def attach_extra_edges(sim: np.ndarray, tau: float, primary_pairs: List[Tuple[int,int,float]], delta: float = 0.1,
                       max_extra_per_node: int = 1) -> List[Tuple[int,int,float]]:
    """在主干匹配外，允许额外边：
    - 一对多（分裂）：某行 i 的第二强候选与第一强差距 < delta 且 >= tau
    - 多对一（合并）：某列 j 的第二强候选满足类似条件
    默认每个节点最多加 1 条额外边，避免图过密。
    """
    m, n = sim.shape
    used_rows = set(i for i,_,_ in primary_pairs)
    used_cols = set(j for _,j,_ in primary_pairs)

    extras = []

    # 行方向（分裂候选）
    for i in range(m):
        # 找到前两强
        order = np.argsort(sim[i])[::-1]
        if len(order) < 2:
            continue
        j1, j2 = order[0], order[1]
        w1, w2 = sim[i, j1], sim[i, j2]
        if w2 >= tau and (w1 - w2) < delta:
            # 避免重复与过密
            cnt = sum(1 for ii,jj,_ in extras if ii == i)
            if cnt < max_extra_per_node:
                extras.append((i, int(j2), float(w2)))

    # 列方向（合并候选）
    for j in range(n):
        order = np.argsort(sim[:, j])[::-1]
        if len(order) < 2:
            continue
        i1, i2 = order[0], order[1]
        w1, w2 = sim[i1, j], sim[i2, j]
        if w2 >= tau and (w1 - w2) < delta:
            cnt = sum(1 for ii,jj,_ in extras if jj == j)
            if cnt < max_extra_per_node:
                extras.append((int(i2), j, float(w2)))

    # 去重：避免与主干重复
    prim_set = {(i,j) for i,j,_ in primary_pairs}
    extras_unique = []
    for i,j,w in extras:
        if (i,j) not in prim_set and w >= tau:
            extras_unique.append((i,j,w))
    return extras_unique


def compute_similarity_matrix(nodes_a: List[TopicNode], nodes_b: List[TopicNode],
                              doc_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str,float]]:
    """混合相似度矩阵 + 各分量权重记录。"""
    alpha, beta, gamma = 0.6, 0.3, 0.1
    m, n = len(nodes_a), len(nodes_b)
    if m == 0 or n == 0:
        return np.zeros((m,n), dtype=float), {"alpha":alpha, "beta":beta, "gamma":gamma}

    # 预取原型嵌入
    def proto_vecs(node: TopicNode) -> np.ndarray:
        idx = node.proto_idx
        if len(idx) == 0:
            return np.zeros((0, doc_embeddings.shape[1]))
        # 绝对索引 → 在 doc_embeddings 的位置是一样的（原始 df 顺序编码）
        return doc_embeddings[idx]

    sim = np.zeros((m, n), dtype=float)
    for i, A in enumerate(nodes_a):
        for j, B in enumerate(nodes_b):
            s1 = float(np.dot(A.centroid, B.centroid))  # 已归一
            s2 = jaccard(A.top_terms, B.top_terms)
            s3 = avg_proto_sim(proto_vecs(A), proto_vecs(B))
            sim[i, j] = alpha*s1 + beta*s2 + gamma*s3
    return sim, {"alpha":alpha, "beta":beta, "gamma":gamma}


def label_events(pairs_primary: List[Tuple[int,int,float]], pairs_extra: List[Tuple[int,int,float]],
                 m: int, n: int) -> Dict[str, List[Tuple[int,int]]]:
    """根据主干匹配 + 附加边，标注延续/分裂/合并/新生/消亡（在切片级别）。"""
    # 图结构：a(i)->b(j)
    children = {i: [] for i in range(m)}
    parents = {j: [] for j in range(n)}

    for i,j,w in pairs_primary + pairs_extra:
        children[i].append(j)
        parents[j].append(i)

    cont, split, merge, newborn, dead = [], [], [], [], []

    for i in range(m):
        if len(children[i]) == 0:
            dead.append((i,))
        elif len(children[i]) == 1:
            cont.append((i, children[i][0]))
        else:
            # 一对多
            for j in children[i]:
                split.append((i, j))

    for j in range(n):
        if len(parents[j]) == 0:
            newborn.append((j,))
        elif len(parents[j]) >= 2:
            for i in parents[j]:
                merge.append((i, j))

    return {"continue": cont, "split": split, "merge": merge, "newborn": newborn, "dead": dead}


def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)


def export_graph(nodes_by_t: List[List[TopicNode]], edges_by_t: List[List[Tuple[int,int,int,int,float,str]]], outdir: str):
    """导出 nodes.csv, edges.csv, sankey.csv, graph.graphml, 以及一个简单的 Neo4j 导入脚本。"""
    ensure_outdir(outdir)

    # Nodes table
    rows = []
    for t, nodes in enumerate(nodes_by_t):
        for u, node in enumerate(nodes):
            rows.append({
                'node_id': f'{t}:{u}',
                'time_id': t,
                'time_from': node.time_from,
                'time_to': node.time_to,
                'tid': node.tid,
                'size': node.size,
                'coherence': node.coherence,
                'label': node.label,
                'top_terms': "; ".join(node.top_terms),
                'mean_score': node.mean_score,
                'sum_fav': node.sum_fav,
            })
    df_nodes = pd.DataFrame(rows)
    df_nodes.to_csv(os.path.join(outdir, 'nodes.csv'), index=False)

    # Edges table
    rows_e = []
    for t, edges in enumerate(edges_by_t):
        for (t_from, u_from, t_to, v_to, w, kind) in edges:
            rows_e.append({
                'source': f'{t_from}:{u_from}',
                'target': f'{t_to}:{v_to}',
                'time_from': t_from,
                'time_to': t_to,
                'weight': w,
                'kind': kind,
            })
    df_edges = pd.DataFrame(rows_e)
    df_edges.to_csv(os.path.join(outdir, 'edges.csv'), index=False)

    # Sankey (聚合到时间片层面，按主题边)
    df_edges[['value']] = 1
    df_edges[['source_id', 'target_id']] = df_edges[['source','target']]
    df_edges.to_csv(os.path.join(outdir, 'sankey.csv'), index=False)

    # GraphML 导出
    G = nx.DiGraph()
    for _, r in df_nodes.iterrows():
        G.add_node(r['node_id'], **{k: r[k] for k in df_nodes.columns if k not in ['node_id']})
    for _, r in df_edges.iterrows():
        G.add_edge(r['source'], r['target'], weight=float(r['weight']), kind=r['kind'])
    nx.write_graphml(G, os.path.join(outdir, 'graph.graphml'))

    nodes_path = os.path.abspath(os.path.join(outdir, 'nodes.csv')).replace("\\", "/")
    edges_path = os.path.abspath(os.path.join(outdir, 'edges.csv')).replace("\\", "/")

    # Neo4j 导入脚本（简化）
    cypher = f"""
// Load nodes
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///{nodes_path}' AS row
MERGE (t:Topic {{id: row.node_id}})
SET t.time_id = toInteger(row.time_id),
    t.time_from = toInteger(row.time_from),
    t.time_to   = toInteger(row.time_to),
    t.tid       = toInteger(row.tid),
    t.size      = toInteger(row.size),
    t.coherence = toFloat(row.coherence),
    t.label     = row.label,
    t.top_terms = row.top_terms,
    t.mean_score = CASE WHEN row.mean_score = '' THEN null ELSE toFloat(row.mean_score) END,
    t.sum_fav    = CASE WHEN row.sum_fav = '' THEN null ELSE toFloat(row.sum_fav) END;

// Load edges
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///{edges_path}' AS row
MATCH (a:Topic {{id: row.source}})
MATCH (b:Topic {{id: row.target}})
MERGE (a)-[e:EVOLVE {{kind: row.kind}}]->(b)
SET e.weight = toFloat(row.weight),
    e.time_from = toInteger(row.time_from),
    e.time_to   = toInteger(row.time_to);
"""
    with open(os.path.join(outdir, 'neo4j_import.cypher'), 'w', encoding='utf-8') as f:
        f.write(cypher)


# ----------------------------
# 主流程
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV 路径')
    ap.add_argument('--outdir', default='./ten_out')

    ap.add_argument('--textcol', default='story')
    ap.add_argument('--titlecol', default='title')
    ap.add_argument('--yearcol', default='Year')
    ap.add_argument('--groupcol', default='Group')
    ap.add_argument('--genrecol', default='Genre')
    ap.add_argument('--scorecol', default='global_point')
    ap.add_argument('--favcol', default='fav_novel_cnt')

    ap.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--batch', type=int, default=64)

    ap.add_argument('--time-window', type=int, default=1, help='滑动窗口：年数')
    ap.add_argument('--time-step', type=int, default=1, help='滑动步长：年数')
    ap.add_argument('--min-docs-per-slice', type=int, default=150)

    ap.add_argument('--min-cluster-size', type=int, default=15)
    ap.add_argument('--min-samples', type=int, default=5)
    ap.add_argument('--topk-terms', type=int, default=15)

    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--delta', type=float, default=0.1, help='附加边一二强差距阈值')
    ap.add_argument('--max-extra-per-node', type=int, default=1)
    ap.add_argument('--use-fugashi', action='store_true',
                    help='Use fugashi (MeCab) for Japanese tokenization in TF-IDF keywords.')
    
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.input)

    # 基本清洗
    for c in [args.textcol, args.titlecol, args.groupcol, args.genrecol]:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str).map(sanitize_text)
    if args.yearcol not in df.columns:
        raise ValueError('Year 列缺失！请指定 --yearcol')

    # 预计算所有文本嵌入（用于原型相似度 & 质心相似度）
    print('[Info] Loading embedding model:', args.model)
    model = SentenceTransformer(args.model)

    print('[Info] Encoding all documents...')
    all_texts = df[args.textcol].fillna('').astype(str).tolist()
    all_embeddings = embed_texts(all_texts, model, batch=args.batch)  # shape (N, D), L2 normalized

    # 时间片划分
    years = pd.to_numeric(df[args.yearcol], errors='coerce').astype('Int64')
    df = df[~years.isna()].copy()
    df[args.yearcol] = years.astype(int)

    bins = sliding_year_bins(df[args.yearcol].values, window=args.time_window, step=args.time_step)
    print('[Info] Year windows:', bins)

    # Optional: Japanese tokenizer via fugashi
    ja_tokenizer = None
    if args.use_fugashi:
        try:
            from fugashi import Tagger
            _tagger = Tagger()  # uses unidic-lite by default
            def tokenize_ja(text: str):
                return [w.surface for w in _tagger(text)]

            # 粗口径停用词（你可继续补充）
            JA_STOPWORDS = set("の は に を が で と も から まで など そして しかし また その この あの する なる いる ある".split())
            PUNCT_RE = re.compile(r"[。、・「」『』（）()［］\\[\\]【】〈〉《》…—--]")

            def tokenize_ja_content(text: str):
                if not isinstance(text, str):
                    return []
                text = PUNCT_RE.sub(" ", text)
                toks = []
                for w in _tagger(text):
                    pos = getattr(w.feature, "pos1", "")
                    if pos in {"助詞","助動詞","記号","接続詞","連体詞","フィラー","感動詞"}:
                        continue
                    lemma = getattr(w.feature, "lemma", None)
                    if not lemma or lemma == "*" or lemma.strip() == "":
                        lemma = w.surface
                    lemma = lemma.strip()
                    if not lemma or lemma in JA_STOPWORDS:
                        continue
                    if len(lemma) == 1 and re.match(r"[\u3040-\u309F\u30A0-\u30FF]", lemma):
                        continue
                    toks.append(lemma)
                return toks

            # ja_tokenizer = tokenize_ja
            ja_tokenizer = tokenize_ja_content
            print('[Info] Using fugashi (MeCab) tokenizer for TF-IDF keywords.')
        except Exception as e:
            print('[Warn] Failed to load fugashi; falling back to char n-grams. Error:', e)

    nodes_by_t: List[List[TopicNode]] = []
    slices_indices: List[np.ndarray] = []  # 每片的绝对索引

    for t, (y_from, y_to) in enumerate(bins):
        df_t = subset_by_year(df, args.yearcol, y_from, y_to)
        if len(df_t) < args.min_docs_per_slice:
            print(f'[Warn] slice {t} ({y_from}-{y_to}) has only {len(df_t)} docs; skip.')
            nodes_by_t.append([])
            slices_indices.append(df_t.index.values)
            continue

        idx_abs = df_t.index.values
        X_t = all_embeddings[idx_abs]

        print(f'[Info] Slice {t}: {y_from}-{y_to}, docs={len(df_t)} -> clustering...')
        nodes_t = build_topics_for_slice(
            df_slice=df_t,
            embeddings=X_t,
            titlecol=args.titlecol,
            textcol=args.textcol,
            scorecol=args.scorecol if args.scorecol in df_t.columns else None,
            favcol=args.favcol if args.favcol in df_t.columns else None,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            topk_terms=args.topk_terms,
            tokenizer=ja_tokenizer,
        )
        # 回填时间信息
        for nd in nodes_t:
            nd.time_id = t
            nd.time_from = y_from
            nd.time_to = y_to
        nodes_by_t.append(nodes_t)
        slices_indices.append(idx_abs)

    # 相邻切片对齐 & 事件标注
    edges_by_t: List[List[Tuple[int,int,int,int,float,str]]] = []

    for t in range(len(bins)-1):
        A = nodes_by_t[t]
        B = nodes_by_t[t+1]
        if len(A) == 0 or len(B) == 0:
            edges_by_t.append([])
            continue

        print(f'[Info] Matching slice {t} -> {t+1} ...')
        sim, weights = compute_similarity_matrix(A, B, all_embeddings)
        # 阈值：基于当前矩阵的随机对照分位
        tau = pick_threshold_from_null(sim, n_null=2000)
        primary = max_weight_bipartite(sim, tau)
        extras = attach_extra_edges(sim, tau, primary, delta=args.delta, max_extra_per_node=args.max_extra_per_node)

        # 事件标注（仅在切片层面，后续可沿谱系整合）
        events = label_events(primary, extras, m=len(A), n=len(B))

        # 汇总为边列表： (t, u_from, t+1, v_to, weight, kind)
        edges = []
        prim_set = {(i,j):w for i,j,w in primary}
        for i,j in events['continue']:
            w = prim_set.get((i,j), float(sim[i,j]))
            edges.append((t, i, t+1, j, float(w), 'continue'))
        for i,j in events['split']:
            edges.append((t, i, t+1, j, float(sim[i,j]), 'split'))
        for i,j in events['merge']:
            edges.append((t, i, t+1, j, float(sim[i,j]), 'merge'))
        # 新生/消亡 不作为边导出，但可用于统计

        edges_by_t.append(edges)

    # 导出
    export_graph(nodes_by_t, edges_by_t, args.outdir)

    print('[Done] Exported to:', args.outdir)
    print(' - nodes.csv (节点属性)')
    print(' - edges.csv (包含 kind=continue/split/merge)')
    print(' - sankey.csv (可用于桑基图)')
    print(' - graph.graphml (可在 Gephi/Graphistry 打开)')
    print(' - neo4j_import.cypher (简化导入脚本)')


if __name__ == '__main__':
    main()
