#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic + Topic Evolution Network (TEN)
========================================
输入: CSV (必须包含 story, Year 列, 可选 fav_novel_cnt, global_point)
输出: nodes.csv, edges.csv, sankey.csv, graph.graphml

用法:
    python bertopic_to_ten.py --input Narou_rensai.csv --outdir ./ten_out
可选:
    --min-docs-per-year 50   # 每年最少文档数
"""
import os, re, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech, KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from fugashi import Tagger
    tagger = Tagger()
except Exception:
    tagger = None

# ------------------------
# Tokenizer for Japanese
# ------------------------
JA_STOPWORDS = set("の は に を が で と も から まで など そして しかし また その この あの する なる いる ある".split())
PUNCT = re.compile(r"[。、・「」『』（）()［］\[\]【】〈〉《》…—\-.,!?]")

def ja_tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = PUNCT.sub(" ", text)
    if tagger is None:
        return [t for t in re.split(r"\s+", text) if t]
    toks = []
    for w in tagger(text):
        pos = getattr(w.feature, "pos1", "")
        if pos in {"助詞","助動詞","記号","接続詞","連体詞","フィラー","感動詞"}:
            continue
        lemma = getattr(w.feature, "lemma", None) or w.surface
        lemma = lemma.strip()
        if not lemma or lemma in JA_STOPWORDS:
            continue
        toks.append(lemma)
    return toks

# ------------------------
# Build BERTopic model
# ------------------------
def build_model():
    vectorizer = CountVectorizer(
        tokenizer=ja_tokenize,
        token_pattern=None,
        ngram_range=(1,2),
        min_df=3, max_df=0.5
    )

    rep = [KeyBERTInspired(top_n_words=20),
           MaximalMarginalRelevance(diversity=0.3)]

    emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        embedding_model=emb_model,
        ctfidf_model=ctfidf,
        vectorizer_model=vectorizer,
        representation_model=rep,
        language="japanese",
        verbose=True,
        calculate_probabilities=True
    )
    return model

# ------------------------
# Evolution alignment
# ------------------------
def align_topics(year_topics, threshold=0.6):
    edges = []
    years = sorted(year_topics.keys())
    for i in range(len(years)-1):
        y1, y2 = years[i], years[i+1]
        nodes1, nodes2 = year_topics[y1], year_topics[y2]
        if not nodes1 or not nodes2:
            continue
        emb1 = np.vstack([n['embedding'] for n in nodes1])
        emb2 = np.vstack([n['embedding'] for n in nodes2])
        sims = cosine_similarity(emb1, emb2)
        for i1, n1 in enumerate(nodes1):
            for i2, n2 in enumerate(nodes2):
                w = sims[i1, i2]
                if w >= threshold:
                    edges.append({
                        'source': f"{y1}:{n1['topic']}",
                        'target': f"{y2}:{n2['topic']}",
                        'kind': 'continue',
                        'weight': w
                    })
    return edges

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--min-docs-per-year', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    # Debug: Check input DataFrame
    print('[Debug] Input DataFrame info:')
    print(df.info())
    print('[Debug] First few rows of the DataFrame:')
    print(df.head())

    df = df.dropna(subset=['story', 'Year'])
    print('[Debug] DataFrame after dropna:')
    print(df.info())

    # Debug: Check unique years
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    print('[Debug] Unique years in the data:', df['Year'].unique())

    # Debug: Check document count per year
    for y in sorted(df['Year'].unique()):
        print(f'[Debug] Year {y}: {len(df[df["Year"] == y])} documents')

    docs = df['story'].astype(str).tolist()
    years = df['Year'].astype(int).tolist()

    # Embeddings
    # emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # embeddings = emb_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    # Fit BERTopic
    topic_model = build_model()
    topics, probs = topic_model.fit_transform(docs)

    # Debug: Check topics after fit_transform
    print('[Debug] Topics:', set(topics))

    # Topics over time
    topics_over_time = topic_model.topics_over_time(docs, years, nr_bins=len(set(years)))

    # Build year->topic nodes
    year_topics = {}
    for y in sorted(set(years)):
        nodes = []
        df_y = topics_over_time[topics_over_time['Timestamp'].astype(int) == y]
        for _, row in df_y.iterrows():
            t = int(row['Topic'])
            if t == -1:
                continue
            emb = topic_model.topic_embeddings_[t]
            nodes.append({
                'year': y,
                'topic': t,
                'top_terms': row['Words'],
                'count': row['Frequency'],
                'embedding': emb
            })
        if len(nodes) >= args.min_docs_per_year:
            year_topics[y] = nodes

    # Debug: Check year_topics content
    for y, nodes in year_topics.items():
        print(f'[Debug] Year {y}: {len(nodes)} nodes')

    # Build nodes.csv
    all_nodes = []
    for y, nodes in year_topics.items():
        for n in nodes:
            all_nodes.append({
                'id': f"{y}:{n['topic']}",
                'year': y,
                'topic': n['topic'],
                'top_terms': n['top_terms'],
                'count': n['count']
            })
    pd.DataFrame(all_nodes).to_csv(os.path.join(args.outdir,'nodes.csv'), index=False)

    # Debug: Check all_nodes and edges before export
    print('[Debug] All nodes:', all_nodes[:5])
    print('[Debug] Edges:', edges[:5])

    # Build edges.csv
    edges = align_topics(year_topics, threshold=0.6)
    pd.DataFrame(edges).to_csv(os.path.join(args.outdir,'edges.csv'), index=False)

    # Sankey
    sankey = [{'source':e['source'], 'target':e['target'], 'value':e['weight']} for e in edges]
    pd.DataFrame(sankey).to_csv(os.path.join(args.outdir,'sankey.csv'), index=False)

    # GraphML
    with open(os.path.join(args.outdir,'graph.graphml'),'w',encoding='utf-8') as f:
        f.write("<?xml version='1.0' encoding='UTF-8'?>\n")
        f.write("<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>\n")
        f.write("<graph edgedefault='directed'>\n")
        for n in all_nodes:
            f.write(f"<node id='{n['id']}'><data key='label'>{n['top_terms']}</data></node>\n")
        for e in edges:
            f.write(f"<edge source='{e['source']}' target='{e['target']}'><data key='weight'>{e['weight']}</data></edge>\n")
        f.write("</graph></graphml>")

    print("[Done] Exported nodes.csv, edges.csv, sankey.csv, graph.graphml")

if __name__ == '__main__':
    main()
