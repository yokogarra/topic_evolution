#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic → TEN (Topic Evolution Network) — Step 1: LLM 标注主题
================================================================
给 nodes.csv 中的每个“年份:主题”节点生成：
- label_llm  （≤5 个词的简短主题名）
- summary_llm（≤60 字的说明）
并另存为 nodes_enriched.csv（默认覆盖到同目录）。

可选地，你可以提供每个节点的代表性文本（代表文档/句子）来提升标签质量：
- 通过 --reps-jsonl 传入 JSONL 文件，每行形如：
  {"id": "2013:9", "reps": ["……文本1……", "……文本2……"]}

使用方式
--------
基础：仅基于 top_terms 标注
    python bertopic_to_ten_llm.py \
        --nodes ./ten_out/nodes.csv \
        --api openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY

带代表文本：
    python bertopic_to_ten_llm.py \
        --nodes ./ten_out/nodes.csv \
        --reps-jsonl ./ten_out/reps.jsonl \
        --api openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY

兼容 OpenAI 风格的本地/自建推理（vLLM/Ollama 等），只要支持 /v1/chat/completions：
    python bertopic_to_ten_llm.py \
        --nodes ./ten_out/nodes.csv \
        --api openai --api-base http://localhost:8000/v1 --model my-model --api-key-env DUMMY

参数
----
--nodes           nodes.csv 路径（需要列：id, year, topic, top_terms[, count]）
--reps-jsonl      (可选) 节点→代表文本 的 JSONL 文件
--out             输出 CSV（默认与 nodes 同目录，名为 nodes_enriched.csv）
--api             LLM 提供方：openai（OpenAI 及其兼容端点）
--api-base        (可选) 自定义 API Base（OpenAI 兼容）
--model           模型名，如 gpt-4o-mini / gpt-4o / 自建模型名
--api-key-env     读取 API Key 的环境变量名（默认 OPENAI_API_KEY）
--rate-limit      每次调用之间的休眠秒数（默认 0.3）
--lang            提示词语言：zh/ja/en（默认 zh）

输出
----
在原有 nodes.csv 的基础上新增两列：label_llm, summary_llm。

注意
----
1) 若某些节点没有 top_terms，将回退为使用 id。
2) reps 文本会截断，总 token 控制较为温和（避免超长）。
3) 若调用失败，该节点置空并继续。
"""
from __future__ import annotations
import os, sys, json, time, argparse
from typing import Dict, Any, List
import pandas as pd
import unicodedata, re

# -----------------------------
# LLM 客户端（OpenAI 兼容）
# -----------------------------
try:
    from openai import OpenAI  # pip install openai>=1.0
except Exception:
    OpenAI = None

class LLMClient:
    def __init__(self, api: str, model: str, api_base: str|None, api_key_env: str):
        self.api = api
        self.model = model
        self.api_base = api_base
        self.api_key = os.getenv(api_key_env or "OPENAI_API_KEY")
        if api == "openai":
            if OpenAI is None:
                raise RuntimeError("openai 库未安装：pip install openai")
            kwargs = {}
            if api_base:
                kwargs["base_url"] = api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self.client = OpenAI(**kwargs)
        else:
            raise ValueError(f"不支持的 --api: {api}")

    def label_topic(self, lang: str, year: str, topic_id: str, top_terms: str, reps: List[str]|None) -> Dict[str, str]:
        # 轻裁剪代表文本
        reps = (reps or [])[:3]
        reps = [r.strip().replace("\n"," ")[:280] for r in reps if isinstance(r, str) and r.strip()]
        # 提示词
        if lang == "ja":
            sys_prompt = (
                "あなたは文学研究の助手です。与えられたキーワードと代表文から、テーマ名(5語以内)と60字以内の要約を出力してください。"
            )
            user_prompt = (
                f"年: {year}\nトピックID: {topic_id}\nキーワード: {top_terms}\n代表文:\n- " + "\n- ".join(reps) +
                "\nJSON だけを返してください: {\"label\": \"…\", \"summary\": \"…\"}"
            )
        elif lang == "en":
            sys_prompt = (
                "You are a literary research assistant. Given keywords and representative snippets, produce a concise topic name (≤5 words) and a ≤60-char summary."
            )
            user_prompt = (
                f"Year: {year}\nTopicID: {topic_id}\nKeywords: {top_terms}\nRepresentatives:\n- " + "\n- ".join(reps) +
                "\nReturn JSON only: {\"label\": \"…\", \"summary\": \"…\"}"
            )
        else:  # zh
            sys_prompt = (
                "你是文学研究助手。根据给定关键词与代表性文本，生成：1) 简短主题名(≤5词)；2) 60字以内说明。只输出 JSON。"
            )
            user_prompt = (
                f"年份: {year}\n主题ID: {topic_id}\n关键词: {top_terms}\n代表文本:\n- " + "\n- ".join(reps) +
                "\n仅返回 JSON: {\"label\": \"…\", \"summary\": \"…\"}"
            )

        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            txt = rsp.choices[0].message.content.strip()
            # 容错：裁掉围绕代码块
            if txt.startswith("```"):
                txt = txt.strip("`\n ")
                if txt.lower().startswith("json"):
                    txt = txt[4:].lstrip()
            data = json.loads(txt)
            label = str(data.get("label", "")).strip()
            summary = str(data.get("summary", "")).strip()
            return {"label_llm": label, "summary_llm": summary}
        except Exception as e:
            # 失败返回空
            return {"label_llm": "", "summary_llm": ""}

# -----------------------------
# 加载 reps.jsonl（可选）
# -----------------------------

def load_reps_jsonl(path: str|None) -> Dict[str, List[str]]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, List[str]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                nid = str(obj.get("id"))
                reps = obj.get("reps") or obj.get("docs") or obj.get("texts") or []
                if nid:
                    mapping[nid] = [str(x) for x in reps if isinstance(x, str)]
            except Exception:
                continue
    return mapping

SEP = re.compile(r"[;,，、|\s]+")

def normalize_terms_signature(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s.strip().lower())
    toks = [t for t in SEP.split(s) if t]
    # 去重但保留大致顺序（或改成 sorted(toks) 更激进）
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return ";".join(out)

# ---- 新增：加载/保存 cache ----
def load_cache(path: str|None) -> dict:
    if not path or not os.path.exists(path): return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}

def save_cache(path: str|None, cache: dict):
    if not path: return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# -----------------------------
# 主流程
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--reps-jsonl', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--api', default='openai', choices=['openai'])
    ap.add_argument('--api-base', default=None)
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--rate-limit', type=float, default=0.3)
    ap.add_argument('--lang', default='zh', choices=['zh','ja','en'])
    ap.add_argument('--dedup-key', default='top_terms', choices=['none','topic','top_terms'])
    ap.add_argument('--cache', default=None, help='保存/读取 LLM 结果缓存的 JSON 文件')
    ap.add_argument('--skip-existing', action='store_true', help='若节点已有 label_llm 则跳过')

    args = ap.parse_args()

    df = pd.read_csv(args.nodes)
    need_cols = {'id','year','topic','top_terms'}
    miss = need_cols - set(c.lower() for c in df.columns)
    # 容错：大小写统一
    df.columns = [c.strip().lower() for c in df.columns]
    if not need_cols.issubset(set(df.columns)):
        print(f"[Warn] nodes.csv 缺少列: {miss}. 将尽力使用已有列。")
    reps_map = load_reps_jsonl(args.reps_jsonl)

    client = LLMClient(api=args.api, model=args.model, api_base=args.api_base, api_key_env=args.api_key_env)

    labels, summaries = [], []
    for i, r in df.iterrows():
        nid  = str(r.get('id', ''))
        year = str(r.get('year', ''))
        terms= str(r.get('top_terms', nid))
        reps = reps_map.get(nid, [])
        out  = client.label_topic(args.lang, year, nid, terms, reps)
        labels.append(out['label_llm'])
        summaries.append(out['summary_llm'])
        if (i+1) % 20 == 0:
            print(f"[Info] labeled {i+1}/{len(df)}")
        time.sleep(max(0.0, args.rate_limit))

    df['label_llm'] = labels
    df['summary_llm'] = summaries

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.nodes)), 'nodes_enriched.csv')
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")

if __name__ == '__main__':
    main()
