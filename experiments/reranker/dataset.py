"""
Carga de datasets BEIR y first-stage retrieval (BM25 / dense).
"""

import os
import re
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path

import dspy
from .signatures import DocumentCandidate


BEIR_DATASETS = {
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "fiqa": "BeIR/fiqa",
    "trec-covid": "BeIR/trec-covid",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
}


def load_beir_dataset(
    dataset_name: str,
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Carga un dataset BEIR. Devuelve (corpus, queries, qrels)."""
    if dataset_name not in BEIR_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {list(BEIR_DATASETS.keys())}")
    
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.util import download_and_unzip
        
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        out_dir = cache_dir or os.path.join(os.getcwd(), "datasets")
        data_dir = download_and_unzip(url, out_dir)
        
        loader = GenericDataLoader(data_folder=data_dir)
        corpus, queries, qrels = loader.load(split=split)
        return corpus, queries, qrels
    except ImportError:
        print("beir not found, trying HuggingFace datasets...")
        return _load_from_hf(BEIR_DATASETS[dataset_name], split)


def _load_from_hf(hf_dataset: str, split: str = "test"):
    from datasets import load_dataset
    
    queries_ds = load_dataset(hf_dataset, "query", split=split)
    corpus_ds = load_dataset(hf_dataset, "corpus", split="train")
    qrels_ds = load_dataset(hf_dataset, "qrels", split=split)
    
    queries = {q["_id"]: q["text"] for q in queries_ds}
    corpus = {c["_id"]: c["text"] for c in corpus_ds}
    qrels = {}
    for q in qrels_ds:
        qid, did, score = q["query-id"], q["corpus-id"], q["score"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score
    return corpus, queries, qrels


def _get_doc_text(corpus: dict, doc_id: str) -> str:
    """Extrae texto de un documento. BEIR devuelve dict con title/text."""
    doc = corpus.get(doc_id, "")
    if isinstance(doc, dict):
        title = doc.get("title", "")
        text = doc.get("text", "")
        if title and text:
            return f"{title}. {text}"
        return text or title
    return str(doc)


def _bm25_retrieval(query, doc_ids, doc_texts, top_k, bm25_index=None):
    from rank_bm25 import BM25Okapi
    
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    if bm25_index is None:
        tokenized_corpus = [tokenize(doc) for doc in doc_texts]
        bm25_index = BM25Okapi(tokenized_corpus)
    
    scores = bm25_index.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], scores[i]) for i in top_indices]


def _dense_retrieval(query, doc_ids, doc_texts, top_k, embedding_model=None):
    import numpy as np
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_emb = embedding_model.encode([query], normalize_embeddings=True)
    doc_embs = embedding_model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)
    scores = (doc_embs @ query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]


def first_stage_retrieval(
    query: str,
    corpus: Dict[str, str],
    top_k: int = 100,
    method: str = "bm25",
    bm25_index=None,
    embedding_model=None,
) -> List[Tuple[str, float]]:
    """First-stage retrieval: BM25 o dense. Devuelve (doc_id, score) ordenado."""
    doc_ids = list(corpus.keys())
    doc_texts = [_get_doc_text(corpus, did) for did in doc_ids]
    
    if method == "bm25":
        return _bm25_retrieval(query, doc_ids, doc_texts, top_k, bm25_index)
    elif method == "dense":
        return _dense_retrieval(query, doc_ids, doc_texts, top_k, embedding_model)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def build_dspy_examples(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    top_k_retrieval: int = 100,
    retrieval_method: str = "bm25",
    max_examples: Optional[int] = None,
    doc_text_truncate: int = 512,
) -> List[dspy.Example]:
    """Construye dspy.Examples con query, search_results y relevant_doc_ids."""
    examples = []
    
    # Pre-construir índice BM25
    bm25_index = None
    if retrieval_method == "bm25":
        try:
            from rank_bm25 import BM25Okapi
            doc_ids = list(corpus.keys())
            doc_texts = [_get_doc_text(corpus, did) for did in doc_ids]
            tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in doc_texts]
            bm25_index = BM25Okapi(tokenized_corpus)
            print(f"Built BM25 index over {len(doc_ids)} documents")
        except ImportError:
            print("rank_bm25 not found, falling back to dense retrieval")
            retrieval_method = "dense"
    
    query_items = list(queries.items())
    if max_examples:
        query_items = query_items[:max_examples]
    
    for query_id, query_text in query_items:
        if query_id not in qrels:
            continue
        
        retrieved = first_stage_retrieval(
            query=query_text, corpus=corpus, top_k=top_k_retrieval,
            method=retrieval_method, bm25_index=bm25_index,
        )
        
        doc_id_to_int = {doc_id: i + 1 for i, (doc_id, _) in enumerate(retrieved)}
        
        search_results = []
        for i, (doc_id, score) in enumerate(retrieved):
            doc_text = _get_doc_text(corpus, doc_id)[:doc_text_truncate]
            search_results.append(DocumentCandidate(id=doc_id_to_int[doc_id], text=doc_text, initial_rank=i + 1))
        
        relevant_doc_ids = [doc_id_to_int[doc_id] for doc_id in qrels[query_id] if doc_id in doc_id_to_int]
        if not relevant_doc_ids:
            continue
        
        example = dspy.Example(
            query=query_text,
            search_results=search_results,
            relevant_doc_ids=relevant_doc_ids,
        ).with_inputs("query", "search_results")
        examples.append(example)
    
    print(f"Built {len(examples)} examples ({retrieval_method} retrieval, top-{top_k_retrieval})")
    return examples


def load_and_prepare(
    dataset_name: str,
    split: str = "test",
    top_k_retrieval: int = 100,
    retrieval_method: str = "bm25",
    max_examples: Optional[int] = None,
    train_ratio: float = 0.5,
    cache_dir: Optional[str] = None,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Carga dataset, ejecuta retrieval y prepara train/val splits."""
    import random
    
    corpus, queries, qrels = load_beir_dataset(dataset_name, split=split, cache_dir=cache_dir)
    examples = build_dspy_examples(
        queries=queries, corpus=corpus, qrels=qrels,
        top_k_retrieval=top_k_retrieval, retrieval_method=retrieval_method,
        max_examples=max_examples,
    )
    
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    trainset, valset = examples[:split_idx], examples[split_idx:]
    print(f"Split: {len(trainset)} train / {len(valset)} val")
    return trainset, valset