"""
Carga de datasets BEIR y first-stage retrieval.

BEIR es un benchmark estándar para IR con datasets como:
- SCIFACT: Verificación científica (5,183 queries, 5,183 docs)
- NFCORPUS: Nutrición (323 queries, 36,333 docs)  
- FIQA: Q&A financiero (648 queries, 57,638 docs)
- TREC-COVID: Literatura COVID (50 queries, 171,332 docs)
- ROBUST04: News (300 queries, 528,155 docs)

Esta módulo:
1. Carga un dataset BEIR con sus qrels (query -> relevant doc IDs).
2. Ejecuta first-stage retrieval (BM25 o dense) para obtener candidatos.
3. Formatea todo como dspy.Example listos para el reranker.
"""

import os
import json
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path

import dspy
from .signatures import DocumentCandidate


# Datasets BEIR soportados (los más comunes para experimentos rápidos)
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
    """
    Carga un dataset BEIR.
    
    Usa la librería `beir` si está disponible, sino carga desde HuggingFace datasets.
    
    Args:
        dataset_name: Nombre del dataset (ver BEIR_DATASETS).
        split: Split a cargar ("test", "dev", "train").
        cache_dir: Directorio de caché opcional.
    
    Returns:
        Tuple de (corpus, queries, qrels) donde:
        - corpus: {doc_id: document_text}
        - queries: {query_id: query_text}
        - qrels: {query_id: {doc_id: relevance_score}}
    """
    if dataset_name not in BEIR_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available: {list(BEIR_DATASETS.keys())}"
        )
    
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
        # Fallback: cargar desde HuggingFace datasets
        print(f"Warning: beir library not found. Trying HuggingFace datasets...")
        return _load_from_hf(BEIR_DATASETS[dataset_name], split)


def _load_from_hf(
    hf_dataset: str,
    split: str = "test",
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Carga dataset desde HuggingFace como fallback."""
    from datasets import load_dataset
    
    # BEIR datasets en HF tienen splits: queries, corpus, qrels
    queries_ds = load_dataset(hf_dataset, "queries", split=split)
    corpus_ds = load_dataset(hf_dataset, "corpus", split="train")
    qrels_ds = load_dataset(hf_dataset, "qrels", split=split)
    
    queries = {q["_id"]: q["text"] for q in queries_ds}
    corpus = {c["_id"]: c["text"] for c in corpus_ds}
    qrels = {}
    for q in qrels_ds:
        qid = q["query-id"]
        did = q["corpus-id"]
        score = q["score"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score
    
    return corpus, queries, qrels


def first_stage_retrieval(
    query: str,
    corpus: Dict[str, str],
    top_k: int = 100,
    method: str = "bm25",
    bm25_index=None,
    embedding_model=None,
) -> List[Tuple[str, float]]:
    """
    Ejecuta first-stage retrieval para una query.
    
    Métodos soportados:
    - "bm25": BM25 usando rank_bm25. Rápido, lexical.
    - "dense": Dense retrieval usando sentence-transformers. Semántico.
    - "oracle": Devuelve los documentos relevantes primero (solo para debugging).
    
    Args:
        query: Texto de la query.
        corpus: {doc_id: doc_text}
        top_k: Número de documentos a recuperar.
        method: Método de retrieval.
        bm25_index: Índice BM25 pre-construido (opcional, para reutilización).
        embedding_model: Modelo de embeddings pre-cargado (opcional).
    
    Returns:
        Lista de (doc_id, score) ordenada por relevancia descendente.
    """
    doc_ids = list(corpus.keys())
    doc_texts = [_get_doc_text(corpus, did) for did in doc_ids]
    
    if method == "bm25":
        return _bm25_retrieval(query, doc_ids, doc_texts, top_k, bm25_index)
    elif method == "dense":
        return _dense_retrieval(query, doc_ids, doc_texts, top_k, embedding_model)
    elif method == "oracle":
        # Solo para testing: devuelve en orden aleatorio (el reranker debe encontrar los relevantes)
        import random
        random.shuffle(doc_ids)
        return [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(doc_ids[:top_k])]
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def _bm25_retrieval(
    query: str,
    doc_ids: List[str],
    doc_texts: List[str],
    top_k: int,
    bm25_index=None,
) -> List[Tuple[str, float]]:
    """Retrieval con BM25 usando rank_bm25."""
    from rank_bm25 import BM25Okapi
    import re
    
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    if bm25_index is None:
        tokenized_corpus = [tokenize(doc) for doc in doc_texts]
        bm25_index = BM25Okapi(tokenized_corpus)
    
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    # Top-k por score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], scores[i]) for i in top_indices]


def _dense_retrieval(
    query: str,
    doc_ids: List[str],
    doc_texts: List[str],
    top_k: int,
    embedding_model=None,
) -> List[Tuple[str, float]]:
    """Dense retrieval usando sentence-transformers."""
    import numpy as np
    
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode query y corpus
    query_emb = embedding_model.encode([query], normalize_embeddings=True)
    doc_embs = embedding_model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Similitud coseno (ya normalizado -> producto punto)
    scores = (doc_embs @ query_emb.T).flatten()
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]


def _get_doc_text(corpus: dict, doc_id: str) -> str:
    """
    Extrae el texto de un documento del corpus de BEIR.
    
    BEIR devuelve cada documento como:
    - dict: {"title": "...", "text": "..."} (formato estándar BEIR)
    - str: "..." (formato simplificado)
    
    Combina title + text si ambos están disponibles.
    """
    doc = corpus.get(doc_id, "")
    if isinstance(doc, dict):
        title = doc.get("title", "")
        text = doc.get("text", "")
        if title and text:
            return f"{title}. {text}"
        return text or title
    return str(doc)


def build_dspy_examples(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    top_k_retrieval: int = 100,
    retrieval_method: str = "bm25",
    max_examples: Optional[int] = None,
    doc_text_truncate: int = 512,
) -> List[dspy.Example]:
    """
    Construye dspy.Examples listos para el reranker.
    
    Para cada query:
    1. Ejecuta first-stage retrieval (top_k_retrieval documentos).
    2. Formatea los candidatos como DocumentCandidate.
    3. Guarda los doc IDs relevantes según las qrels.
    4. Crea un dspy.Example con query, search_results, relevant_doc_ids.
    
    Args:
        queries: {query_id: query_text}
        corpus: {doc_id: doc_text o {"title":..., "text":...}}
        qrels: {query_id: {doc_id: relevance}}
        top_k_retrieval: Cuántos documentos recuperar en first-stage.
        retrieval_method: "bm25" o "dense".
        max_examples: Limitar número de ejemplos (para experimentos rápidos).
        doc_text_truncate: Truncar texto de documentos a N caracteres.
    
    Returns:
        Lista de dspy.Example con campos:
        - query: str
        - search_results: List[DocumentCandidate]
        - relevant_doc_ids: List[int] (IDs 1-indexed de docs relevantes)
    """
    examples = []
    
    # Pre-construir índice BM25 si es necesario
    bm25_index = None
    if retrieval_method == "bm25":
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            def tokenize(text: str) -> List[str]:
                return re.findall(r'\b\w+\b', text.lower())
            
            doc_ids = list(corpus.keys())
            doc_texts = [_get_doc_text(corpus, did) for did in doc_ids]
            tokenized_corpus = [tokenize(doc) for doc in doc_texts]
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
        
        # First-stage retrieval
        retrieved = first_stage_retrieval(
            query=query_text,
            corpus=corpus,
            top_k=top_k_retrieval,
            method=retrieval_method,
            bm25_index=bm25_index,
        )
        
        # Mapear doc_ids del corpus a IDs 1-indexed para el reranker
        # El reranker usa IDs enteros, no strings
        doc_id_to_int = {doc_id: i + 1 for i, (doc_id, _) in enumerate(retrieved)}
        
        # Construir DocumentCandidate list
        search_results = []
        for i, (doc_id, score) in enumerate(retrieved):
            doc_text = _get_doc_text(corpus, doc_id)[:doc_text_truncate]
            search_results.append(
                DocumentCandidate(
                    id=doc_id_to_int[doc_id],
                    text=doc_text,
                    initial_rank=i + 1,
                )
            )
        
        # IDs relevantes (en el espacio de IDs enteros del reranker)
        relevant_doc_ids = []
        for doc_id in qrels[query_id]:
            if doc_id in doc_id_to_int:
                relevant_doc_ids.append(doc_id_to_int[doc_id])
        
        # Solo incluir ejemplos que tengan al menos un documento relevante
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
    """
    Función convenience que carga el dataset, ejecuta retrieval y prepara
    train/val splits para el optimizador.
    
    Args:
        dataset_name: Nombre del dataset BEIR.
        split: Split a cargar.
        top_k_retrieval: Documentos a recuperar en first-stage.
        retrieval_method: "bm25" o "dense".
        max_examples: Limitar número total de ejemplos.
        train_ratio: Fracción para train (resto va a val).
        cache_dir: Cache para BEIR.
    
    Returns:
        (trainset, valset) listas de dspy.Example.
    """
    corpus, queries, qrels = load_beir_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    examples = build_dspy_examples(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        top_k_retrieval=top_k_retrieval,
        retrieval_method=retrieval_method,
        max_examples=max_examples,
    )
    
    # Split train/val
    import random
    random.seed(42)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_ratio)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]
    
    print(f"Split: {len(trainset)} train / {len(valset)} val")
    return trainset, valset