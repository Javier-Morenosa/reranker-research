"""
Métricas de Information Retrieval para evaluar reranking.

Implementa las métricas estándar usadas en los papers de reranking:
- nDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k
- MRR (Mean Reciprocal Rank)
- MAP@k (Mean Average Precision)

Y una función métrica compatible con DSPy que combina estas métricas
para usar con los optimizadores (MIPROv2, GEPA).
"""

import math
import numpy as np
import dspy
from typing import List, Set, Dict, Callable, Optional, Union


def dcg_at_k(relevances: List[Union[float, int]], k: int) -> float:
    """
    Discounted Cumulative Gain @ k.
    
    DCG@k = sum_{i=1}^{k} (rel_i / log2(i + 1))
    
    donde rel_i es la relevancia del documento en posición i.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        # +2 porque: enumerate empieza en 0, y la fórmula usa i+1 que empieza en 1
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: int,
    graded_relevance: Optional[Dict[int, int]] = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.
    
    nDCG@k = DCG@k / IDCG@k
    
    donde IDCG@k es el DCG del ranking ideal (todos los relevantes primero).
    
    Args:
        reranked_doc_ids: Lista de IDs de documentos en el orden del reranking.
        relevant_doc_ids: Set de IDs de documentos relevantes.
        k: Cut-off para la métrica.
        graded_relevance: Si se proporciona, usa relevancia graduada (dict doc_id -> score).
            Si no, usa relevancia binaria (1 si es relevante, 0 si no).
    
    Returns:
        nDCG@k score en [0, 1].
    """
    if not relevant_doc_ids:
        return 0.0
    
    # Relevancias en el orden del reranking
    if graded_relevance:
        reranked_rels = [graded_relevance.get(doc_id, 0) for doc_id in reranked_doc_ids[:k]]
        ideal_rels = sorted(graded_relevance.values(), reverse=True)[:k]
    else:
        reranked_rels = [1.0 if doc_id in relevant_doc_ids else 0.0 for doc_id in reranked_doc_ids[:k]]
        ideal_rels = [1.0] * min(len(relevant_doc_ids), k)
    
    dcg = dcg_at_k(reranked_rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def recall_at_k(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: int,
) -> float:
    """
    Recall@k: fracción de documentos relevantes que aparecen en el top-k.
    
    Recall@k = |relevantes ∩ top_k| / |relevantes|
    
    Args:
        reranked_doc_ids: Lista de IDs en el orden del reranking.
        relevant_doc_ids: Set de IDs relevantes.
        k: Cut-off.
    
    Returns:
        Recall@k en [0, 1].
    """
    if not relevant_doc_ids:
        return 0.0
    
    top_k_set = set(reranked_doc_ids[:k])
    num_relevant_in_top_k = len(top_k_set & relevant_doc_ids)
    return num_relevant_in_top_k / len(relevant_doc_ids)


def mrr(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: Optional[int] = None,
) -> float:
    """
    Mean Reciprocal Rank.
    
    MRR = 1 / rank del primer documento relevante.
    Si no hay relevante en el top-k (o en toda la lista si k=None), devuelve 0.
    
    Args:
        reranked_doc_ids: Lista de IDs en el orden del reranking.
        relevant_doc_ids: Set de IDs relevantes.
        k: Cut-off opcional. Si None, considera toda la lista.
    
    Returns:
        Reciprocal Rank en [0, 1].
    """
    effective_k = k if k is not None else len(reranked_doc_ids)
    
    for i, doc_id in enumerate(reranked_doc_ids[:effective_k]):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    
    return 0.0


def average_precision_at_k(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: int,
) -> float:
    """
    Average Precision @ k.
    
    AP@k = (1 / min(k, |relevantes|)) * sum_{i=1}^{k} (Precision@i * rel_i)
    
    donde rel_i = 1 si el documento en posición i es relevante, 0 si no.
    """
    if not relevant_doc_ids:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    
    for i, doc_id in enumerate(reranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    
    return sum_precision / min(k, len(relevant_doc_ids))


def reranking_metric_factory(
    k: int = 10,
    primary_metric: str = "ndcg",
) -> Callable:
    """
    Factory que crea una función métrica compatible con DSPy.
    
    La métrica recibe un dspy.Example (con los documentos relevantes)
    y un dspy.Prediction (con los reranked_ids), y devuelve un float.
    
    Esto es lo que se pasa a MIPROv2 o GEPA como `metric`.
    
    Args:
        k: Cut-off para las métricas.
        primary_metric: Qué métrica usar como score principal.
            Opciones: "ndcg", "recall", "mrr", "map".
    
    Returns:
        Función (example, pred, trace=None) -> float
    """
    metric_fn = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "mrr": mrr,
        "map": average_precision_at_k,
    }[primary_metric]
    
    def metric(example, pred, trace=None):
        relevant_ids = set(example.relevant_doc_ids)
        reranked_ids = pred.reranked_ids
        
        if primary_metric == "mrr":
            score = metric_fn(reranked_ids, relevant_ids, k=k)
        else:
            score = metric_fn(reranked_ids, relevant_ids, k=k)
        
        return score
    
    return metric


def reranking_metric_with_feedback_factory(
    k: int = 10,
    primary_metric: str = "ndcg",
) -> Callable:
    """
    Factory que crea una métrica con feedback textual para GEPA.
    
    GEPA puede usar feedback en lenguaje natural para guiar la evolución
    del prompt. Esta función devuelve tanto el score numérico como feedback
    sobre qué salió bien/mal.
    
    Returns:
        Función (example, pred, trace=None) -> dspy.Prediction(score, feedback)
    """
    metric_fn = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "mrr": mrr,
        "map": average_precision_at_k,
    }[primary_metric]
    
    def metric(example, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        relevant_ids = set(example.relevant_doc_ids)
        reranked_ids = pred.reranked_ids
        
        score = metric_fn(reranked_ids, relevant_ids, k=k)
        
        # Generar feedback textual
        top_k_ids = set(reranked_ids[:k])
        correctly_ranked = top_k_ids & relevant_ids
        missed = relevant_ids - top_k_ids
        irrelevant_in_top_k = top_k_ids - relevant_ids
        
        feedback_parts = []
        if correctly_ranked:
            feedback_parts.append(
                f"Correctly placed {len(correctly_ranked)} relevant documents in top-{k}."
            )
        if missed:
            feedback_parts.append(
                f"Missed {len(missed)} relevant documents that should have been in top-{k} "
                f"(doc IDs: {sorted(missed)[:5]}...)."
            )
        if irrelevant_in_top_k:
            feedback_parts.append(
                f"Included {len(irrelevant_in_top_k)} irrelevant documents in top-{k} "
                f"(doc IDs: {sorted(irrelevant_in_top_k)[:5]}...)."
            )
        
        # Información sobre el ranking ideal
        feedback_parts.append(
            f"Score ({primary_metric}@{k}): {score:.4f}. "
            f"The ideal ranking would place all {len(relevant_ids)} relevant documents first."
        )
        
        # Información sobre la query para que el reflection model pueda razonar
        if hasattr(example, 'query'):
            feedback_parts.append(f"Query was: '{str(example.query)[:150]}'.")

        feedback = " ".join(feedback_parts)
        
        return dspy.Prediction(score=score, feedback=feedback)
    
    return metric


def compute_all_metrics(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Computa todas las métricas IR estándar para una lista de k values.
    
    Args:
        reranked_doc_ids: IDs en el orden del reranking.
        relevant_doc_ids: Set de IDs relevantes.
        k_values: Lista de cut-offs.
    
    Returns:
        Dict con métricas: ndcg@5, ndcg@10, recall@5, mrr, etc.
    """
    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = ndcg_at_k(reranked_doc_ids, relevant_doc_ids, k)
        results[f"recall@{k}"] = recall_at_k(reranked_doc_ids, relevant_doc_ids, k)
        results[f"map@{k}"] = average_precision_at_k(reranked_doc_ids, relevant_doc_ids, k)
    
    results["mrr"] = mrr(reranked_doc_ids, relevant_doc_ids)
    results["mrr@10"] = mrr(reranked_doc_ids, relevant_doc_ids, k=10)
    
    return results


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Agrega métricas de múltiples queries (media).
    
    Args:
        all_metrics: Lista de dicts de métricas, uno por query.
    
    Returns:
        Dict con la media de cada métrica.
    """
    if not all_metrics:
        return {}
    
    keys = all_metrics[0].keys()
    return {key: np.mean([m[key] for m in all_metrics]) for key in keys}