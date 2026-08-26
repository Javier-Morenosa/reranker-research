"""
Métricas IR para evaluar reranking: nDCG@k, Recall@k, MRR, MAP@k.
Incluye factories compatibles con DSPy (MIPROv2 y GEPA).
"""

import math
import numpy as np
import dspy
from typing import List, Set, Dict, Callable, Optional, Union


def dcg_at_k(relevances: List[Union[float, int]], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: int,
    graded_relevance: Optional[Dict[int, int]] = None,
) -> float:
    """nDCG@k = DCG@k / IDCG@k (normalizado contra el ranking ideal)."""
    if not relevant_doc_ids:
        return 0.0
    
    if graded_relevance:
        reranked_rels = [graded_relevance.get(doc_id, 0) for doc_id in reranked_doc_ids[:k]]
        ideal_rels = sorted(graded_relevance.values(), reverse=True)[:k]
    else:
        reranked_rels = [1.0 if doc_id in relevant_doc_ids else 0.0 for doc_id in reranked_doc_ids[:k]]
        ideal_rels = [1.0] * min(len(relevant_doc_ids), k)
    
    dcg = dcg_at_k(reranked_rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: int,
) -> float:
    """Recall@k = |relevantes ∩ top_k| / |relevantes|"""
    if not relevant_doc_ids:
        return 0.0
    top_k_set = set(reranked_doc_ids[:k])
    return len(top_k_set & relevant_doc_ids) / len(relevant_doc_ids)


def mrr(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k: Optional[int] = None,
) -> float:
    """1 / rank del primer documento relevante."""
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
    if not relevant_doc_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(reranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(k, len(relevant_doc_ids))


def reranking_metric_factory(k: int = 10, primary_metric: str = "ndcg") -> Callable:
    """Crea una métrica (example, pred, trace) -> float para MIPROv2."""
    metric_fn = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "mrr": mrr,
        "map": average_precision_at_k,
    }[primary_metric]
    
    def metric(example, pred, trace=None):
        relevant_ids = set(example.relevant_doc_ids)
        if primary_metric == "mrr":
            return float(metric_fn(pred.reranked_ids, relevant_ids, k=k))
        return float(metric_fn(pred.reranked_ids, relevant_ids, k=k))
    
    return metric


def reranking_metric_with_feedback_factory(k: int = 10, primary_metric: str = "ndcg") -> Callable:
    """Métrica con feedback textual para GEPA. Devuelve dspy.Prediction(score, feedback)."""
    metric_fn = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "mrr": mrr,
        "map": average_precision_at_k,
    }[primary_metric]
    
    def metric(example, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        relevant_ids = set(example.relevant_doc_ids)
        score = metric_fn(pred.reranked_ids, relevant_ids, k=k)
        
        top_k_ids = set(pred.reranked_ids[:k])
        correctly_ranked = top_k_ids & relevant_ids
        missed = relevant_ids - top_k_ids
        irrelevant_in_top_k = top_k_ids - relevant_ids
        
        feedback_parts = []
        if correctly_ranked:
            feedback_parts.append(f"Correctly placed {len(correctly_ranked)} relevant documents in top-{k}.")
        if missed:
            feedback_parts.append(f"Missed {len(missed)} relevant documents that should have been in top-{k}.")
        if irrelevant_in_top_k:
            feedback_parts.append(f"Included {len(irrelevant_in_top_k)} irrelevant documents in top-{k}.")
        
        feedback_parts.append(f"Score ({primary_metric}@{k}): {score:.4f}.")
        if hasattr(example, 'query'):
            feedback_parts.append(f"Query was: '{str(example.query)[:150]}'.")
        
        return dspy.Prediction(score=score, feedback=" ".join(feedback_parts))
    
    return metric


def compute_all_metrics(
    reranked_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = ndcg_at_k(reranked_doc_ids, relevant_doc_ids, k)
        results[f"recall@{k}"] = recall_at_k(reranked_doc_ids, relevant_doc_ids, k)
        results[f"map@{k}"] = average_precision_at_k(reranked_doc_ids, relevant_doc_ids, k)
    results["mrr"] = mrr(reranked_doc_ids, relevant_doc_ids)
    results["mrr@10"] = mrr(reranked_doc_ids, relevant_doc_ids, k=10)
    return results


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {key: float(np.mean([m[key] for m in all_metrics])) for key in keys}