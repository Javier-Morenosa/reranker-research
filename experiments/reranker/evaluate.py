"""
Pipeline de evaluación para el reranker.
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path

import dspy
from tqdm import tqdm

from .module import ListwiseReranker
from .metrics import compute_all_metrics, aggregate_metrics


def evaluate_reranker(
    program: ListwiseReranker,
    testset: List[dspy.Example],
    k: int = 10,
    k_values: List[int] = [1, 5, 10, 20],
    num_threads: int = 1,
    show_progress: bool = True,
    show_examples: bool = False,
) -> Dict:
    """Evalúa el reranker sobre testset y devuelve métricas agregadas."""
    all_metrics = []
    per_query_results = []
    
    start_time = time.time()
    iterator = tqdm(testset, desc="Evaluating") if show_progress else testset
    
    for example in iterator:
        query = example.query
        search_results = example.search_results
        relevant_ids = set(example.relevant_doc_ids)
        
        try:
            pred = program(query=query, search_results=search_results, top_k=k)
            reranked_ids = pred.reranked_ids
        except Exception as e:
            print(f"Error on query '{query[:50]}...': {e}")
            reranked_ids = [doc.id for doc in search_results[:k]]
        
        metrics = compute_all_metrics(reranked_ids, relevant_ids, k_values)
        all_metrics.append(metrics)
        
        if show_examples:
            per_query_results.append({
                "query": query[:100],
                "num_candidates": len(search_results),
                "num_relevant": len(relevant_ids),
                "reranked_top_5": reranked_ids[:5],
                "relevant_in_top_5": len(set(reranked_ids[:5]) & relevant_ids),
                "metrics": metrics,
            })
    
    elapsed = time.time() - start_time
    avg_metrics = aggregate_metrics(all_metrics)
    
    return {
        "metrics": avg_metrics,
        "per_query": per_query_results,
        "config": {"k": k, "k_values": k_values, "num_examples": len(testset)},
        "timing": {"total_seconds": elapsed, "per_query_seconds": elapsed / len(testset) if testset else 0},
    }


def print_eval_results(results: Dict):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDataset: {results['config']['num_examples']} examples")
    print(f"Timing: {results['timing']['total_seconds']:.1f}s ({results['timing']['per_query_seconds']:.2f}s/query)")
    print("\n--- Metrics ---")
    for metric_name, score in sorted(results["metrics"].items()):
        print(f"  {metric_name:>12s}: {score:.4f} ({score*100:.1f}%)")
    print("=" * 60)


def save_eval_results(results: Dict, path: str, tag: str = ""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "tag": tag, "metrics": results["metrics"],
        "config": results["config"], "timing": results["timing"],
        "per_query": results.get("per_query", []),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {path}")


def compare_results(baseline: Dict, optimized: Dict) -> str:
    """Compara baseline vs optimizado."""
    lines = ["\n" + "=" * 60, "BASELINE vs OPTIMIZED", "=" * 60, ""]
    lines.append(f"{'Metric':>15s}  {'Baseline':>10s}  {'Optimized':>10s}  {'Delta':>8s}  {'Rel%':>8s}")
    lines.append("-" * 60)
    
    b, o = baseline["metrics"], optimized["metrics"]
    for m in sorted(b.keys()):
        bv, ov = b[m], o.get(m, 0)
        delta = ov - bv
        rel = (delta / bv * 100) if bv > 0 else float("inf")
        lines.append(f"{m:>15s}  {bv:>10.4f}  {ov:>10.4f}  {delta:>+8.4f}  {rel:>+7.1f}%")
    
    lines.append("=" * 60)
    return "\n".join(lines)