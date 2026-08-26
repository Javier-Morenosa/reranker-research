"""
Pipeline de evaluación end-to-end para el reranker.

Evalúa un reranker (optimizado o no) sobre un dataset y reporta
todas las métricas IR: nDCG@k, Recall@k, MRR, MAP@k.
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path

import dspy
from tqdm import tqdm

from .module import ListwiseReranker
from .metrics import (
    compute_all_metrics,
    aggregate_metrics,
)
from .signatures import DocumentCandidate


def evaluate_reranker(
    program: ListwiseReranker,
    testset: List[dspy.Example],
    k: int = 10,
    k_values: List[int] = [1, 5, 10, 20],
    num_threads: int = 1,
    show_progress: bool = True,
    show_examples: bool = False,
) -> Dict:
    """
    Evalúa el reranker sobre un conjunto de test.
    
    Para cada ejemplo en testset:
    1. Llama al reranker con query + search_results.
    2. Compara el reranking con los documentos relevantes (qrels).
    3. Computa nDCG@k, Recall@k, MRR, MAP@k.
    
    Args:
        program: ListwiseReranker (optimizado o baseline).
        testset: Lista de dspy.Example con query, search_results, relevant_doc_ids.
        k: Cut-off principal para el reranking.
        k_values: Cut-offs para las métricas.
        num_threads: Si > 1, usa evaluación paralela con dspy.Evaluate.
        show_progress: Mostrar barra de progreso.
        show_examples: Mostrar ejemplos individuales con detalles.
    
    Returns:
        Dict con:
        - "metrics": {metric_name: avg_score}
        - "per_query": [{query, metrics, ...}, ...]
        - "config": parámetros de la evaluación
        - "timing": tiempo total
    """
    all_metrics = []
    per_query_results = []
    
    start_time = time.time()
    
    iterator = tqdm(testset, desc="Evaluating") if show_progress else testset
    
    for example in iterator:
        query = example.query
        search_results = example.search_results
        relevant_ids = set(example.relevant_doc_ids)
        
        # Llamar al reranker
        try:
            pred = program(
                query=query,
                search_results=search_results,
                top_k=k,
            )
            reranked_ids = pred.reranked_ids
        except Exception as e:
            print(f"Error on query '{query[:50]}...': {e}")
            # Fallback: usar el ranking original
            reranked_ids = [doc.id for doc in search_results[:k]]
        
        # Computar métricas
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
    
    # Agregar métricas
    avg_metrics = aggregate_metrics(all_metrics)
    
    result = {
        "metrics": avg_metrics,
        "per_query": per_query_results,
        "config": {
            "k": k,
            "k_values": k_values,
            "num_examples": len(testset),
            "num_threads": num_threads,
        },
        "timing": {
            "total_seconds": elapsed,
            "per_query_seconds": elapsed / len(testset) if testset else 0,
        },
    }
    
    return result


def print_eval_results(results: Dict):
    """Imprime los resultados de evaluación de forma legible."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = results["metrics"]
    print(f"\nDataset: {results['config']['num_examples']} examples")
    print(f"Timing: {results['timing']['total_seconds']:.1f}s "
          f"({results['timing']['per_query_seconds']:.2f}s/query)")
    
    print("\n--- Metrics ---")
    for metric_name, score in sorted(metrics.items()):
        # Formatear según el tipo de métrica
        print(f"  {metric_name:>12s}: {score:.4f} ({score*100:.1f}%)")
    
    print("=" * 60)


def save_eval_results(
    results: Dict,
    path: str,
    tag: str = "",
):
    """Guarda resultados de evaluación a JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "tag": tag,
        "metrics": results["metrics"],
        "config": results["config"],
        "timing": results["timing"],
        "per_query": results.get("per_query", []),
    }
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved results to {path}")


def compare_results(
    baseline: Dict,
    optimized: Dict,
) -> str:
    """
    Compara resultados baseline vs optimizado y devuelve un resumen.
    
    Args:
        baseline: Resultados del reranker sin optimizar.
        optimized: Resultados del reranker optimizado.
    
    Returns:
        String con la comparación formateada.
    """
    lines = ["\n" + "=" * 60]
    lines.append("BASELINE vs OPTIMIZED")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Metric':>15s}  {'Baseline':>10s}  {'Optimized':>10s}  {'Delta':>8s}  {'Rel%':>8s}")
    lines.append("-" * 60)
    
    b_metrics = baseline["metrics"]
    o_metrics = optimized["metrics"]
    
    for metric_name in sorted(b_metrics.keys()):
        b_val = b_metrics[metric_name]
        o_val = o_metrics.get(metric_name, 0)
        delta = o_val - b_val
        rel = (delta / b_val * 100) if b_val > 0 else float("inf")
        
        lines.append(
            f"{metric_name:>15s}  {b_val:>10.4f}  {o_val:>10.4f}  "
            f"{delta:>+8.4f}  {rel:>+7.1f}%"
        )
    
    lines.append("=" * 60)
    
    return "\n".join(lines)