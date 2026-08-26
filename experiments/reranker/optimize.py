"""
Optimización del reranker con MIPROv2 y GEPA.
"""

import os
import dspy
from typing import Optional, Literal
from .module import ListwiseReranker
from .metrics import reranking_metric_factory, reranking_metric_with_feedback_factory


def optimize_with_mipro(
    program: ListwiseReranker,
    trainset: list,
    valset: list,
    k: int = 10,
    primary_metric: str = "ndcg",
    auto: Literal["light", "medium", "heavy"] = "light",
    max_bootstrapped_demos: int = 2,
    max_labeled_demos: int = 2,
    num_threads: int = 4,
    log_dir: Optional[str] = None,
) -> ListwiseReranker:
    """Optimiza con MIPROv2 (Bayesian Optimization sobre instrucciones + few-shot demos)."""
    from dspy.teleprompt import MIPROv2
    
    metric = reranking_metric_factory(k=k, primary_metric=primary_metric)
    optimizer = MIPROv2(
        metric=metric, auto=auto,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_threads=num_threads, verbose=True,
        track_stats=True, log_dir=log_dir,
    )
    
    print(f"Running MIPROv2 ({auto})... metric: {primary_metric}@{k}")
    return optimizer.compile(student=program, trainset=trainset, valset=valset)


def optimize_with_gepa(
    program: ListwiseReranker,
    trainset: list,
    valset: list,
    k: int = 10,
    primary_metric: str = "ndcg",
    auto: Literal["light", "medium", "heavy"] = "medium",
    max_metric_calls: int = 300,
    reflection_model: str = "openai/gpt-4o",
    num_threads: int = 8,
    log_dir: Optional[str] = None,
) -> ListwiseReranker:
    """Optimiza con GEPA (evolución reflexiva de prompts con feedback textual)."""
    # Configurar reflection LM (soporta Ollama Cloud)
    if reflection_model.startswith("ollama-cloud/"):
        model_name = reflection_model.replace("ollama-cloud/", "")
        ollama_key = os.getenv("OLLAMA_API_KEY")
        if not ollama_key:
            raise ValueError("Ollama Cloud requires OLLAMA_API_KEY env var")
        reflection_lm = dspy.LM(
            f"openai/{model_name}", api_key=ollama_key,
            api_base="https://ollama.com/v1", temperature=1.0, max_tokens=16000,
        )
    else:
        reflection_lm = dspy.LM(reflection_model, temperature=1.0, max_tokens=16000)
    
    metric_fn = reranking_metric_with_feedback_factory(k=k, primary_metric=primary_metric)
    
    common_kwargs = dict(
        metric=metric_fn, reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto", track_stats=True,
        track_best_outputs=True, log_dir=log_dir or "./gepa_logs",
        num_threads=num_threads, seed=0,
    )
    
    if max_metric_calls and not auto:
        common_kwargs["max_metric_calls"] = max_metric_calls
    else:
        common_kwargs["auto"] = auto
    
    try:
        optimizer = dspy.GEPA(**common_kwargs)
    except AttributeError:
        from dspy.teleprompt import GEPA
        optimizer = GEPA(**common_kwargs)
    
    print(f"Running GEPA ({auto})... metric: {primary_metric}@{k}")
    return optimizer.compile(student=program, trainset=trainset, valset=valset)


def save_optimized_program(program: ListwiseReranker, path: str):
    program.save(path)
    print(f"Saved optimized program to {path}")


def load_optimized_program(path: str, program: ListwiseReranker) -> ListwiseReranker:
    program.load(path)
    print(f"Loaded optimized program from {path}")
    return program