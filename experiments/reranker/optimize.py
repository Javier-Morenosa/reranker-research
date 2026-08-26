"""
Optimización del reranker con DSPy.

Dos optimizadores soportados:
1. MIPROv2: optimiza instrucciones + few-shot examples con Bayesian Optimization.
2. GEPA: evolución reflexiva de prompts con feedback en lenguaje natural.

Ambos toman el reranker sin optimizar + trainset y devuelven
una versión compilada con prompts mejorados.
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
    """
    Optimiza el reranker con MIPROv2.
    
    MIPROv2 optimiza simultáneamente:
    - Las instrucciones del prompt (proponiendo variantes y evaluándolas).
    - Los few-shot examples (seleccionando ejemplos bootstrap).
    - Usa Bayesian Optimization para buscar el mejor combo.
    
    Args:
        program: ListwiseReranker sin optimizar.
        trainset: Datos de entrenamiento (dspy.Example).
        valset: Datos de validación.
        k: Cut-off para la métrica.
        primary_metric: "ndcg", "recall", "mrr", o "map".
        auto: Nivel de optimización. "light" = rápido, "heavy" = exhaustivo.
        max_bootstrapped_demos: Máximo de demos bootstrap (generados por el modelo).
        max_labeled_demos: Máximo de demos etiquetados (del trainset).
        num_threads: Threads para evaluación paralela.
        log_dir: Directorio para logs de optimización.
    
    Returns:
        ListwiseReranker optimizado.
    """
    from dspy.teleprompt import MIPROv2
    
    metric = reranking_metric_factory(k=k, primary_metric=primary_metric)
    
    optimizer = MIPROv2(
        metric=metric,
        auto=auto,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_threads=num_threads,
        verbose=True,
        track_stats=True,
        log_dir=log_dir,
    )
    
    print(f"Running MIPROv2 ({auto}) optimization...")
    print(f"  Metric: {primary_metric}@{k}")
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val: {len(valset)} examples")
    
    optimized = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=valset,
    )
    
    return optimized


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
    """
    Optimiza el reranker con GEPA (Generative Prompt Adaptation).
    
    GEPA introduce varias innovaciones:
    - Mutación reflexiva del prompt: el modelo reflexiona sobre sus errores
      y propone mejoras.
    - System-aware merge: combina candidatos de forma inteligente.
    - Pareto-optimal candidate selection: mantiene candidatos que son mejores
      en al menos un ejemplo de validación.
    - Feedback en lenguaje natural: la métrica devuelve texto explicativo,
      no solo un scalar.
    
    Args:
        program: ListwiseReranker sin optimizar.
        trainset: Datos de entrenamiento.
        valset: Datos de validación.
        k: Cut-off para la métrica.
        primary_metric: Métrica principal.
        auto: Nivel de optimización.
        max_metric_calls: Budget de evaluaciones (más = mejor pero más caro).
        reflection_model: Modelo para reflexión. Soporta:
            - "openai/gpt-4o" (default)
            - "ollama-cloud/glm-5.2" (Ollama Cloud)
            - "ollama/mistral:latest" (Ollama local)
        num_threads: Threads paralelos.
        log_dir: Directorio para logs.
    
    Returns:
        ListwiseReranker optimizado.
    """
    import os
    
    # Configurar reflection LM (soporta Ollama Cloud)
    if reflection_model.startswith("ollama-cloud/"):
        model_name = reflection_model.replace("ollama-cloud/", "")
        ollama_key = os.getenv("OLLAMA_API_KEY")
        if not ollama_key:
            raise ValueError("Ollama Cloud requires OLLAMA_API_KEY env var")
        reflection_lm = dspy.LM(
            f"openai/{model_name}",
            api_key=ollama_key,
            api_base="https://ollama.com/v1",
            temperature=1.0,
            max_tokens=16000,
        )
    else:
        reflection_lm = dspy.LM(reflection_model, temperature=1.0, max_tokens=16000)
    
    # GEPA requiere que exactamente uno de: auto, max_metric_calls, max_full_evals esté set
    # Si auto está set, no podemos pasar max_metric_calls al constructor
    metric_fn = reranking_metric_with_feedback_factory(k=k, primary_metric=primary_metric)
    
    common_kwargs = dict(
        metric=metric_fn,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        track_stats=True,
        track_best_outputs=True,
        log_dir=log_dir or "./gepa_logs",
        num_threads=num_threads,
        seed=0,
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
    
    print(f"Running GEPA ({auto}) optimization...")
    print(f"  Metric: {primary_metric}@{k}")
    print(f"  Max metric calls: {max_metric_calls}")
    print(f"  Reflection model: {reflection_model}")
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val: {len(valset)} examples")
    
    optimized = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=valset,
    )
    
    return optimized


def save_optimized_program(
    program: ListwiseReranker,
    path: str,
):
    """Guarda un programa optimizado a disco."""
    program.save(path)
    print(f"Saved optimized program to {path}")


def load_optimized_program(
    path: str,
    program: ListwiseReranker,
) -> ListwiseReranker:
    """Carga un programa optimizado desde disco."""
    program.load(path)
    print(f"Loaded optimized program from {path}")
    return program