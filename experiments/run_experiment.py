#!/usr/bin/env python
"""
Script principal para experimentos de listwise reranking con DSPy.

Uso:
    # Baseline (sin optimización)
    python run_experiment.py --dataset scifact --k 10 --no-optimize

    # Optimización con MIPROv2
    python run_experiment.py --dataset scifact --k 10 --optimizer mipro

    # Optimización con GEPA
    python run_experiment.py --dataset scifact --k 10 --optimizer gepa

    # Con modelo local (Ollama)
    python run_experiment.py --dataset scifact --lm ollama/qwen2.5:7b --optimizer mipro
"""

import argparse
import os
import json
import sys
from pathlib import Path

# Añadir el directorio del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

import dspy

from reranker import (
    ListwiseReranker,
    SummarizedListwiseReranker,
    load_beir_dataset,
    load_and_prepare,
    evaluate_reranker,
    optimize_with_mipro,
    optimize_with_gepa,
    save_optimized_program,
    load_optimized_program,
    print_eval_results,
    save_eval_results,
    compare_results,
)


def setup_lm(lm_name: str, api_key: str = None, cache: bool = True):
    """
    Configura el LLM en DSPy.
    
    Soporta:
    - OpenAI: --lm openai/gpt-4o-mini (usa OPENAI_API_KEY)
    - Ollama local: --lm ollama/mistral:latest (sin API key)
    - Ollama Cloud: --lm ollama-cloud/glm-5.2 (usa OLLAMA_API_KEY)
    - Cualquier provider compatible con litellm
    """
    kwargs = {}
    
    # Detectar Ollama Cloud
    if lm_name.startswith("ollama-cloud/"):
        model = lm_name.replace("ollama-cloud/", "")
        ollama_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not ollama_key:
            raise ValueError(
                "Ollama Cloud requires OLLAMA_API_KEY. "
                "Set it with: export OLLAMA_API_KEY=your_key"
            )
        lm = dspy.LM(
            f"openai/{model}",
            api_key=ollama_key,
            api_base="https://ollama.com/v1",
            cache=cache,
        )
    else:
        if api_key:
            kwargs["api_key"] = api_key
        lm = dspy.LM(lm_name, cache=cache, **kwargs)
    
    dspy.configure(lm=lm, track_usage=True)
    print(f"Configured LM: {lm_name}")
    return lm


def main():
    parser = argparse.ArgumentParser(
        description="Listwise Reranker experiment with DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="scifact",
        choices=["scifact", "nfcorpus", "fiqa", "trec-covid", "arguana", "scidocs"],
        help="BEIR dataset to use",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Max number of examples (for quick experiments)",
    )
    
    # Retrieval
    parser.add_argument(
        "--retrieval-method", type=str, default="bm25",
        choices=["bm25", "dense"],
        help="First-stage retrieval method",
    )
    parser.add_argument(
        "--top-k-retrieval", type=int, default=20,
        help="Number of documents to retrieve in first stage",
    )
    
    # Reranking
    parser.add_argument(
        "--k", type=int, default=10,
        help="Cut-off k for reranking and metrics",
    )
    parser.add_argument(
        "--use-cot", action="store_true",
        help="Use ChainOfThought instead of Predict",
    )
    parser.add_argument(
        "--verbose-signature", action="store_true",
        help="Use verbose signature with detailed ranking instructions",
    )
    parser.add_argument(
        "--summarized", action="store_true",
        help="Use SummarizedListwiseReranker (summary stage before reranking)",
    )
    
    # Model
    parser.add_argument(
        "--lm", type=str, default="openai/gpt-4o-mini",
        help="Language model. Examples: openai/gpt-4o-mini, ollama/mistral:latest (local), "
             "ollama-cloud/glm-5.2 (Ollama Cloud, needs OLLAMA_API_KEY)",
    )
    parser.add_argument(
        "--reflection-lm", type=str, default="openai/gpt-4o",
        help="Reflection model for GEPA (should be strong). "
             "Example: ollama-cloud/glm-5.2",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable DSPy cache",
    )
    
    # Optimization
    parser.add_argument(
        "--optimizer", type=str, default=None,
        choices=["mipro", "gepa"],
        help="Optimizer to use (skip optimization if not set)",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip optimization, only evaluate baseline",
    )
    parser.add_argument(
        "--auto", type=str, default="light",
        choices=["light", "medium", "heavy"],
        help="Optimization intensity",
    )
    parser.add_argument(
        "--max-metric-calls", type=int, default=300,
        help="Max metric calls for GEPA",
    )
    parser.add_argument(
        "--primary-metric", type=str, default="ndcg",
        choices=["ndcg", "recall", "mrr", "map"],
        help="Primary IR metric to optimize",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.5,
        help="Fraction of data for training (rest for validation)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of threads for parallel evaluation",
    )
    
    # Output
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for this experiment run",
    )
    parser.add_argument(
        "--load-optimized", type=str, default=None,
        help="Load a previously saved optimized program instead of optimizing",
    )
    
    args = parser.parse_args()
    
    # Setup
    # Detectar API key según el provider
    if "ollama-cloud" in args.lm:
        api_key = os.getenv("OLLAMA_API_KEY")
    elif "openai" in args.lm:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = None
    setup_lm(args.lm, api_key=api_key, cache=not args.no_cache)
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tag = args.tag or f"{args.dataset}_{args.optimizer or 'baseline'}_{args.k}"
    
    # 1. Load dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*60}")
    
    trainset, valset = load_and_prepare(
        dataset_name=args.dataset,
        split=args.split,
        top_k_retrieval=args.top_k_retrieval,
        retrieval_method=args.retrieval_method,
        max_examples=args.max_examples,
        train_ratio=args.train_ratio,
    )
    
    # 2. Create reranker program
    if args.summarized:
        program = SummarizedListwiseReranker(
            top_k=args.k,
            use_cot=True,
        )
        print(f"\nCreated SummarizedListwiseReranker (k={args.k})")
    else:
        program = ListwiseReranker(
            top_k=args.k,
            use_cot=args.use_cot,
            verbose_signature=args.verbose_signature,
        )
        print(f"\nCreated ListwiseReranker (k={args.k}, cot={args.use_cot}, verbose={args.verbose_signature})")
    
    # 3. Evaluate baseline
    print(f"\n{'='*60}")
    print(f"Evaluating BASELINE (unoptimized)")
    print(f"{'='*60}")
    
    baseline_results = evaluate_reranker(
        program=program,
        testset=valset,
        k=args.k,
        k_values=[1, 5, 10, 20],
        num_threads=args.num_threads,
        show_progress=True,
    )
    print_eval_results(baseline_results)
    save_eval_results(baseline_results, str(results_dir / f"{tag}_baseline.json"), tag=f"{tag}_baseline")
    
    # 4. Optimization (optional)
    if args.optimizer and not args.no_optimize:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING with {args.optimizer.upper()}")
        print(f"{'='*60}")
        
        log_dir = str(results_dir / f"{tag}_logs")
        
        if args.optimizer == "mipro":
            optimized_program = optimize_with_mipro(
                program=program,
                trainset=trainset,
                valset=valset,
                k=args.k,
                primary_metric=args.primary_metric,
                auto=args.auto,
                num_threads=args.num_threads,
                log_dir=log_dir,
            )
        elif args.optimizer == "gepa":
            optimized_program = optimize_with_gepa(
                program=program,
                trainset=trainset,
                valset=valset,
                k=args.k,
                primary_metric=args.primary_metric,
                auto=args.auto,
                max_metric_calls=args.max_metric_calls,
                reflection_model=args.reflection_lm,
                num_threads=args.num_threads,
                log_dir=log_dir,
            )
        
        # Save optimized program
        prog_path = str(results_dir / f"{tag}_optimized.json")
        save_optimized_program(optimized_program, prog_path)
        
        # 5. Evaluate optimized
        print(f"\n{'='*60}")
        print(f"Evaluating OPTIMIZED")
        print(f"{'='*60}")
        
        optimized_results = evaluate_reranker(
            program=optimized_program,
            testset=valset,
            k=args.k,
            k_values=[1, 5, 10, 20],
            num_threads=args.num_threads,
            show_progress=True,
        )
        print_eval_results(optimized_results)
        save_eval_results(optimized_results, str(results_dir / f"{tag}_optimized.json"), tag=f"{tag}_optimized")
        
        # 6. Compare
        comparison = compare_results(baseline_results, optimized_results)
        print(comparison)
        
        # Save comparison
        with open(results_dir / f"{tag}_comparison.txt", "w") as f:
            f.write(comparison)
    
    elif args.load_optimized:
        print(f"\nLoading optimized program from {args.load_optimized}")
        program = load_optimized_program(args.load_optimized, program)
        
        optimized_results = evaluate_reranker(
            program=program,
            testset=valset,
            k=args.k,
            k_values=[1, 5, 10, 20],
            num_threads=args.num_threads,
            show_progress=True,
        )
        print_eval_results(optimized_results)
        save_eval_results(optimized_results, str(results_dir / f"{tag}_loaded.json"), tag=f"{tag}_loaded")
    
    print(f"\nDone! Results saved to {results_dir}/")


if __name__ == "__main__":
    main()