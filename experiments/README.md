# Listwise Reranker con DSPy

Implementación de un reranker listwise basado en LLMs usando DSPy.

## Estructura

```
experiments/
├── reranker/
│   ├── __init__.py
│   ├── signatures.py    # DSPy Signatures (input/output schemas)
│   ├── module.py        # ListwiseReranker dspy.Module
│   ├── metrics.py       # IR metrics: nDCG@k, Recall@k, MRR
│   ├── dataset.py       # BEIR dataset loading + first-stage retrieval
│   ├── optimize.py      # MIPROv2 / GEPA optimizers
│   └── evaluate.py      # End-to-end evaluation pipeline
├── run_experiment.py    # Main script
├── requirements.txt
└── results/             # Saved optimized prompts + eval results
```

## Setup

```bash
cd experiments
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Uso

```bash
# Evaluación baseline (sin optimización)
python run_experiment.py --dataset scifact --k 10 --no-optimize

# Optimización con MIPROv2
python run_experiment.py --dataset scifact --k 10 --optimizer mipro

# Optimización con GEPA
python run_experiment.py --dataset scifact --k 10 --optimizer gepa
```