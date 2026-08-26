# Listwise Reranker con DSPy

Reranker listwise basado en LLMs usando DSPy. Optimización automática de prompts con MIPROv2 y GEPA.

## Setup

```bash
cd experiments
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...     # o OLLAMA_API_KEY=...
```

## Uso

```bash
python run_experiment.py --dataset scifact --k 10 --no-optimize
python run_experiment.py --dataset scifact --k 10 --optimizer mipro --auto light
python run_experiment.py --dataset scifact --lm ollama-cloud/glm-5.2 --optimizer gepa
```