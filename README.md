# reranker-research

Repositorio personal de investigación sobre **rerankers** para retrieval / RAG.

Centralizar papers, notas y experimentos en un solo sitio, en vez de PDFs dispersos por Downloads.

## Estructura

```
reranker-research/
├── papers/          # PDFs de papers relevantes (nombrados YYYY_ShortTitle.pdf)
├── notes/           # Notas en markdown por paper / tema (TODO)
├── experiments/     # Código y resultados de experimentos propios
│   ├── reranker/        # Paquete Python del reranker
│   │   ├── signatures.py    # DSPy Signatures
│   │   ├── module.py        # ListwiseReranker dspy.Module
│   │   ├── metrics.py       # nDCG@k, Recall@k, MRR, MAP@k
│   │   ├── dataset.py       # BEIR + first-stage retrieval (BM25/dense)
│   │   ├── optimize.py      # MIPROv2 / GEPA
│   │   └── evaluate.py      # Evaluación end-to-end
│   ├── run_experiment.py    # Script principal
│   ├── requirements.txt
│   └── results/         # JSONs con métricas
└── README.md
```

## Papers

### Listwise reranking con LLMs

| Año | Paper | Autores | Aportación clave |
|---|---|---|---|
| 2024 | [FIRST: Faster Improved Listwise Reranking with Single Token Decoding](papers/2024_FIRST_Faster_Listwise_Reranking_Single_Token.pdf) | Reddy et al. | Usa logits del primer token en vez de generar la secuencia ordenada completa → 50% más rápido sin perder calidad. Añade learning-to-rank loss en training. [arxiv](https://arxiv.org/abs/2406.15657) |
| 2023 | [RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!](papers/2023_RankZephyr_Zero_Shot_Listwise_Reranking.pdf) | Pradeep, Sharifymoghaddam, Lin | LLM open-source para reranking que cierra la brecha con GPT-4. Fuerte en benchmarks IR zero-shot. [arxiv](https://arxiv.org/abs/2312.02724) |

## Cómo añadir un paper nuevo

1. Descarga el PDF desde arxiv: `curl -L -o papers/YYYY_ShortTitle.pdf https://arxiv.org/pdf/XXXX.XXXXX`
2. Añade la fila a la tabla del README con link al PDF local y a arxiv.
3. (Opcional) Crea `notes/YYYY_ShortTitle.md` con resumen propio, ideas aplicables y dudas.

## Líneas de investigación abiertas

- Listwise vs pointwise vs pairwise — trade-offs reales en producción.
- Reranking eficiente: single-token decoding, distillation, modelos pequeños.
- Cross-encoders clásicos (BGE, Cohere Rerank) vs LLM-based rerankers.

## Experimentos: Listwise Reranker con DSPy

Reranker listwise basado en LLMs usando [DSPy](https://dspy.ai). Toma una query y los top-k documentos del first-stage retrieval, y usa un LLM para reordenarlos por relevancia. Optimización automática del prompt con MIPROv2 y GEPA.

### Setup

```bash
cd experiments
pip install -r requirements.txt

# OpenAI
export OPENAI_API_KEY=sk-...

# Ollama Cloud
export OLLAMA_API_KEY=your_key

# Ollama local (sin API key)
# https://ollama.com
```

### Uso

```bash
# Baseline
python run_experiment.py --dataset scifact --k 10 --no-optimize

# MIPROv2
python run_experiment.py --dataset scifact --k 10 --optimizer mipro --auto light

# GEPA
python run_experiment.py --dataset scifact --k 10 --optimizer gepa --auto light

# Ollama Cloud
python run_experiment.py --dataset scifact --lm ollama-cloud/glm-5.2 --optimizer mipro

# Ollama local
python run_experiment.py --dataset scifact --lm ollama/mistral:latest --optimizer mipro
```

### Arquitectura

1. **First-stage retrieval**: BM25 o dense (sentence-transformers) para obtener top-k candidatos del corpus BEIR.
2. **DSPy Signature**: `RelevanceRanker` recibe query + lista de documentos y devuelve los IDs reordenados.
3. **DSPy Module**: `ListwiseReranker` con `dspy.Predict` o `dspy.ChainOfThought`. Variante `SummarizedListwiseReranker` con etapa de summary previo.
4. **Optimización**: MIPROv2 (Bayesian Optimization) o GEPA (evolución reflexiva) optimizan el prompt usando nDCG@k como objetivo.

### Métricas

| Métrica | Descripción |
|---|---|
| nDCG@k | Calidad del ranking teniendo en cuenta la posición |
| Recall@k | Fracción de documentos relevantes en top-k |
| MRR | 1/rank del primer documento relevante |
| MAP@k | Average precision en cada posición relevante |

### Datasets BEIR soportados

| Dataset | Dominio | Queries | Documentos |
|---|---|---|---|
| scifact | Científico | 300 | 5,183 |
| nfcorpus | Nutrición | 323 | 36,333 |
| fiqa | Finanzas | 648 | 57,638 |
| trec-covid | Médico | 50 | 171,332 |
| arguana | Argumentación | 1,406 | 8,674 |
| scidocs | Científico | 1,000 | 25,657 |

## Resultados: BEIR SCIFACT + glm-5.2

18 ejemplos (9 train / 9 val), BM25 top-20 → LLM reranking top-10.

| Métrica | Baseline | MIPROv2 | GEPA |
|---|---|---|---|
| nDCG@1 | 66.7% | **77.8%** (+16.7%) | 66.7% |
| nDCG@5 | 79.2% | **84.8%** (+7.0%) | 80.7% (+1.8%) |
| nDCG@10 | 83.2% | **88.3%** (+6.1%) | 84.4% (+1.4%) |
| Recall@1 | 66.7% | **77.8%** (+16.7%) | 66.7% |
| Recall@10 | 100% | 100% | 100% |
| MRR | 77.8% | **84.7%** (+8.9%) | 79.4% (+2.0%) |

MIPROv2 gana por que añade few-shot demos además de optimizar la instrucción. GEPA solo muta la instrucción. MIPROv2 generó automáticamente una instrucción específica para el dominio biomédico sin que se le indicara.

| Aspecto | MIPROv2 | GEPA |
|---|---|---|
| Estrategia | Bayesian Optimization | Evolución reflexiva |
| Optimiza | Instrucción + few-shot demos | Solo instrucción |
| Feedback | Score numérico | Score + feedback textual |
| Mejor para | Pocos datos, demos efectivos | Mucho feedback, baseline débil |