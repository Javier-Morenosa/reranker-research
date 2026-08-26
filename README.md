# reranker-research

Repositorio personal de investigación sobre **rerankers** para retrieval / RAG.

Objetivo: centralizar papers, notas y experimentos para maximizar productividad investigadora. Una sola fuente de verdad, en vez de PDFs dispersos por Downloads.

## Estructura

```
reranker-research/
├── papers/          # PDFs de papers relevantes (nombrados YYYY_ShortTitle.pdf)
├── notes/           # Notas en markdown por paper / tema (TODO)
├── experiments/     # Código y resultados de experimentos propios
│   ├── reranker/        # Paquete Python del reranker
│   │   ├── signatures.py    # DSPy Signatures (input/output schemas)
│   │   ├── module.py        # ListwiseReranker dspy.Module
│   │   ├── metrics.py       # IR metrics: nDCG@k, Recall@k, MRR, MAP@k
│   │   ├── dataset.py       # BEIR dataset loading + first-stage retrieval
│   │   ├── optimize.py      # MIPROv2 / GEPA optimizers
│   │   └── evaluate.py      # End-to-end evaluation pipeline
│   ├── run_experiment.py    # Script principal
│   ├── requirements.txt
│   └── results/         # Resultados guardados
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

Implementación de un reranker listwise basado en LLMs usando [DSPy](https://dspy.ai).
El reranker toma una query y los top-k documentos del first-stage retrieval, y usa
un LLM para reordenarlos por relevancia.

### Setup

```bash
cd experiments
pip install -r requirements.txt

# Opción 1: OpenAI
export OPENAI_API_KEY=sk-...

# Opción 2: Ollama Cloud (modelos cloud potentes, necesita suscripción)
export OLLAMA_API_KEY=your_key

# Opción 3: Ollama local (gratis, corre en tu máquina)
# No necesita API key. Instalar: https://ollama.com
```

### Uso

```bash
# Evaluación baseline (sin optimización)
python run_experiment.py --dataset scifact --k 10 --no-optimize

# Optimización con MIPROv2 (Bayesian Optimization sobre prompts + few-shot)
python run_experiment.py --dataset scifact --k 10 --optimizer mipro --auto light

# Optimización con GEPA (evolución reflexiva de prompts con feedback)
python run_experiment.py --dataset scifact --k 10 --optimizer gepa --auto medium

# Con Ollama Cloud (modelos potentes en la nube)
python run_experiment.py --dataset scifact --lm ollama-cloud/glm-5.2 --optimizer mipro
python run_experiment.py --dataset scifact --lm ollama-cloud/gpt-oss:120b --optimizer gepa --reflection-lm ollama-cloud/glm-5.2

# Con Ollama local (gratis, sin API key)
python run_experiment.py --dataset scifact --lm ollama/mistral:latest --optimizer mipro

# Quick test (20 ejemplos)
python run_experiment.py --dataset scifact --max-examples 20 --k 5 --no-optimize
```

### Modelos disponibles en Ollama Cloud

| Modelo | Tamaño | Bueno para |
|---|---|---|
| gpt-oss:20b | 20B | Rápido y barato, ideal para experimentos |
| gpt-oss:120b | 120B | Más calidad, ideal para reranking |
| glm-5.2 | - | Potente, buena relación calidad/precio |
| deepseek-v4-flash | - | Rápido, eficiente |
| kimi-k3 | - | Muy potente para razonamiento |
| mistral-large-3 | 675B | Alto rendimiento |

### Arquitectura

El pipeline tiene 4 componentes:

1. **First-stage retrieval**: BM25 o dense retrieval (sentence-transformers) para
   obtener top-k candidatos del corpus BEIR.

2. **DSPy Signature**: Define el contrato input/output del LLM. La signature
   `RelevanceRanker` recibe query + lista de documentos y devuelve los IDs
   reordenados por relevancia.

3. **DSPy Module**: `ListwiseReranker` encapsula la lógica de inferencia.
   Soporta `dspy.Predict` (rápido) o `dspy.ChainOfThought` (con reasoning).
   Variante `SummarizedListwiseReranker` añade una etapa de summary previo.

4. **Optimización**: MIPROv2 o GEPA optimizan automáticamente el prompt del
   reranker usando las métricas IR (nDCG@k, Recall@k) como objetivo.

### Métricas

| Métrica | Descripción |
|---|---|
| nDCG@k | Normalized Discounted Cumulative Gain — mide la calidad del ranking teniendo en cuenta la posición |
| Recall@k | Fracción de documentos relevantes que aparecen en el top-k |
| MRR | Mean Reciprocal Rank — 1/rank del primer documento relevante |
| MAP@k | Mean Average Precision — promedio de precision en cada posición relevante |

### Datasets BEIR soportados

| Dataset | Dominio | Queries | Documentos |
|---|---|---|---|
| scifact | Científico | 300 | 5,183 |
| nfcorpus | Nutrición | 323 | 36,333 |
| fiqa | Finanzas | 648 | 57,638 |
| trec-covid | Médico | 50 | 171,332 |
| arguana | Argumentación | 1,406 | 8,674 |
| scidocs | Científico | 1,000 | 25,657 |

## Resultados experimentales

### BEIR SCIFACT + glm-5.2 (Ollama Cloud)

Experimento real con 18 ejemplos (9 train / 9 val), BM25 top-20 retrieval, LLM listwise reranking top-10.
Métrica objetivo: nDCG@10.

| Métrica | Baseline | MIPROv2 | GEPA | MIPROv2 Δ | GEPA Δ |
|---|---|---|---|---|---|
| nDCG@1 | 66.7% | **77.8%** | 66.7% | +16.7% | +0.0% |
| nDCG@5 | 79.2% | **84.8%** | 80.7% | +7.0% | +1.8% |
| nDCG@10 | 83.2% | **88.3%** | 84.4% | +6.1% | +1.4% |
| Recall@1 | 66.7% | **77.8%** | 66.7% | +16.7% | +0.0% |
| Recall@5 | 88.9% | 88.9% | 88.9% | +0.0% | +0.0% |
| Recall@10 | 100% | 100% | 100% | +0.0% | +0.0% |
| MRR | 77.8% | **84.7%** | 79.4% | +8.9% | +2.0% |
| MAP@10 | 77.8% | **84.7%** | 79.4% | +8.9% | +2.0% |

**MIPROv2 gana en todas las métricas donde hay mejora.** La diferencia clave: MIPROv2
añade few-shot demos al prompt (ejemplos de reranking correcto) además de optimizar
la instrucción. GEPA solo muta la instrucción mediante evolución reflexiva.

### Lo que MIPROv2 aprendió automáticamente

MIPROv2 generó una instrucción específica para el dominio biomédico de SCIFACT
sin que se le indicara explícitamente:

> *"You are an expert biomedical scientist and information retrieval specialist.
> Your task is to rerank a list of research paper snippets based on their semantic
> relevance to a given biomedical claim. The query is a concise declarative
> statement about a medical or biological mechanism..."*

Y añadió 2 few-shot demos con ejemplos de reranking correcto.

### Lo que GEPA aprendió

GEPA usó el feedback textual de la métrica ("Missed 2 relevant documents...",
"Query was: '...'") para reflexionar sobre los errores y proponer mutaciones
con conocimiento del dominio específico (PPM1D, p53, APOE4, vCJD). Generó prompts
muy elaborados pero la mejora fue modesta porque el baseline ya era muy fuerte
(83.2% nDCG@10).

### Análisis

1. **MIPROv2 > GEPA en este setup**: Los few-shot demos son más efectivos que la
   mutación de instrucciones para reranking. MIPROv2 usa Bayesian Optimization
   para buscar el mejor combo instrucción + demos.

2. **GEPA sería mejor con**: un baseline más débil (más margen de mejora), más datos
   (más trayectorias imperfectas para reflexionar), o un modelo de reflexión más
   potente.

3. **Recall@10 ya está en 100%**: el reranker encuentra todos los docs relevantes.
   La mejora está en ORDENARLOS mejor, no en encontrar más.

4. **La mejora mayor es en nDCG@1 (+16.7%)**: MIPROv2 mejoró especialmente la
   posición 1 del ranking. El doc más relevante aparece arriba con más frecuencia.

### Comparación de optimizadores

| Aspecto | MIPROv2 | GEPA |
|---|---|---|
| Estrategia | Bayesian Optimization (optuna) | Evolución reflexiva |
| Qué optimiza | Instrucción + few-shot demos | Solo instrucción |
| Feedback | Score numérico | Score + feedback textual |
| Búsqueda | Sistemática (Bayesian) | Exploratoria (Pareto frontier) |
| Mejor para | Pocos datos, demos efectivos | Mucho feedback textual, baseline débil |
| Dependencias | optuna | gepa (incluido en dspy) |
