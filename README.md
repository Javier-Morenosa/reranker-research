# reranker-research

Repositorio personal de investigación sobre **rerankers** para retrieval / RAG.

Objetivo: centralizar papers, notas y experimentos para maximizar productividad investigadora. Una sola fuente de verdad, en vez de PDFs dispersos por Downloads.

## Estructura

```
reranker-research/
├── papers/      # PDFs de papers relevantes (nombrados YYYY_ShortTitle.pdf)
├── notes/       # Notas en markdown por paper / tema (TODO)
└── experiments/ # Código y resultados de experimentos propios (TODO)
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
