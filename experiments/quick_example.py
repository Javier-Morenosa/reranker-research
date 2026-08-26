"""
Ejemplo mínimo de listwise reranking con DSPy.

export OLLAMA_API_KEY=your_key
python quick_example.py
"""

import os
import dspy
from reranker import ListwiseReranker, DocumentCandidate, ndcg_at_k, recall_at_k, mrr

# 1. Configurar LLM
lm = dspy.LM(
    "openai/glm-5.2",
    api_key=os.getenv("OLLAMA_API_KEY"),
    api_base="https://ollama.com/v1",
    cache=False,
)
dspy.configure(lm=lm)

# 2. Documentos candidatos (simulando first-stage retrieval)
query = "What is the capital of France?"
docs = [
    DocumentCandidate(id=1, text="Paris is the capital and largest city of France.", initial_rank=1),
    DocumentCandidate(id=2, text="Python is a popular programming language.", initial_rank=2),
    DocumentCandidate(id=3, text="The Eiffel Tower is located in Paris, France.", initial_rank=3),
    DocumentCandidate(id=4, text="Tokyo is the capital of Japan.", initial_rank=4),
    DocumentCandidate(id=5, text="The Louvre museum houses the Mona Lisa in Paris.", initial_rank=5),
]
relevant_ids = {1, 3, 5}

# 3. Rerankear
reranker = ListwiseReranker(top_k=5)
result = reranker(query=query, search_results=docs, top_k=5)
print(f"Query: {query}")
print(f"Reranked IDs: {result.reranked_ids}")

# 4. Métricas
print(f"\nnDCG@5:  {ndcg_at_k(result.reranked_ids, relevant_ids, k=5):.4f}")
print(f"Recall@5: {recall_at_k(result.reranked_ids, relevant_ids, k=5):.4f}")
print(f"MRR:     {mrr(result.reranked_ids, relevant_ids):.4f}")