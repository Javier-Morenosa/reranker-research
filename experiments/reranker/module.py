"""
ListwiseReranker: módulo DSPy para reranking listwise.
"""

import dspy
from typing import List, Optional
from .signatures import (
    DocumentCandidate,
    RelevanceRanker,
    VerboseRelevanceRanker,
    SummarizeSearchRelevance,
)


class ListwiseReranker(dspy.Module):
    """Reranker listwise. Toma una query y top-k candidatos del first-stage
    retriever y los reordena por relevancia usando un LLM.
    
    Args:
        top_k: Número de documentos a devolver tras el reranking.
        use_cot: ChainOfThought en vez de Predict (más reasoning, más caro).
        verbose_signature: Usar VerboseRelevanceRanker con criterios detallados.
    """
    
    def __init__(self, top_k: int = 10, use_cot: bool = False, verbose_signature: bool = False):
        super().__init__()
        self.top_k = top_k
        signature = VerboseRelevanceRanker if verbose_signature else RelevanceRanker
        if use_cot:
            self.reranker = dspy.ChainOfThought(signature)
        else:
            self.reranker = dspy.Predict(signature)
    
    def forward(self, query: str, search_results: List[DocumentCandidate],
                top_k: Optional[int] = None) -> dspy.Prediction:
        effective_k = top_k or self.top_k
        
        pred = self.reranker(query=query, search_results=search_results, top_k=effective_k)
        
        # Validar IDs y completar si el LLM devolvió menos de los necesarios
        valid_ids = {doc.id for doc in search_results}
        reranked_ids = []
        for doc_id in pred.reranked_ids:
            if doc_id in valid_ids and doc_id not in reranked_ids:
                reranked_ids.append(doc_id)
        
        returned_set = set(reranked_ids)
        remaining = [doc.id for doc in search_results if doc.id not in returned_set]
        reranked_ids.extend(remaining[:effective_k - len(reranked_ids)])
        reranked_ids = reranked_ids[:effective_k]
        
        return dspy.Prediction(
            reranked_ids=reranked_ids,
            reasoning=getattr(pred, "reasoning", ""),
        )


class SummarizedListwiseReranker(dspy.Module):
    """Reranker listwise con summarization previo. En vez de pasar el texto
    completo de cada documento al reranker, primero genera un summary de
    relevancia por documento y luego rerankea sobre esos summaries.
    
    Reduce tokens con documentos largos a costa de doble llamada al LLM.
    
    Args:
        top_k: Número de documentos a devolver.
        use_cot: ChainOfThought para el reranking.
    """
    
    def __init__(self, top_k: int = 10, use_cot: bool = True):
        super().__init__()
        self.top_k = top_k
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)
        self.reranker = dspy.ChainOfThought(RelevanceRanker) if use_cot else dspy.Predict(RelevanceRanker)
    
    def forward(self, query: str, search_results: List[DocumentCandidate],
                top_k: Optional[int] = None) -> dspy.Prediction:
        effective_k = top_k or self.top_k
        
        # Etapa 1: Summarizar relevancia de cada documento
        summarized_results = []
        for doc in search_results:
            summary_pred = self.summarizer(query=query, passage=doc.text)
            summarized_results.append(
                DocumentCandidate(
                    id=doc.id,
                    text=summary_pred.relevance_summary,
                    initial_rank=doc.initial_rank,
                )
            )
        
        # Etapa 2: Reranking sobre summaries
        pred = self.reranker(query=query, search_results=summarized_results, top_k=effective_k)
        
        valid_ids = {doc.id for doc in search_results}
        reranked_ids = []
        for doc_id in pred.reranked_ids:
            if doc_id in valid_ids and doc_id not in reranked_ids:
                reranked_ids.append(doc_id)
        
        returned_set = set(reranked_ids)
        remaining = [doc.id for doc in search_results if doc.id not in returned_set]
        reranked_ids.extend(remaining[:effective_k - len(reranked_ids)])
        reranked_ids = reranked_ids[:effective_k]
        
        return dspy.Prediction(
            reranked_ids=reranked_ids,
            reasoning=getattr(pred, "reasoning", ""),
        )