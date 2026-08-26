"""
ListwiseReranker: módulo DSPy para reranking listwise.

Un dspy.Module encapsula el flujo de inferencia. DSPy puede optimizar
automáticamente los prompts dentro de este módulo usando MIPROv2 o GEPA.
"""

import dspy
from typing import List, Optional, Dict, Any
from .signatures import (
    DocumentCandidate,
    RelevanceRanker,
    VerboseRelevanceRanker,
    SummarizeSearchRelevance,
)


class ListwiseReranker(dspy.Module):
    """
    Reranker listwise básico.
    
    Toma una query y una lista de documentos candidatos (típicamente
    top-k del first-stage retriever) y devuelve los documentos reordenados
    por relevancia usando un LLM.
    
    Args:
        top_k: Número de documentos a devolver tras el reranking.
        use_cot: Si True, usa ChainOfThought en vez de Predict (más reasoning,
            más caro). Recomendado para queries difíciles.
        verbose_signature: Si True, usa VerboseRelevanceRanker con instrucciones
            más detalladas sobre los criterios de ranking.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        use_cot: bool = False,
        verbose_signature: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        signature = VerboseRelevanceRanker if verbose_signature else RelevanceRanker
        if use_cot:
            self.reranker = dspy.ChainOfThought(signature)
        else:
            self.reranker = dspy.Predict(signature)
    
    def forward(
        self,
        query: str,
        search_results: List[DocumentCandidate],
        top_k: Optional[int] = None,
    ) -> dspy.Prediction:
        """
        Args:
            query: La consulta del usuario.
            search_results: Lista de DocumentCandidate con id, text, initial_rank.
            top_k: Override del top_k del constructor.
            
        Returns:
            dspy.Prediction con reranked_ids (lista de IDs ordenados por relevancia).
        """
        effective_k = top_k or self.top_k
        
        pred = self.reranker(
            query=query,
            search_results=search_results,
            top_k=effective_k,
        )
        
        # Validar que los IDs devueltos son válidos
        valid_ids = {doc.id for doc in search_results}
        reranked_ids = []
        for doc_id in pred.reranked_ids:
            if doc_id in valid_ids and doc_id not in reranked_ids:
                reranked_ids.append(doc_id)
        
        # Si el LLM devolvió menos IDs de los necesarios, completar con los que faltan
        returned_set = set(reranked_ids)
        remaining = [doc.id for doc in search_results if doc.id not in returned_set]
        reranked_ids.extend(remaining[: effective_k - len(reranked_ids)])
        
        # Truncar a top_k
        reranked_ids = reranked_ids[:effective_k]
        
        return dspy.Prediction(
            reranked_ids=reranked_ids,
            reasoning=getattr(pred, "reasoning", ""),
        )


class SummarizedListwiseReranker(dspy.Module):
    """
    Reranker listwise con etapa de summarization.
    
    En vez de pasar el texto completo de cada documento al reranker,
    primero genera un summary de la relevancia de cada documento
    respecto a la query, y luego hace el reranking sobre esos summaries.
    
    Ventajas:
    - Reduce el coste en tokens del reranking (especialmente con documentos largos).
    - El reranker ve información pre-filtrada y condensada.
    
    Desventajas:
    - Doble llamada al LLM (N summaries + 1 reranking).
    - Puede perder matices del documento original.
    
    Args:
        top_k: Número de documentos a devolver.
        use_cot: Si True, usa ChainOfThought para el reranking.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        use_cot: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)
        self.reranker = dspy.ChainOfThought(RelevanceRanker) if use_cot else dspy.Predict(RelevanceRanker)
    
    def forward(
        self,
        query: str,
        search_results: List[DocumentCandidate],
        top_k: Optional[int] = None,
    ) -> dspy.Prediction:
        effective_k = top_k or self.top_k
        
        # Etapa 1: Summarizar la relevancia de cada documento
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
        
        # Etapa 2: Reranking sobre los summaries
        pred = self.reranker(
            query=query,
            search_results=summarized_results,
            top_k=effective_k,
        )
        
        # Validar IDs
        valid_ids = {doc.id for doc in search_results}
        reranked_ids = []
        for doc_id in pred.reranked_ids:
            if doc_id in valid_ids and doc_id not in reranked_ids:
                reranked_ids.append(doc_id)
        
        returned_set = set(reranked_ids)
        remaining = [doc.id for doc in search_results if doc.id not in returned_set]
        reranked_ids.extend(remaining[: effective_k - len(reranked_ids)])
        reranked_ids = reranked_ids[:effective_k]
        
        return dspy.Prediction(
            reranked_ids=reranked_ids,
            reasoning=getattr(pred, "reasoning", ""),
        )