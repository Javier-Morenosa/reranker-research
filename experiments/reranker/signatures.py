"""
DSPy Signatures para listwise reranking.
"""

import dspy
from typing import List
from pydantic import BaseModel, Field


class DocumentCandidate(BaseModel):
    id: int = Field(description="Identificador único del documento (1-indexed)")
    text: str = Field(description="Contenido del documento")
    initial_rank: int = Field(description="Ranking original del first-stage retriever")


class RelevanceRanker(dspy.Signature):
    """Given a query and a list of search results, rerank them by relevance to the query.
    
    Return the IDs of the documents in descending order of relevance (most relevant first).
    Only return the top_k most relevant document IDs.
    """
    query: str = dspy.InputField(desc="The user's question or information need")
    search_results: List[DocumentCandidate] = dspy.InputField(
        desc="List of candidate documents to rerank, each with id, text, and initial_rank"
    )
    top_k: int = dspy.InputField(desc="Number of top documents to return after reranking")
    reranked_ids: List[int] = dspy.OutputField(
        desc="List of document IDs in descending order of relevance (most relevant first), length <= top_k"
    )


class VerboseRelevanceRanker(dspy.Signature):
    """You are an expert search reranker. Given a query and a list of candidate documents,
    carefully analyze each document's relevance to the query and produce an optimal ranking.
    
    Consider the following when ranking:
    1. Direct relevance: Does the document directly address the query's information need?
    2. Information completeness: Does the document provide a complete answer or only partial?
    3. Topical authority: Is the document authoritative on the topic?
    4. Specificity: Does the document contain specific, actionable information?
    5. Comparative quality: How does this document compare to the other candidates?
    
    Return the IDs of the documents in descending order of relevance.
    """
    query: str = dspy.InputField(desc="The user's question or information need")
    search_results: List[DocumentCandidate] = dspy.InputField(
        desc="List of candidate documents to rerank"
    )
    top_k: int = dspy.InputField(desc="Number of top documents to return")
    reasoning: str = dspy.OutputField(desc="Brief reasoning about why the documents were ranked this way")
    reranked_ids: List[int] = dspy.OutputField(
        desc="List of document IDs in descending order of relevance, length <= top_k"
    )


class AssessRelevance(dspy.Signature):
    """Assess whether or not the candidate document is relevant to the query."""
    query: str = dspy.InputField(desc="The user's question or information need")
    candidate_document: str = dspy.InputField(desc="The candidate document to assess")
    relevance_assessment: bool = dspy.OutputField(
        desc="Whether or not the candidate document is relevant to the query"
    )


class SummarizeSearchRelevance(dspy.Signature):
    """Given a query and a passage, summarize how the passage is (or isn't) relevant
    to answering the query.
    """
    query: str = dspy.InputField(desc="The user's question or information need")
    passage: str = dspy.InputField(desc="The candidate document passage")
    relevance_summary: str = dspy.OutputField(
        desc="A concise summary of how this passage relates to the query"
    )