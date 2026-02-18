"""
Reasoning and Retrieval Engine

Advanced multi-stage retrieval pipeline:
1. Dense retrieval (vector search)
2. Reranking (cross-encoder)
3. Graph expansion (knowledge graph)

Features:
- Chain-of-thought reasoning
- Multi-hop question answering
- Context-aware retrieval
- Explainable results
"""

import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    content: str
    score: float
    source: str
    metadata: Dict = field(default_factory=dict)
    reasoning: Optional[str] = None


@dataclass
class ReasoningTrace:
    """Chain-of-thought reasoning trace."""
    query: str
    steps: List[str]
    retrieved_contexts: List[str]
    answer: str
    confidence: float
    sources: List[str]


class DenseRetriever:
    """
    Dense retrieval using vector similarity search.
    First stage of the retrieval pipeline.
    """
    
    def __init__(self, vector_db, embedding_engine):
        self.vector_db = vector_db
        self.embedding_engine = embedding_engine
    
    def retrieve(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using dense embeddings.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of retrieval results
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)
        
        # Search vector database
        raw_results = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Convert to RetrievalResult objects
        results = []
        for res in raw_results:
            results.append(RetrievalResult(
                content=res['content'],
                score=res['score'],
                source='dense_retrieval',
                metadata={
                    'id': res['id'],
                    'timestamp': res.get('timestamp'),
                    'retrieval_time': time.time() - start_time
                }
            ))
        
        return results


class Reranker:
    """
    Reranking stage using cross-encoder or LLM-based scoring.
    Second stage of the retrieval pipeline.
    """
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """
        Rerank results based on query-document relevance.
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to keep
            
        Returns:
            Reranked results
        """
        start_time = time.time()
        
        # For each result, compute relevance score
        reranked = []
        for result in results:
            # In production, use a cross-encoder model
            # For now, use a heuristic based on keyword overlap
            relevance_score = self._compute_relevance(query, result.content)
            
            # Combine with original score
            combined_score = 0.6 * result.score + 0.4 * relevance_score
            
            result.score = combined_score
            result.metadata['rerank_time'] = time.time() - start_time
            reranked.append(result)
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:top_k]
    
    def _compute_relevance(self, query: str, content: str) -> float:
        """
        Compute relevance score between query and content.
        Simple keyword-based heuristic (in production, use cross-encoder).
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Jaccard similarity
        intersection = query_words & content_words
        union = query_words | content_words
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)


class GraphExpander:
    """
    Graph expansion stage using knowledge graph.
    Third stage of the retrieval pipeline.
    """
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
    
    def expand(self, results: List[RetrievalResult], max_hops: int = 2) -> List[RetrievalResult]:
        """
        Expand results using knowledge graph traversal.
        
        Args:
            results: Current retrieval results
            max_hops: Maximum graph traversal depth
            
        Returns:
            Expanded results with related content
        """
        expanded = list(results)  # Start with original results
        
        # Extract entities from top results
        entities = self._extract_entities(results[:5])
        
        # For each entity, find neighbors in knowledge graph
        for entity in entities:
            neighbors = self.knowledge_graph.get_neighbors(entity, depth=max_hops)
            
            # Add neighbors as additional context
            for neighbor in neighbors[:5]:  # Limit expansion
                expanded.append(RetrievalResult(
                    content=f"Related entity: {neighbor['name']} ({neighbor.get('type', 'unknown')})",
                    score=0.5,  # Lower score for expanded content
                    source='graph_expansion',
                    metadata={'entity': entity, 'neighbor': neighbor}
                ))
        
        return expanded
    
    def _extract_entities(self, results: List[RetrievalResult]) -> List[str]:
        """
        Extract entities from retrieval results.
        Simple heuristic: capitalize words (in production, use NER).
        """
        entities = set()
        for result in results:
            words = result.content.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    entities.add(word.strip('.,!?'))
        
        return list(entities)[:10]  # Limit entities


class RetrievalPipeline:
    """
    Complete 3-stage retrieval pipeline:
    1. Dense retrieval (50 results)
    2. Reranking (10 results)
    3. Graph expansion (additional context)
    """
    
    def __init__(self, vector_db, embedding_engine, knowledge_graph):
        self.dense_retriever = DenseRetriever(vector_db, embedding_engine)
        self.reranker = Reranker(embedding_engine)
        self.graph_expander = GraphExpander(knowledge_graph)
        
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        use_expansion: bool = True
    ) -> List[RetrievalResult]:
        """
        Execute full retrieval pipeline.
        
        Args:
            query: Search query
            top_k: Number of final results
            use_reranking: Whether to apply reranking
            use_expansion: Whether to apply graph expansion
            
        Returns:
            Final retrieval results
        """
        start_time = time.time()
        
        # Stage 1: Dense retrieval
        results = self.dense_retriever.retrieve(query, top_k=50)
        
        # Stage 2: Reranking
        if use_reranking and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        # Stage 3: Graph expansion
        if use_expansion:
            results = self.graph_expander.expand(results, max_hops=2)
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + elapsed)
            / self.stats['total_queries']
        )
        
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'total_queries': self.stats['total_queries'],
            'avg_retrieval_time_ms': round(self.stats['avg_retrieval_time'] * 1000, 2)
        }


class ReasoningEngine:
    """
    Chain-of-thought reasoning engine for question answering.
    """
    
    def __init__(self, retrieval_pipeline, inference_engine):
        self.retrieval_pipeline = retrieval_pipeline
        self.inference_engine = inference_engine
    
    def answer_question(self, question: str, max_context: int = 5) -> ReasoningTrace:
        """
        Answer a question using chain-of-thought reasoning.
        
        Args:
            question: User question
            max_context: Maximum number of context documents
            
        Returns:
            Reasoning trace with answer
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        reasoning_steps = ["Retrieving relevant information..."]
        results = self.retrieval_pipeline.retrieve(question, top_k=max_context)
        
        contexts = [r.content for r in results[:max_context]]
        sources = [r.metadata.get('id', 'unknown') for r in results[:max_context]]
        
        reasoning_steps.append(f"Retrieved {len(contexts)} relevant documents")
        
        # Step 2: Analyze and synthesize
        reasoning_steps.append("Analyzing context and formulating answer...")
        
        # Build prompt for LLM
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Given the following context, answer the question using chain-of-thought reasoning.

Context:
{context_text}

Question: {question}

Think step by step:
1. What information is relevant from the context?
2. How do these pieces of information relate to each other?
3. What is the final answer?

Answer:"""
        
        # Step 3: Generate answer
        answer = self.inference_engine.generate(prompt, max_tokens=256, temperature=0.3)
        
        reasoning_steps.append("Generated final answer")
        
        # Step 4: Compute confidence
        confidence = self._compute_confidence(question, contexts, answer)
        
        reasoning_steps.append(f"Confidence: {confidence:.2f}")
        
        return ReasoningTrace(
            query=question,
            steps=reasoning_steps,
            retrieved_contexts=contexts,
            answer=answer,
            confidence=confidence,
            sources=sources
        )
    
    def _compute_confidence(self, question: str, contexts: List[str], answer: str) -> float:
        """
        Compute confidence score for the answer.
        Based on context overlap and answer length.
        """
        if not contexts or not answer:
            return 0.0
        
        # Simple heuristic: more context overlap = higher confidence
        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())
        
        overlap = len(answer_words & context_words)
        total = len(answer_words)
        
        if total == 0:
            return 0.0
        
        base_confidence = overlap / total
        
        # Boost if answer is not too short or too long
        if 20 < len(answer) < 500:
            base_confidence *= 1.2
        
        return min(1.0, base_confidence)
    
    def multi_hop_reasoning(self, question: str, max_hops: int = 3) -> ReasoningTrace:
        """
        Perform multi-hop reasoning for complex questions.
        
        Args:
            question: Complex question requiring multiple reasoning steps
            max_hops: Maximum number of reasoning hops
            
        Returns:
            Reasoning trace with multi-hop answer
        """
        reasoning_steps = [f"Starting multi-hop reasoning (max {max_hops} hops)"]
        all_contexts = []
        all_sources = []
        
        current_query = question
        
        for hop in range(max_hops):
            reasoning_steps.append(f"Hop {hop + 1}: Querying '{current_query[:50]}...'")
            
            # Retrieve for current query
            results = self.retrieval_pipeline.retrieve(current_query, top_k=3)
            
            hop_contexts = [r.content for r in results]
            hop_sources = [r.metadata.get('id', 'unknown') for r in results]
            
            all_contexts.extend(hop_contexts)
            all_sources.extend(hop_sources)
            
            # Check if we have enough information
            if len(all_contexts) >= 5:
                reasoning_steps.append("Sufficient context gathered")
                break
            
            # Generate next query based on current context
            if hop < max_hops - 1:
                current_query = self._generate_followup_query(question, hop_contexts)
        
        # Generate final answer
        reasoning_steps.append("Synthesizing final answer from all hops...")
        
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(all_contexts)])
        
        prompt = f"""Using the following information gathered through multi-hop reasoning, answer the question.

Context:
{context_text}

Question: {question}

Final Answer:"""
        
        answer = self.inference_engine.generate(prompt, max_tokens=300, temperature=0.3)
        
        confidence = self._compute_confidence(question, all_contexts, answer)
        
        return ReasoningTrace(
            query=question,
            steps=reasoning_steps,
            retrieved_contexts=all_contexts,
            answer=answer,
            confidence=confidence,
            sources=all_sources
        )
    
    def _generate_followup_query(self, original_question: str, contexts: List[str]) -> str:
        """
        Generate a follow-up query based on current context.
        In production, use LLM to generate. For now, simple heuristic.
        """
        # Extract key terms from contexts
        key_terms = []
        for ctx in contexts:
            words = ctx.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 3:
                    key_terms.append(word)
        
        if key_terms:
            return f"{original_question} {key_terms[0]}"
        
        return original_question


if __name__ == '__main__':
    # Demo usage
    print("=== Reasoning and Retrieval Engine Demo ===\n")
    
    # Mock components for demo
    from vector_db import MilvusVectorDB
    from nvidia_inference import NVIDIAInferenceEngine
    from core import KnowledgeGraph
    
    # Initialize components
    vector_db = MilvusVectorDB()
    embedding_engine = NVIDIAInferenceEngine()
    knowledge_graph = KnowledgeGraph()
    
    # Add sample data
    docs = [
        "NVIDIA provides GPU-accelerated computing for AI and machine learning.",
        "Vector databases enable semantic search using embeddings.",
        "Knowledge graphs represent relationships between entities.",
        "Chain-of-thought reasoning improves language model performance."
    ]
    
    for i, doc in enumerate(docs):
        doc_id = f"doc{i+1}"
        embedding = embedding_engine.embed(doc)
        vector_db.insert(doc_id, embedding, doc)
    
    # Create pipeline
    pipeline = RetrievalPipeline(vector_db, embedding_engine, knowledge_graph)
    
    # Test retrieval
    print("1. Testing retrieval pipeline:")
    query = "How does GPU acceleration help with AI?"
    results = pipeline.retrieve(query, top_k=3)
    
    for i, result in enumerate(results):
        print(f"   Result {i+1}:")
        print(f"      Content: {result.content[:80]}...")
        print(f"      Score: {result.score:.4f}")
        print(f"      Source: {result.source}")
    
    # Test reasoning
    print("\n2. Testing reasoning engine:")
    reasoning_engine = ReasoningEngine(pipeline, embedding_engine)
    
    question = "What technologies enable semantic search?"
    trace = reasoning_engine.answer_question(question, max_context=3)
    
    print(f"   Question: {trace.query}")
    print(f"   Reasoning steps:")
    for step in trace.steps:
        print(f"      - {step}")
    print(f"   Answer: {trace.answer[:100]}...")
    print(f"   Confidence: {trace.confidence:.2f}")
    
    # Statistics
    print("\n3. Pipeline statistics:")
    stats = pipeline.get_stats()
    print(json.dumps(stats, indent=2))
