"""
Second Brain Package

Production-grade second brain with NVIDIA stack.

Components:
- core: Multi-tier memory system + knowledge graph
- nvidia_inference: NVIDIA NIM + TensorRT integration
- vector_db: Milvus GPU-accelerated vector database
- reasoning: 3-stage retrieval + chain-of-thought reasoning
- server: REST API with 15+ endpoints
- cli: Interactive command-line interface

Usage:
    from brain import SecondBrain
    
    brain = SecondBrain()
    brain.ingest("Your knowledge here")
    results = brain.retrieve("search query")
"""

__version__ = "1.0.0"
__author__ = "Second Brain Team"

from .core import SecondBrain, MemoryStore, KnowledgeGraph, MemoryNode
from .nvidia_inference import NVIDIAInferenceEngine, InferenceConfig
from .vector_db import MilvusVectorDB, VectorConfig
from .reasoning import RetrievalPipeline, ReasoningEngine, RetrievalResult

__all__ = [
    # Core
    "SecondBrain",
    "MemoryStore",
    "KnowledgeGraph",
    "MemoryNode",
    
    # Inference
    "NVIDIAInferenceEngine",
    "InferenceConfig",
    
    # Vector DB
    "MilvusVectorDB",
    "VectorConfig",
    
    # Reasoning
    "RetrievalPipeline",
    "ReasoningEngine",
    "RetrievalResult",
]
