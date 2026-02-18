"""
Second Brain REST API Server

FastAPI-based REST API with 15+ endpoints for:
- Memory management (CRUD operations)
- Search and retrieval
- Question answering
- Knowledge graph operations
- System monitoring

Endpoints:
- POST /api/ingest - Ingest new information
- POST /api/search - Semantic search
- POST /api/question - Question answering
- GET /api/memory/{id} - Get memory by ID
- GET /api/review - Get review queue
- POST /api/entity - Add entity to knowledge graph
- POST /api/relationship - Add relationship
- GET /api/neighbors/{entity} - Get entity neighbors
- GET /api/stats - System statistics
- GET /api/health - Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import time
from datetime import datetime
import json

# Import second brain components
from core import SecondBrain, MemoryNode
from nvidia_inference import NVIDIAInferenceEngine
from vector_db import MilvusVectorDB
from reasoning import RetrievalPipeline, ReasoningEngine

# Initialize FastAPI app
app = FastAPI(
    title="Second Brain API",
    description="Production-grade second brain with NVIDIA stack",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
brain = None
vector_db = None
inference_engine = None
retrieval_pipeline = None
reasoning_engine = None


# Request/Response Models
class IngestRequest(BaseModel):
    content: str = Field(..., description="Content to ingest")
    memory_type: str = Field("long_term", description="Memory type: long_term, short_term, episodic, semantic")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    entities: Optional[List[str]] = Field(None, description="Entities to extract and link")


class IngestResponse(BaseModel):
    memory_id: str
    status: str
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results")
    use_reranking: bool = Field(True, description="Apply reranking")
    use_expansion: bool = Field(True, description="Apply graph expansion")


class SearchResult(BaseModel):
    content: str
    score: float
    source: str
    metadata: Dict


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    max_context: int = Field(5, description="Maximum context documents")
    multi_hop: bool = Field(False, description="Use multi-hop reasoning")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    reasoning_steps: List[str]
    sources: List[str]


class EntityRequest(BaseModel):
    name: str
    entity_type: str
    properties: Optional[Dict] = None


class RelationshipRequest(BaseModel):
    entity1: str
    relation: str
    entity2: str
    properties: Optional[Dict] = None


class MemoryResponse(BaseModel):
    id: str
    content: str
    memory_type: str
    timestamp: float
    last_accessed: float
    access_count: int
    strength: float
    retention: float
    tags: List[str]
    metadata: Dict


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global brain, vector_db, inference_engine, retrieval_pipeline, reasoning_engine
    
    print("Starting Second Brain API...")
    
    # Initialize components
    brain = SecondBrain()
    vector_db = MilvusVectorDB()
    inference_engine = NVIDIAInferenceEngine()
    retrieval_pipeline = RetrievalPipeline(vector_db, inference_engine, brain.knowledge_graph)
    reasoning_engine = ReasoningEngine(retrieval_pipeline, inference_engine)
    
    print("Second Brain API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global brain, vector_db
    
    print("Shutting down Second Brain API...")
    
    if brain:
        brain.close()
    if vector_db:
        vector_db.close()
    
    print("Shutdown complete.")


# Health & Status Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "brain": brain is not None,
            "vector_db": vector_db is not None,
            "inference_engine": inference_engine is not None,
            "retrieval_pipeline": retrieval_pipeline is not None,
            "reasoning_engine": reasoning_engine is not None
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get comprehensive system statistics."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = brain.get_comprehensive_stats()
    
    # Add component stats
    if vector_db:
        stats['vector_db'] = vector_db.get_stats()
    if inference_engine:
        stats['inference_engine'] = inference_engine.get_stats()
    if retrieval_pipeline:
        stats['retrieval_pipeline'] = retrieval_pipeline.get_stats()
    
    return stats


# Memory Management Endpoints
@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_content(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest new content into the second brain.
    Stores in memory, generates embeddings, and links entities.
    """
    if not brain or not inference_engine or not vector_db:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Ingest into memory system
        memory_id = brain.ingest(
            content=request.content,
            memory_type=request.memory_type,
            tags=request.tags,
            entities=request.entities
        )
        
        # Generate embedding and store in vector DB (async)
        async def store_embedding():
            embedding = inference_engine.embed(request.content)
            vector_db.insert(
                id=memory_id,
                embedding=embedding,
                content=request.content,
                metadata={'tags': request.tags or [], 'memory_type': request.memory_type}
            )
        
        background_tasks.add_task(store_embedding)
        
        return IngestResponse(
            memory_id=memory_id,
            status="success",
            message="Content ingested successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/memory/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    memory = brain.memory_store.get_memory(memory_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        memory_type=memory.memory_type,
        timestamp=memory.timestamp,
        last_accessed=memory.last_accessed,
        access_count=memory.access_count,
        strength=memory.strength,
        retention=memory.calculate_retention(),
        tags=memory.tags,
        metadata=memory.metadata
    )


@app.get("/api/review")
async def get_review_queue(limit: int = 10):
    """Get memories that need review (spaced repetition)."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    memories = brain.get_review_queue(limit)
    
    return {
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                "last_accessed": m.last_accessed,
                "strength": m.strength,
                "retention": m.calculate_retention(),
                "should_review": m.should_review()[0],
                "days_until_next": m.should_review()[1]
            }
            for m in memories
        ]
    }


@app.post("/api/cleanup")
async def cleanup_memories(threshold: float = 0.1):
    """Remove weak memories below retention threshold."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    removed_count = brain.cleanup(threshold)
    
    return {
        "status": "success",
        "removed_count": removed_count,
        "threshold": threshold
    }


# Search & Retrieval Endpoints
@app.post("/api/search")
async def search_content(request: SearchRequest):
    """
    Semantic search using the retrieval pipeline.
    Returns ranked results with scores.
    """
    if not retrieval_pipeline:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_expansion=request.use_expansion
        )
        
        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question using chain-of-thought reasoning.
    Retrieves context and generates reasoned answer.
    """
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        if request.multi_hop:
            trace = reasoning_engine.multi_hop_reasoning(request.question)
        else:
            trace = reasoning_engine.answer_question(request.question, request.max_context)
        
        return QuestionResponse(
            question=trace.query,
            answer=trace.answer,
            confidence=trace.confidence,
            reasoning_steps=trace.steps,
            sources=trace.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


# Knowledge Graph Endpoints
@app.post("/api/entity")
async def add_entity(request: EntityRequest):
    """Add an entity to the knowledge graph."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        entity_name = brain.knowledge_graph.add_entity(
            name=request.name,
            entity_type=request.entity_type,
            properties=request.properties
        )
        
        return {
            "status": "success",
            "entity": entity_name,
            "type": request.entity_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity creation failed: {str(e)}")


@app.post("/api/relationship")
async def add_relationship(request: RelationshipRequest):
    """Create a relationship between two entities."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        brain.knowledge_graph.add_relationship(
            entity1=request.entity1,
            relation=request.relation,
            entity2=request.entity2,
            properties=request.properties
        )
        
        return {
            "status": "success",
            "relationship": f"{request.entity1} -{request.relation}-> {request.entity2}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relationship creation failed: {str(e)}")


@app.get("/api/neighbors/{entity}")
async def get_neighbors(entity: str, depth: int = 1):
    """Get neighboring entities in the knowledge graph."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        neighbors = brain.knowledge_graph.get_neighbors(entity, depth=depth)
        
        return {
            "entity": entity,
            "depth": depth,
            "count": len(neighbors),
            "neighbors": neighbors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neighbor retrieval failed: {str(e)}")


@app.get("/api/paths/{entity1}/{entity2}")
async def find_paths(entity1: str, entity2: str, max_depth: int = 3):
    """Find paths between two entities."""
    if not brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        paths = brain.knowledge_graph.find_paths(entity1, entity2, max_depth)
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "max_depth": max_depth,
            "count": len(paths),
            "paths": paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path finding failed: {str(e)}")


# Batch Operations
@app.post("/api/batch/ingest")
async def batch_ingest(contents: List[str], memory_type: str = "long_term"):
    """Batch ingest multiple pieces of content."""
    if not brain or not inference_engine or not vector_db:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        memory_ids = []
        
        # Ingest all content
        for content in contents:
            memory_id = brain.ingest(content, memory_type=memory_type)
            memory_ids.append(memory_id)
        
        # Generate embeddings in batch
        embeddings = inference_engine.batch_embed(contents)
        
        # Store in vector DB
        vector_db.batch_insert(
            ids=memory_ids,
            embeddings=embeddings,
            contents=contents,
            metadatas=[{'memory_type': memory_type} for _ in contents]
        )
        
        return {
            "status": "success",
            "ingested_count": len(memory_ids),
            "memory_ids": memory_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """API root with basic info."""
    return {
        "name": "Second Brain API",
        "version": "1.0.0",
        "description": "Production-grade second brain with NVIDIA stack",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "stats_url": "/api/stats"
    }


def run_server(host: str = "0.0.0.0", port: int = 8888):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
