"""
Second Brain - Comprehensive Integration Demo

Demonstrates all 6 layers of the system:
1. Memory System (long-term, short-term, episodic, semantic)
2. NVIDIA Inference (embeddings + generation)
3. Vector Database (semantic search)
4. Knowledge Graph (entity linking)
5. Retrieval Pipeline (3-stage)
6. Reasoning Engine (Q&A)

Performance benchmarks included.
"""

import time
import json
from datetime import datetime


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


def benchmark(func, *args, **kwargs):
    """Benchmark a function execution."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def main():
    """Run comprehensive integration demo."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                  SECOND BRAIN - Integration Demo                      ║
║              Production-grade NVIDIA-powered system                    ║
╚════════════════════════════════════════════════════════════════════════╝
""")
    
    # Import components
    from core import SecondBrain, MemoryStore, KnowledgeGraph
    from nvidia_inference import NVIDIAInferenceEngine, TensorRTOptimizer
    from vector_db import MilvusVectorDB
    from reasoning import RetrievalPipeline, ReasoningEngine
    
    # Track performance
    perf_metrics = {}
    
    # ========================================================================
    # LAYER 1: Memory System
    # ========================================================================
    print_section("LAYER 1: Multi-Tier Memory System")
    
    brain = SecondBrain()
    
    # Add various types of memories
    print_subsection("Adding memories to different tiers")
    
    memories_data = [
        ("NVIDIA provides GPU-accelerated computing for AI and deep learning applications.",
         "semantic", ["nvidia", "gpu", "ai"]),
        ("Today I learned about vector databases and their role in semantic search.",
         "episodic", ["learning", "vectors"]),
        ("Remember to review spaced repetition algorithm tomorrow.",
         "short_term", ["todo", "review"]),
        ("Python is a high-level programming language known for its readability.",
         "long_term", ["python", "programming"]),
        ("TensorRT optimizes neural networks for inference on NVIDIA GPUs.",
         "semantic", ["tensorrt", "optimization"]),
    ]
    
    memory_ids = []
    for content, mem_type, tags in memories_data:
        mid = brain.ingest(content, memory_type=mem_type, tags=tags)
        memory_ids.append(mid)
        print(f"✓ Added {mem_type} memory: {content[:50]}...")
    
    # Test memory retrieval and retention
    print_subsection("Testing memory retrieval and retention")
    
    for mid in memory_ids[:2]:
        memory = brain.memory_store.get_memory(mid)
        retention = memory.calculate_retention()
        should_review, days = memory.should_review()
        print(f"Memory: {memory.content[:40]}...")
        print(f"  Type: {memory.memory_type} | Retention: {retention:.2f} | Strength: {memory.strength:.2f}")
        print(f"  Review: {'Yes' if should_review else 'No'} (next in {days} days)")
    
    # Test spaced repetition
    print_subsection("Spaced repetition review queue")
    review_queue = brain.get_review_queue(limit=3)
    print(f"Found {len(review_queue)} memories for review")
    
    # Memory statistics
    stats = brain.memory_store.get_stats()
    print_subsection("Memory statistics")
    print(json.dumps(stats, indent=2))
    
    perf_metrics['memory_operations'] = len(memory_ids)
    
    # ========================================================================
    # LAYER 2: NVIDIA Inference Engine
    # ========================================================================
    print_section("LAYER 2: NVIDIA Inference Engine")
    
    inference_engine = NVIDIAInferenceEngine()
    
    # Single embedding
    print_subsection("Single text embedding")
    text = "GPU acceleration dramatically speeds up AI inference"
    embedding, elapsed = benchmark(inference_engine.embed, text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Time: {elapsed*1000:.2f}ms")
    perf_metrics['single_embedding_ms'] = round(elapsed * 1000, 2)
    
    # Batch embeddings
    print_subsection("Batch embeddings (64 items)")
    batch_texts = [f"Sample text number {i} for batch processing" for i in range(64)]
    batch_embeddings, elapsed = benchmark(inference_engine.batch_embed, batch_texts)
    print(f"Batch size: {len(batch_texts)}")
    print(f"Embeddings shape: {batch_embeddings.shape}")
    print(f"Total time: {elapsed*1000:.2f}ms")
    print(f"Per-item: {elapsed*1000/len(batch_texts):.2f}ms")
    perf_metrics['batch_embedding_ms'] = round(elapsed * 1000, 2)
    perf_metrics['per_item_ms'] = round(elapsed * 1000 / len(batch_texts), 2)
    
    # Text generation
    print_subsection("Text generation")
    prompt = "Explain the benefits of GPU-accelerated AI inference in one sentence:"
    response, elapsed = benchmark(inference_engine.generate, prompt, max_tokens=100)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:150]}...")
    print(f"Time: {elapsed*1000:.2f}ms")
    perf_metrics['generation_ms'] = round(elapsed * 1000, 2)
    
    # Inference statistics
    print_subsection("Inference engine statistics")
    print(json.dumps(inference_engine.get_stats(), indent=2))
    
    # TensorRT optimization demo
    print_subsection("TensorRT INT8 quantization")
    optimizer = TensorRTOptimizer()
    result = optimizer.quantize_int8("model.onnx", "model_int8.trt")
    print(json.dumps(result, indent=2))
    
    # ========================================================================
    # LAYER 3: Vector Database
    # ========================================================================
    print_section("LAYER 3: Milvus Vector Database")
    
    vector_db = MilvusVectorDB()
    
    # Insert vectors
    print_subsection("Inserting vectors")
    docs = [
        "NVIDIA GPUs accelerate machine learning training and inference",
        "Vector databases enable fast similarity search for semantic retrieval",
        "Knowledge graphs represent entities and their relationships",
        "Chain-of-thought reasoning improves language model performance",
        "Spaced repetition optimizes long-term memory retention"
    ]
    
    embeddings = inference_engine.batch_embed(docs)
    doc_ids = [f"doc{i+1}" for i in range(len(docs))]
    
    success = vector_db.batch_insert(
        ids=doc_ids,
        embeddings=embeddings,
        contents=docs,
        metadatas=[{'source': 'demo'} for _ in docs]
    )
    print(f"✓ Inserted {len(docs)} vectors")
    print(f"Total vectors in DB: {vector_db.count()}")
    
    # Search
    print_subsection("Semantic search")
    query = "How can I speed up AI model inference?"
    query_emb = inference_engine.embed(query)
    
    results, elapsed = benchmark(vector_db.search, query_emb, top_k=3)
    print(f"Query: {query}")
    print(f"Search time: {elapsed*1000:.2f}ms")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] Score: {result['score']:.4f}")
        print(f"      {result['content']}")
    
    perf_metrics['vector_search_ms'] = round(elapsed * 1000, 2)
    
    # Hybrid search
    print_subsection("Hybrid search (vector + keyword)")
    results = vector_db.hybrid_search(query_emb, keyword="GPU", top_k=3)
    print("Results with keyword 'GPU' boost:")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] Score: {result['score']:.4f} - {result['content'][:60]}...")
    
    # Database statistics
    print_subsection("Vector database statistics")
    print(json.dumps(vector_db.get_stats(), indent=2))
    
    # ========================================================================
    # LAYER 4: Knowledge Graph
    # ========================================================================
    print_section("LAYER 4: Knowledge Graph (Neo4j)")
    
    kg = brain.knowledge_graph
    
    # Add entities
    print_subsection("Adding entities")
    entities = [
        ("NVIDIA", "company"),
        ("GPU", "hardware"),
        ("TensorRT", "software"),
        ("AI", "concept"),
        ("Inference", "process")
    ]
    
    for name, entity_type in entities:
        kg.add_entity(name, entity_type)
        print(f"✓ Added entity: {name} ({entity_type})")
    
    # Add relationships
    print_subsection("Adding relationships")
    relationships = [
        ("NVIDIA", "develops", "GPU"),
        ("NVIDIA", "develops", "TensorRT"),
        ("TensorRT", "accelerates", "Inference"),
        ("GPU", "enables", "AI"),
        ("AI", "requires", "Inference")
    ]
    
    for e1, rel, e2 in relationships:
        kg.add_relationship(e1, rel, e2)
        print(f"✓ {e1} -{rel}-> {e2}")
    
    # Find neighbors
    print_subsection("Finding neighbors")
    entity = "NVIDIA"
    neighbors = kg.get_neighbors(entity, depth=2)
    print(f"Neighbors of '{entity}' (depth=2):")
    for neighbor in neighbors[:5]:
        print(f"  • {neighbor.get('name')} ({neighbor.get('type')})")
    
    # Find paths
    print_subsection("Finding paths between entities")
    paths = kg.find_paths("NVIDIA", "AI", max_depth=3)
    if paths:
        print(f"Paths from NVIDIA to AI:")
        for i, path in enumerate(paths[:2], 1):
            print(f"  [{i}] {' -> '.join(path)}")
    
    # Knowledge graph statistics
    print_subsection("Knowledge graph statistics")
    print(json.dumps(kg.get_stats(), indent=2))
    
    # ========================================================================
    # LAYER 5: Retrieval Pipeline
    # ========================================================================
    print_section("LAYER 5: 3-Stage Retrieval Pipeline")
    
    pipeline = RetrievalPipeline(vector_db, inference_engine, kg)
    
    # Test full pipeline
    print_subsection("Full 3-stage retrieval")
    query = "What technologies are used for AI acceleration?"
    
    results, elapsed = benchmark(
        pipeline.retrieve,
        query,
        top_k=5,
        use_reranking=True,
        use_expansion=True
    )
    
    print(f"Query: {query}")
    print(f"Total retrieval time: {elapsed*1000:.2f}ms")
    print(f"\nRetrieved {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f} | Source: {result.source}")
        print(f"    {result.content[:80]}...")
    
    perf_metrics['full_retrieval_ms'] = round(elapsed * 1000, 2)
    
    # Compare different retrieval strategies
    print_subsection("Comparing retrieval strategies")
    
    strategies = [
        ("Dense only", {"use_reranking": False, "use_expansion": False}),
        ("Dense + Rerank", {"use_reranking": True, "use_expansion": False}),
        ("Full pipeline", {"use_reranking": True, "use_expansion": True})
    ]
    
    for name, kwargs in strategies:
        results, elapsed = benchmark(pipeline.retrieve, query, top_k=5, **kwargs)
        print(f"{name:20} - {elapsed*1000:6.2f}ms - {len(results)} results")
    
    # Pipeline statistics
    print_subsection("Retrieval pipeline statistics")
    print(json.dumps(pipeline.get_stats(), indent=2))
    
    # ========================================================================
    # LAYER 6: Reasoning Engine
    # ========================================================================
    print_section("LAYER 6: Chain-of-Thought Reasoning")
    
    reasoning_engine = ReasoningEngine(pipeline, inference_engine)
    
    # Simple question answering
    print_subsection("Question answering with reasoning")
    question = "How does NVIDIA technology help with AI?"
    
    trace, elapsed = benchmark(reasoning_engine.answer_question, question, max_context=3)
    
    print(f"Question: {trace.query}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"\nReasoning steps:")
    for step in trace.steps:
        print(f"  • {step}")
    
    print(f"\nAnswer (confidence: {trace.confidence:.2f}):")
    print(f"  {trace.answer[:200]}...")
    
    if trace.sources:
        print(f"\nSources: {', '.join(trace.sources[:3])}")
    
    perf_metrics['qa_ms'] = round(elapsed * 1000, 2)
    
    # Multi-hop reasoning
    print_subsection("Multi-hop reasoning")
    complex_question = "What is the relationship between GPU hardware and AI inference?"
    
    trace, elapsed = benchmark(reasoning_engine.multi_hop_reasoning, complex_question, max_hops=2)
    
    print(f"Question: {trace.query}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"\nReasoning trace ({len(trace.steps)} steps):")
    for step in trace.steps:
        print(f"  • {step}")
    
    print(f"\nFinal answer (confidence: {trace.confidence:.2f}):")
    print(f"  {trace.answer[:200]}...")
    
    perf_metrics['multi_hop_ms'] = round(elapsed * 1000, 2)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("PERFORMANCE SUMMARY")
    
    summary = {
        "System Components": {
            "Memory Tiers": 4,
            "Total Memories": brain.memory_store.get_stats()['total_memories'],
            "Vector Database": vector_db.count(),
            "Knowledge Graph Entities": kg.get_stats()['entities'],
            "Knowledge Graph Relationships": kg.get_stats()['relationships']
        },
        "Performance Metrics (ms)": perf_metrics,
        "Capabilities": {
            "Multi-tier memory": True,
            "Spaced repetition": True,
            "GPU acceleration": True,
            "Semantic search": True,
            "Knowledge graph": True,
            "3-stage retrieval": True,
            "Chain-of-thought": True,
            "Multi-hop reasoning": True
        }
    }
    
    print(json.dumps(summary, indent=2))
    
    print_section("DEMO COMPLETE")
    print("""
✓ All 6 layers tested successfully!

The Second Brain system is ready for:
- Personal knowledge management
- Research assistance
- Question answering
- Document analysis
- Semantic search
- Knowledge discovery

Next steps:
1. Start the API server: python server.py
2. Try the CLI: python cli.py
3. Run with Docker: docker compose up -d
""")
    
    # Cleanup
    brain.close()
    vector_db.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
