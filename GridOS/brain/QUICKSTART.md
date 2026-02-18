# Second Brain - Quick Start Guide

Get your Second Brain running in 5 minutes! ⚡

---

## 🎯 Goal

By the end of this guide, you'll have:
- ✅ A running Second Brain system
- ✅ Ingested your first knowledge
- ✅ Performed semantic search
- ✅ Asked questions and got answers

---

## 📋 Prerequisites

Choose your path:

### Path A: Docker (Recommended)
- Docker + Docker Compose installed
- 8GB RAM minimum
- (Optional) NVIDIA GPU for acceleration

### Path B: Local Development
- Python 3.11+
- 8GB RAM minimum

---

## 🚀 Method 1: Docker (Production)

### Step 1: Clone & Navigate

```bash
cd GridOS/brain
```

### Step 2: Start Services

```bash
docker compose up -d
```

Wait 2-3 minutes for all services to start.

### Step 3: Verify

```bash
# Check all services are running
docker compose ps

# Check API health
curl http://localhost:8888/api/health
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2024-...",
  "components": {
    "brain": true,
    "vector_db": true,
    "inference_engine": true,
    "retrieval_pipeline": true,
    "reasoning_engine": true
  }
}
```

### Step 4: Run Demo

```bash
docker exec -it second-brain-api python demo.py
```

### Step 5: Try CLI

```bash
docker exec -it second-brain-api python cli.py
```

---

## 🔧 Method 2: Local Development

### Step 1: Install Dependencies

```bash
cd GridOS/brain
pip install -r requirements.txt
```

### Step 2: Start External Services

You need Milvus and Neo4j running. Quick setup:

```bash
# Start just the databases
docker compose up -d milvus neo4j redis
```

Or install them locally (see their documentation).

### Step 3: Run Demo

```bash
python demo.py
```

### Step 4: Try CLI

```bash
python cli.py
```

### Step 5: Start API (Optional)

```bash
python server.py
```

Access API docs at: http://localhost:8888/docs

---

## 📖 Your First Commands

### Using the CLI

```bash
python cli.py
```

Then try these commands:

```
🧠 > ingest NVIDIA GPUs accelerate AI inference #nvidia #gpu
✓ Content ingested successfully

🧠 > ingest Vector databases enable semantic search #vectors #search
✓ Content ingested successfully

🧠 > search GPU acceleration
🔍 Search results for: 'GPU acceleration'
[1] Score: 0.8943 | Source: dense_retrieval
    NVIDIA GPUs accelerate AI inference

🧠 > ask What is the benefit of GPU acceleration?
💭 Thinking...
✓ Answer (confidence: 0.85):
   GPU acceleration significantly speeds up AI inference by...

🧠 > stats
📊 System Statistics
{
  "memory": {"total_memories": 2, "long_term": 0, ...},
  ...
}

🧠 > exit
```

### Using the API

```bash
# Ingest content
curl -X POST http://localhost:8888/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning uses statistical techniques",
    "memory_type": "semantic",
    "tags": ["ml", "ai"]
  }'

# Search
curl -X POST http://localhost:8888/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5
  }'

# Ask a question
curl -X POST http://localhost:8888/api/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does machine learning work?",
    "max_context": 5
  }'

# Get statistics
curl http://localhost:8888/api/stats
```

---

## 🎨 Using Python API

```python
from brain import SecondBrain

# Initialize
brain = SecondBrain()

# Ingest knowledge
memory_id = brain.ingest(
    "Python is a high-level programming language",
    memory_type='semantic',
    tags=['python', 'programming']
)

# Search
results = brain.retrieve("programming language", method='semantic')
for result in results:
    print(f"- {result.content}")

# Get review queue (spaced repetition)
review_queue = brain.get_review_queue(limit=10)
print(f"Items to review: {len(review_queue)}")

# Statistics
stats = brain.get_comprehensive_stats()
print(stats)

# Cleanup
brain.close()
```

---

## 🔍 Next Steps

### 1. Explore the Demo

```bash
python demo.py
```

This runs through all 6 layers:
1. Memory system
2. NVIDIA inference
3. Vector database
4. Knowledge graph
5. Retrieval pipeline
6. Reasoning engine

### 2. Read the Docs

- **README.md** - Full documentation
- **ARCHITECTURE.md** - System design
- **API Docs** - http://localhost:8888/docs

### 3. Build Your Use Case

Examples:
- **Personal Knowledge Base**: Ingest notes, articles, books
- **Research Assistant**: Ask questions about your research
- **Customer Support**: Build a knowledge base for support
- **Code Documentation**: Index and search codebases

### 4. Integrate

```python
# Your application
from brain import SecondBrain, NVIDIAInferenceEngine, MilvusVectorDB

brain = SecondBrain()
engine = NVIDIAInferenceEngine()

# Ingest documents
for doc in your_documents:
    brain.ingest(doc.content, tags=doc.tags)

# Search
query = "your search query"
results = brain.retrieve(query)

# Question answering
from brain import ReasoningEngine, RetrievalPipeline

pipeline = RetrievalPipeline(vector_db, engine, brain.knowledge_graph)
reasoning = ReasoningEngine(pipeline, engine)

answer = reasoning.answer_question("Your question?")
print(answer.answer)
```

---

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check Docker
docker --version
docker compose --version

# Check logs
docker compose logs -f

# Restart specific service
docker compose restart brain-api
```

### Connection Errors

```bash
# Test Milvus
curl http://localhost:19530/healthz

# Test Neo4j
curl http://localhost:7474

# Test API
curl http://localhost:8888/api/health
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Reduce batch size in config
# Edit InferenceConfig in nvidia_inference.py
max_batch_size = 16  # Instead of 64

# Use smaller context
# When asking questions
max_context = 3  # Instead of 5
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.11+
```

---

## 📚 Learning Resources

### Video Tutorials
- [System Overview](#) (Coming soon)
- [Building Your First Knowledge Base](#) (Coming soon)
- [Advanced Retrieval Techniques](#) (Coming soon)

### Blog Posts
- Introduction to Second Brain Architecture
- Optimizing Vector Search Performance
- Chain-of-Thought Reasoning Explained

### Example Projects
- `examples/personal_notes/` - Personal knowledge management
- `examples/research_assistant/` - Research Q&A system
- `examples/code_search/` - Code documentation search

---

## 💬 Getting Help

### Documentation
- **README.md** - Main documentation
- **ARCHITECTURE.md** - Technical design
- **API Reference** - http://localhost:8888/docs

### Community
- GitHub Issues - Bug reports and feature requests
- GitHub Discussions - Questions and ideas

### Common Commands

```bash
# View logs
docker compose logs -f brain-api

# Restart everything
docker compose restart

# Stop everything
docker compose down

# Remove all data (careful!)
docker compose down -v

# Update images
docker compose pull
```

---

## ✅ Quick Reference

### CLI Commands

```
ingest <content> [#tags] [--type=<type>]
search <query> [--top=<n>] [--no-rerank] [--no-expand]
ask <question> [--multi-hop] [--context=<n>]
review [limit]
stats
cleanup [threshold]
entity add <name> <type>
relation add <e1> <rel> <e2>
neighbors <entity> [depth]
exit
```

### API Endpoints

```
POST /api/ingest          - Ingest content
POST /api/search          - Semantic search
POST /api/question        - Question answering
GET  /api/memory/{id}     - Get memory
GET  /api/review          - Review queue
POST /api/cleanup         - Cleanup memories
POST /api/entity          - Add entity
POST /api/relationship    - Add relationship
GET  /api/neighbors/{e}   - Get neighbors
GET  /api/stats           - Statistics
GET  /api/health          - Health check
```

---

## 🎉 Congratulations!

You now have a production-grade Second Brain system running!

**What you can do:**
- ✅ Store and retrieve knowledge semantically
- ✅ Ask questions and get reasoned answers
- ✅ Build knowledge graphs automatically
- ✅ Implement spaced repetition
- ✅ Scale to 100K+ documents

**Next steps:**
1. Ingest your own data
2. Experiment with different queries
3. Build your custom application
4. Share your use case!

---

<div align="center">

**Happy Learning! 🧠**

*Need help? Check the [troubleshooting](#-troubleshooting) section or open an issue.*

</div>
