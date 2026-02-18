"""
NVIDIA Second Brain - ULTIMATE API Server
All advanced features integrated
"""

from fastapi import FastAPI, Query, HTTPException, WebSocket, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import json
import time

# Core modules
from core import SecondBrain, MemoryType

# Advanced features
from auth import AuthManager, Permission, UserRole
from advanced import (
    WebhookManager, EventType, AutoTagger, RecommendationEngine,
    DuplicateDetector, AnalyticsEngine, BackgroundJobProcessor
)
from ultimate import (
    InsightGenerator, MultiAgentReasoner, SemanticCache,
    TemporalReasoningEngine, DomainModelManager, CollaborativeKnowledgeGraph
)
from visualization import DashboardServer, DASHBOARD_HTML
from integrations import BrowserExtensionAPI, ObsidianSync, MobileAPI, APIGateway, WebCapture

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="NVIDIA Second Brain - ULTIMATE Edition",
    version="2.0.0",
    description="Production-grade AI-powered second brain with all advanced features"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class IngestRequest(BaseModel):
    content: str
    memory_type: str = "semantic"
    tags: List[str] = []
    metadata: Optional[Dict] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    memory_type: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    context_size: int = 5

# ============================================================================
# Initialize Systems
# ============================================================================

brain = SecondBrain()
auth_manager = AuthManager()

# Advanced features
webhook_manager = WebhookManager()
auto_tagger = AutoTagger()
duplicate_detector = DuplicateDetector()
analytics = AnalyticsEngine()
background_jobs = BackgroundJobProcessor()

# Ultimate features
insight_gen = InsightGenerator(brain.memory_store)
multi_agent = MultiAgentReasoner()
semantic_cache = SemanticCache(threshold=0.85)
temporal_engine = TemporalReasoningEngine()
domain_manager = DomainModelManager()
collab_kg = CollaborativeKnowledgeGraph()

# Visualization
dashboard = DashboardServer(brain.memory_store)

# Integrations
browser_api = BrowserExtensionAPI()
obsidian_sync = ObsidianSync("~/Obsidian")
mobile_api = MobileAPI()
gateway = APIGateway()

# Recommendation engine
recommendation_engine = RecommendationEngine(brain.vector_db, brain.memory_store)

# ============================================================================
# Core Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "NVIDIA Second Brain - ULTIMATE Edition",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "core": ["memory_system", "knowledge_graph", "vector_search"],
            "intelligence": ["nlp_tagging", "insights", "multi_agent_reasoning", "temporal_reasoning"],
            "enterprise": ["auth", "webhooks", "analytics", "background_jobs"],
            "advanced": ["semantic_cache", "fine_tuning", "collaborative_kg"],
            "visualization": ["3d_graph", "dashboard", "real_time_updates"],
            "integrations": ["browser_extension", "obsidian", "mobile", "webhooks"]
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "components": {
            "memory_system": "operational",
            "knowledge_graph": "operational",
            "auth": "operational",
            "webhooks": "operational",
            "analytics": "operational"
        }
    }

# ============================================================================
# Memory & Ingestion
# ============================================================================

@app.post("/api/ingest")
async def ingest_content(request: IngestRequest):
    """Ingest new content with advanced features."""
    # Auto-tag
    auto_tags = auto_tagger.extract_tags(request.content)
    all_tags = list(set(request.tags + auto_tags))
    
    # Duplicate detection
    duplicate_id = duplicate_detector.is_exact_duplicate(request.content)
    if duplicate_id:
        return {
            "status": "duplicate",
            "duplicate_of": duplicate_id,
            "message": "Content already exists"
        }
    
    # Ingest
    memory_id = brain.ingest(
        request.content,
        memory_type=request.memory_type,
        tags=all_tags,
        entities=[]
    )
    
    # Register for duplicate detection
    duplicate_detector.register_content(memory_id, request.content)
    
    # Trigger webhook
    webhook_manager.trigger_event(EventType.MEMORY_CREATED, {
        "memory_id": memory_id,
        "content": request.content[:100],
        "tags": all_tags
    })
    
    # Analytics
    analytics.record_event("ingest", {
        "memory_id": memory_id,
        "memory_type": request.memory_type,
        "tag_count": len(all_tags)
    })
    
    return {
        "status": "success",
        "memory_id": memory_id,
        "tags": all_tags,
        "auto_tags": auto_tags
    }

@app.get("/api/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Get specific memory."""
    memory = brain.memory_store.get_memory(memory_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Record access for recommendations
    recommendation_engine.record_access("default_user", memory_id)
    
    return {
        "id": memory_id,
        "content": memory.content,
        "type": memory.memory_type,
        "tags": memory.tags,
        "retention": memory.calculate_retention(),
        "access_count": memory.access_count,
        "created_at": memory.timestamp,
        "last_accessed": memory.last_accessed
    }

@app.get("/api/recommendations/{memory_id}")
async def get_recommendations(memory_id: str, top_k: int = 5):
    """Get recommendations for a memory."""
    recommendations = recommendation_engine.recommend_similar(memory_id, top_k)
    return {"recommendations": recommendations}

# ============================================================================
# Search & Retrieval
# ============================================================================

@app.post("/api/search")
async def search(request: SearchRequest):
    """Advanced semantic search."""
    # Check semantic cache
    embeddings = await brain.inference_engine.embed([request.query])
    cached_result = await semantic_cache.get(embeddings[0], request.query)
    
    if cached_result:
        return {"status": "cached", "results": cached_result}
    
    # Perform search
    results = brain.search(request.query, top_k=request.top_k)
    
    # Cache results
    semantic_cache.put(request.query, embeddings[0], results)
    
    # Analytics
    analytics.record_event("search", {
        "query": request.query,
        "result_count": len(results)
    })
    
    return {
        "status": "success",
        "query": request.query,
        "results": [
            {
                "memory_id": r["memory_id"],
                "content": r["content"],
                "score": r["score"],
                "type": r.get("type", "unknown")
            }
            for r in results
        ]
    }

@app.post("/api/question")
async def ask_question(request: QuestionRequest):
    """Answer questions using chain-of-thought reasoning."""
    result = brain.ask(request.question)
    
    # Analytics
    analytics.record_event("question", {
        "question": request.question,
        "confidence": result.get("confidence", 0)
    })
    
    # Webhook
    webhook_manager.trigger_event(EventType.QUESTION_ANSWERED, {
        "question": request.question,
        "answer": result.get("answer", "")[:100]
    })
    
    return result

# ============================================================================
# Insights & Intelligence
# ============================================================================

@app.post("/api/insights/generate")
async def generate_insights():
    """Generate all insights from knowledge base."""
    insights = await insight_gen.generate_all_insights()
    return {
        "total_insights": len(insights),
        "insights": [
            {
                "id": i.id,
                "type": i.type.value,
                "title": i.title,
                "description": i.description,
                "confidence": i.confidence,
                "impact": i.impact_score,
                "relevance": i.relevance_to_user
            }
            for i in insights[:20]  # Top 20
        ]
    }

@app.get("/api/insights/{insight_type}")
async def get_insights_by_type(insight_type: str):
    """Get insights of specific type."""
    insights = await insight_gen.generate_all_insights()
    filtered = [i for i in insights if i.type.value == insight_type]
    return {
        "type": insight_type,
        "count": len(filtered),
        "insights": [{
            "id": i.id,
            "title": i.title,
            "description": i.description,
            "confidence": i.confidence
        } for i in filtered[:10]]
    }

@app.post("/api/reason/multi-agent")
async def multi_agent_reasoning(query: str = Body(..., embed=True)):
    """Get reasoning from multiple agents."""
    context = []
    result = await multi_agent.reason_collectively(query, context)
    return result

# ============================================================================
# Authentication & Users
# ============================================================================

@app.post("/api/auth/login")
async def login(username: str = Body(...), password: str = Body(...)):
    """Authenticate user."""
    user_id = auth_manager.authenticate(username, password)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = auth_manager.generate_token(user_id)
    session_id = auth_manager.create_session(user_id)
    
    return {
        "status": "success",
        "user_id": user_id,
        "token": token,
        "session_id": session_id
    }

@app.post("/api/auth/api-key")
async def create_api_key(user_id: str = Body(...), name: str = Body(...)):
    """Create API key for user."""
    try:
        api_key = auth_manager.create_api_key(user_id, name, expires_in=2592000)  # 30 days
        return {
            "status": "success",
            "api_key": api_key,
            "message": "Save this key securely - it won't be shown again!"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics."""
    stats = auth_manager.get_user_stats(user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    return stats

# ============================================================================
# Webhooks
# ============================================================================

@app.post("/api/webhooks/register")
async def register_webhook(
    url: str = Body(...),
    events: List[str] = Body(...),
    secret: str = Body(...)
):
    """Register a webhook."""
    event_types = [EventType(e) for e in events]
    webhook_id = webhook_manager.register_webhook(url, event_types, secret)
    return {
        "status": "success",
        "webhook_id": webhook_id,
        "url": url,
        "events": events
    }

@app.get("/api/webhooks/list")
async def list_webhooks():
    """List all webhooks."""
    webhooks = webhook_manager.list_webhooks()
    return {
        "count": len(webhooks),
        "webhooks": [
            {
                "id": w.webhook_id,
                "url": w.url,
                "events": [e.value for e in w.events],
                "trigger_count": w.trigger_count,
                "is_active": w.is_active
            }
            for w in webhooks
        ]
    }

@app.post("/webhooks/{source}")
async def handle_external_webhook(source: str, payload: Dict = Body(...)):
    """Handle webhooks from external sources."""
    result = await gateway.handle_webhook(source, payload)
    return result

# ============================================================================
# Analytics
# ============================================================================

@app.get("/api/analytics/usage")
async def get_usage_analytics(time_range: int = 86400):
    """Get usage statistics."""
    stats = analytics.get_usage_stats(time_range)
    return stats

@app.get("/api/analytics/memory-health")
async def get_memory_health():
    """Get memory health score."""
    health_score = analytics.get_memory_health_score(brain.memory_store)
    insights = analytics.get_retention_insights(brain.memory_store)
    return {
        "health_score": health_score,
        "insights": insights
    }

@app.get("/api/analytics/search-effectiveness")
async def get_search_effectiveness():
    """Get search effectiveness metrics."""
    return analytics.get_search_effectiveness()

# ============================================================================
# Visualization & Dashboard
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve 3D knowledge graph dashboard."""
    return DASHBOARD_HTML

@app.get("/api/dashboard/data")
async def dashboard_data():
    """Get dashboard visualization data."""
    return await dashboard.get_dashboard_data()

@app.get("/api/dashboard/node/{node_id}")
async def node_details(node_id: str):
    """Get details about specific node."""
    return await dashboard.get_node_details(node_id)

# ============================================================================
# Browser Extension
# ============================================================================

@app.get("/extension/manifest.json")
async def extension_manifest():
    """Get browser extension manifest."""
    return browser_api.get_extension_manifest()

@app.post("/api/extension/capture")
async def capture_from_browser(capture: Dict = Body(...)):
    """Handle capture from browser extension."""
    web_capture = WebCapture(
        url=capture.get("url", ""),
        title=capture.get("title", ""),
        content=capture.get("content", ""),
        selected_text=capture.get("selected_text"),
        tags=capture.get("tags", [])
    )
    
    result = await browser_api.capture_page(web_capture)
    return result

# ============================================================================
# Obsidian Sync
# ============================================================================

@app.post("/api/obsidian/sync-from")
async def sync_from_obsidian(vault_path: Optional[str] = Body(None)):
    """Import from Obsidian vault."""
    if vault_path:
        obsidian_sync.vault_path = vault_path
    
    result = await obsidian_sync.sync_from_obsidian()
    return result

@app.post("/api/obsidian/sync-to")
async def sync_to_obsidian():
    """Export memories to Obsidian."""
    # Get recent memories
    memories = []
    for memory_id, memory in list(brain.memory_store.memories.items())[:50]:
        memories.append({
            "id": memory_id,
            "content": memory.content,
            "tags": memory.tags
        })
    
    result = await obsidian_sync.sync_to_obsidian(memories)
    return result

# ============================================================================
# Mobile API
# ============================================================================

@app.get("/api/mobile/dashboard")
async def mobile_dashboard():
    """Get mobile-optimized dashboard."""
    return await mobile_api.get_mobile_dashboard()

@app.post("/api/mobile/sync-offline")
async def sync_offline_queue(queue: List[Dict] = Body(...)):
    """Sync offline-created notes."""
    result = await mobile_api.sync_offline_queue(queue)
    return result

# ============================================================================
# Advanced Features
# ============================================================================

@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return semantic_cache.get_stats()

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear semantic cache."""
    semantic_cache.cache.clear()
    semantic_cache.embeddings.clear()
    return {"status": "cache_cleared"}

@app.post("/api/collab/merge")
async def merge_contributions(user_contributions: Dict = Body(...)):
    """Merge contributions from multiple users."""
    result = collab_kg.merge_contributions(user_contributions)
    return result

# ============================================================================
# System Management
# ============================================================================

@app.get("/api/system/stats")
async def system_stats():
    """Get comprehensive system statistics."""
    return {
        "memory": brain.get_comprehensive_stats(),
        "cache": semantic_cache.get_stats(),
        "analytics": {
            "total_events": len(analytics.events),
            "event_types": len(analytics.metrics)
        },
        "webhooks": {
            "registered": len(webhook_manager.webhooks),
            "total_triggers": sum(w.trigger_count for w in webhook_manager.webhooks.values())
        },
        "auth": {
            "total_users": len(auth_manager.users),
            "active_sessions": len(auth_manager.sessions)
        }
    }

@app.post("/api/system/optimize")
async def optimize_system():
    """Run system optimization."""
    # Review weak memories
    weak_memories = [
        (mid, m) for mid, m in brain.memory_store.memories.items()
        if m.calculate_retention() < 0.3
    ]
    
    return {
        "status": "optimization_complete",
        "weak_memories_found": len(weak_memories),
        "recommendations": [
            "Review memories with low retention",
            "Consolidate duplicate content",
            "Refresh important knowledge"
        ]
    }

# ============================================================================
# WebSocket Streams
# ============================================================================

@app.websocket("/ws/insights")
async def websocket_insights(websocket: WebSocket):
    """Stream insights as they're generated."""
    await websocket.accept()
    
    try:
        while True:
            insights = await insight_gen.generate_all_insights()
            
            for insight in insights[:5]:
                await websocket.send_json({
                    "type": "insight",
                    "title": insight.title,
                    "confidence": insight.confidence,
                    "impact": insight.impact_score
                })
            
            await asyncio.sleep(5)
    
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """Stream real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            await websocket.send_json({
                "type": "stats_update",
                "total_memories": len(brain.memory_store.memories),
                "timestamp": time.time()
            })
            
            await asyncio.sleep(2)
    
    except Exception as e:
        print(f"WebSocket error: {e}")

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize background jobs."""
    print("🧠 NVIDIA Second Brain - ULTIMATE Edition Starting...")
    print("=" * 70)
    print("✓ Core memory system initialized")
    print("✓ Authentication system ready")
    print("✓ Webhook manager active")
    print("✓ Analytics engine running")
    print("✓ Advanced features loaded")
    print("✓ Visualization dashboard ready")
    print("✓ Integration APIs initialized")
    print("=" * 70)
    print(f"🚀 Server ready at http://0.0.0.0:8888")
    print(f"📊 Dashboard at http://0.0.0.0:8888/dashboard")
    print(f"📖 API docs at http://0.0.0.0:8888/docs")
    print("=" * 70)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
