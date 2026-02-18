"""
Comprehensive Test Suite for Ultimate Second Brain
Tests all advanced features
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

# Import modules
from auth import AuthManager, UserRole, Permission
from advanced import (
    WebhookManager, EventType, AutoTagger, RecommendationEngine,
    DuplicateDetector, AnalyticsEngine
)
from ultimate import (
    InsightGenerator, MultiAgentReasoner, SemanticCache,
    TemporalReasoningEngine, TemporalEvent, TemporalRelation,
    DomainModelManager, CollaborativeKnowledgeGraph
)
from core import SecondBrain
from visualization import KnowledgeGraphVisualizer
from integrations import BrowserExtensionAPI, WebCapture


class TestAuthentication:
    """Test authentication system."""
    
    def test_create_user(self):
        """Test user creation."""
        auth = AuthManager()
        user_id = auth.create_user("testuser", "test@example.com", "password123")
        
        assert user_id is not None
        assert "testuser" in auth.username_index
        
    def test_authenticate(self):
        """Test authentication."""
        auth = AuthManager()
        user_id = auth.create_user("testuser", "test@example.com", "password123")
        
        auth_id = auth.authenticate("testuser", "password123")
        assert auth_id == user_id
        
        # Wrong password
        auth_id = auth.authenticate("testuser", "wrongpassword")
        assert auth_id is None
    
    def test_api_key(self):
        """Test API key creation and verification."""
        auth = AuthManager()
        user_id = auth.create_user("testuser", "test@example.com", "password123")
        
        # Create API key
        api_key = auth.create_api_key(user_id, "Test Key", expires_in=3600)
        assert api_key is not None
        
        # Verify API key
        verified = auth.verify_api_key(api_key)
        assert verified is not None
        assert verified.user_id == user_id
    
    def test_permissions(self):
        """Test permission checking."""
        auth = AuthManager()
        
        # Admin user
        admin_id = auth.create_user("admin", "admin@example.com", "pass", UserRole.ADMIN)
        assert auth.check_permission(admin_id, Permission.WRITE)
        assert auth.check_permission(admin_id, Permission.MANAGE_USERS)
        
        # Readonly user
        readonly_id = auth.create_user("readonly", "ro@example.com", "pass", UserRole.READONLY)
        assert auth.check_permission(readonly_id, Permission.READ)
        assert not auth.check_permission(readonly_id, Permission.WRITE)


class TestAdvancedFeatures:
    """Test advanced features."""
    
    def test_auto_tagging(self):
        """Test automatic tagging."""
        tagger = AutoTagger()
        
        text = "Machine learning with Python using neural networks for AI applications"
        tags = tagger.extract_tags(text)
        
        assert len(tags) > 0
        assert any('python' in tag.lower() or 'machine' in tag.lower() for tag in tags)
        
        category = tagger.suggest_category(text)
        assert category == 'technology'
    
    def test_duplicate_detection(self):
        """Test duplicate detection."""
        detector = DuplicateDetector(threshold=0.9)
        
        # Register content
        detector.register_content("mem1", "This is a test document")
        
        # Check exact duplicate
        is_dup = detector.is_exact_duplicate("This is a test document")
        assert is_dup == "mem1"
        
        # Check non-duplicate
        is_dup = detector.is_exact_duplicate("This is completely different content")
        assert is_dup is None
    
    def test_webhook_manager(self):
        """Test webhook system."""
        manager = WebhookManager()
        
        # Register webhook
        webhook_id = manager.register_webhook(
            url="http://example.com/webhook",
            events=[EventType.MEMORY_CREATED],
            secret="test_secret"
        )
        
        assert webhook_id is not None
        assert len(manager.webhooks) == 1
        
        # Trigger event
        manager.trigger_event(EventType.MEMORY_CREATED, {"test": "data"})
        assert len(manager.event_log) == 1
    
    def test_analytics_engine(self):
        """Test analytics."""
        analytics = AnalyticsEngine()
        
        # Record events
        analytics.record_event("search", {"query": "test", "result_count": 5})
        analytics.record_event("search", {"query": "test2", "result_count": 3})
        
        # Get stats
        stats = analytics.get_usage_stats(time_range=86400)
        assert stats["total_events"] == 2
        assert "search" in stats["event_breakdown"]


class TestUltimateFeatures:
    """Test ultimate edition features."""
    
    @pytest.mark.asyncio
    async def test_insight_generation(self):
        """Test insight generation."""
        brain = SecondBrain()
        
        # Add some memories
        brain.ingest("AI and machine learning are transforming industries", memory_type="semantic", tags=["ai", "ml"])
        brain.ingest("Machine learning algorithms improve with data", memory_type="semantic", tags=["ml", "data"])
        brain.ingest("Neural networks are used in deep learning", memory_type="semantic", tags=["ai", "neural"])
        
        insight_gen = InsightGenerator(brain.memory_store)
        
        # Generate insights
        insights = await insight_gen.generate_all_insights()
        
        assert len(insights) > 0
        # Should find patterns in ML/AI tags
        pattern_insights = [i for i in insights if i.type.value == "pattern"]
        assert len(pattern_insights) > 0
    
    @pytest.mark.asyncio
    async def test_multi_agent_reasoning(self):
        """Test multi-agent reasoning."""
        reasoner = MultiAgentReasoner()
        
        result = await reasoner.reason_collectively(
            "How can we improve data analysis?",
            context=[]
        )
        
        assert "query" in result
        assert "agent_responses" in result
        assert len(result["agent_responses"]) == 5  # All 5 agents
    
    def test_temporal_reasoning(self):
        """Test temporal reasoning engine."""
        engine = TemporalReasoningEngine()
        
        # Create events
        event1 = TemporalEvent(
            id="e1",
            name="Project Start",
            description="Started project",
            start_time=datetime(2024, 1, 1)
        )
        
        event2 = TemporalEvent(
            id="e2",
            name="Phase 1 Complete",
            description="Completed phase 1",
            start_time=datetime(2024, 2, 1)
        )
        
        event1.related_events[TemporalRelation.CAUSES] = ["e2"]
        
        engine.add_event(event1)
        engine.add_event(event2)
        
        # Find causal chain
        chain = engine.find_causal_chain("e1")
        assert len(chain) == 2
        assert chain[0].id == "e1"
        assert chain[1].id == "e2"
    
    @pytest.mark.asyncio
    async def test_semantic_cache(self):
        """Test semantic caching."""
        cache = SemanticCache(threshold=0.85)
        
        # Store in cache
        embedding = [0.1] * 384
        cache.put("test query", embedding, {"result": "data"})
        
        # Retrieve from cache
        result = await cache.get(embedding, "test query")
        assert result == {"result": "data"}
        
        # Stats
        stats = cache.get_stats()
        assert stats["cached_queries"] == 1
    
    @pytest.mark.asyncio
    async def test_domain_model_manager(self):
        """Test domain-specific fine-tuning."""
        manager = DomainModelManager()
        
        training_data = [{"text": "sample", "label": "positive"}] * 100
        
        model_name = await manager.fine_tune("medical", training_data)
        
        assert model_name is not None
        assert "medical" in manager.models
        assert manager.models["medical"]["data_size"] == 100
    
    def test_collaborative_kg(self):
        """Test collaborative knowledge graph."""
        collab_kg = CollaborativeKnowledgeGraph()
        
        # Add nodes
        collab_kg.add_node("user1", {"id": "node1", "content": "Test1"})
        collab_kg.add_node("user2", {"id": "node2", "content": "Test2"})
        
        assert len(collab_kg.nodes) == 2
        assert len(collab_kg.history) == 2
        
        # Merge contributions
        contributions = {
            "user3": [{"id": "node3", "content": "Test3", "version": 1}]
        }
        
        result = collab_kg.merge_contributions(contributions)
        assert result["merged"] == 1
        assert len(collab_kg.nodes) == 3


class TestVisualization:
    """Test visualization components."""
    
    def test_knowledge_graph_visualizer(self):
        """Test 3D visualization generation."""
        brain = SecondBrain()
        
        # Add some memories
        brain.ingest("Test memory 1", memory_type="semantic", tags=["test"])
        brain.ingest("Test memory 2", memory_type="episodic", tags=["test"])
        
        visualizer = KnowledgeGraphVisualizer(brain.memory_store)
        
        viz_data = visualizer.generate_visualization()
        
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert "metadata" in viz_data
        assert len(viz_data["nodes"]) == 2


class TestIntegrations:
    """Test integration modules."""
    
    @pytest.mark.asyncio
    async def test_browser_extension_api(self):
        """Test browser extension API."""
        api = BrowserExtensionAPI()
        
        # Test manifest
        manifest = api.get_extension_manifest()
        assert manifest["name"] == "NVIDIA Second Brain"
        assert manifest["version"] == "1.0.0"
        
        # Note: Actual capture test requires running server
    
    def test_web_capture(self):
        """Test web capture data structure."""
        capture = WebCapture(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            tags=["test"]
        )
        
        assert capture.url == "https://example.com"
        assert capture.title == "Test Page"
        assert "test" in capture.tags


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # Initialize brain
        brain = SecondBrain()
        
        # Ingest content
        memory_id = brain.ingest(
            "NVIDIA GPUs accelerate machine learning",
            memory_type="semantic",
            tags=["nvidia", "gpu", "ml"]
        )
        
        assert memory_id is not None
        
        # Search
        results = brain.search("GPU machine learning")
        assert len(results) > 0
        
        # Ask question
        answer = brain.ask("What accelerates machine learning?")
        assert "answer" in answer
        
        # Get stats
        stats = brain.get_comprehensive_stats()
        assert stats["memory"]["total_memories"] == 1
    
    @pytest.mark.asyncio
    async def test_advanced_workflow(self):
        """Test advanced features workflow."""
        brain = SecondBrain()
        auth = AuthManager()
        
        # Create user
        user_id = auth.create_user("testuser", "test@example.com", "pass123")
        assert user_id is not None
        
        # Generate API key
        api_key = auth.create_api_key(user_id, "Test Key")
        assert api_key is not None
        
        # Verify key
        key_obj = auth.verify_api_key(api_key)
        assert key_obj is not None
        
        # Ingest with advanced features
        auto_tagger = AutoTagger()
        content = "Python machine learning with neural networks"
        tags = auto_tagger.extract_tags(content)
        
        memory_id = brain.ingest(content, memory_type="semantic", tags=tags)
        
        # Generate insights
        insight_gen = InsightGenerator(brain.memory_store)
        insights = await insight_gen.generate_all_insights()
        
        # Should have some insights
        assert isinstance(insights, list)


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("🧪 Running ULTIMATE Second Brain Test Suite")
    print("=" * 70)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == '__main__':
    run_tests()
