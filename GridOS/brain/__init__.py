"""
NVIDIA Second Brain - ULTIMATE Edition
Complete AI-powered knowledge management system with all advanced features

Production-grade second brain with NVIDIA stack + Enterprise Features.

Components:
- core: Multi-tier memory system + knowledge graph
- nvidia_inference: NVIDIA NIM + TensorRT integration
- vector_db: Milvus GPU-accelerated vector database
- reasoning: 3-stage retrieval + chain-of-thought reasoning
- auth: JWT authentication + RBAC + API keys
- advanced: Webhooks, auto-tagging, analytics, recommendations
- ultimate: Insights, multi-agent reasoning, temporal reasoning, semantic cache
- visualization: 3D knowledge graph dashboard
- integrations: Browser extension, Obsidian sync, mobile API
- server_ultimate: REST API with 40+ endpoints

Usage:
    from brain import SecondBrain
    
    brain = SecondBrain()
    brain.ingest("Your knowledge here")
    results = brain.retrieve("search query")
"""

__version__ = "2.0.0"
__edition__ = "ULTIMATE"
__author__ = "NVIDIA Second Brain Team"

from .core import SecondBrain, MemoryStore, KnowledgeGraph, MemoryNode
from .nvidia_inference import NVIDIAInferenceEngine, InferenceConfig
from .vector_db import MilvusVectorDB, VectorConfig
from .reasoning import RetrievalPipeline, ReasoningEngine, RetrievalResult

# Ultimate Edition Features
try:
    from .auth import AuthManager, UserRole, Permission
    from .advanced import WebhookManager, AutoTagger, AnalyticsEngine
    from .ultimate import InsightGenerator, MultiAgentReasoner, SemanticCache
    from .visualization import KnowledgeGraphVisualizer, DashboardServer
    from .integrations import BrowserExtensionAPI, ObsidianSync, MobileAPI
    
    ULTIMATE_FEATURES_AVAILABLE = True
except ImportError:
    ULTIMATE_FEATURES_AVAILABLE = False

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

# Add ultimate features if available
if ULTIMATE_FEATURES_AVAILABLE:
    __all__.extend([
        "AuthManager",
        "UserRole",
        "Permission",
        "WebhookManager",
        "AutoTagger",
        "AnalyticsEngine",
        "InsightGenerator",
        "MultiAgentReasoner",
        "SemanticCache",
        "KnowledgeGraphVisualizer",
        "DashboardServer",
        "BrowserExtensionAPI",
        "ObsidianSync",
        "MobileAPI",
    ])


def get_version():
    """Get version and feature info."""
    return {
        'version': __version__,
        'edition': __edition__,
        'ultimate_features': ULTIMATE_FEATURES_AVAILABLE,
        'features': {
            'core': True,
            'authentication': ULTIMATE_FEATURES_AVAILABLE,
            'webhooks': ULTIMATE_FEATURES_AVAILABLE,
            'analytics': ULTIMATE_FEATURES_AVAILABLE,
            'insights': ULTIMATE_FEATURES_AVAILABLE,
            'multi_agent': ULTIMATE_FEATURES_AVAILABLE,
            'temporal_reasoning': ULTIMATE_FEATURES_AVAILABLE,
            'semantic_cache': ULTIMATE_FEATURES_AVAILABLE,
            'fine_tuning': ULTIMATE_FEATURES_AVAILABLE,
            'collaboration': ULTIMATE_FEATURES_AVAILABLE,
            'visualization': ULTIMATE_FEATURES_AVAILABLE,
            'browser_extension': ULTIMATE_FEATURES_AVAILABLE,
            'obsidian_sync': ULTIMATE_FEATURES_AVAILABLE,
            'mobile_api': ULTIMATE_FEATURES_AVAILABLE
        }
    }


def print_banner():
    """Print startup banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           🧠 NVIDIA SECOND BRAIN - {__edition__:^10} 🚀              ║
║                                                                  ║
║        AI-Powered Knowledge Management System                    ║
║                   Version {__version__}                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Core Features:
✓ Multi-tier Memory System (Ebbinghaus, Spaced Repetition)
✓ NVIDIA NIM + TensorRT Inference
✓ GPU-Accelerated Vector Search (Milvus)
✓ Knowledge Graph (Neo4j)
✓ Chain-of-Thought Reasoning
"""
    
    if ULTIMATE_FEATURES_AVAILABLE:
        banner += """
Advanced Features:
✓ JWT Authentication & RBAC
✓ Webhook Notifications
✓ Auto-Tagging & NLP
✓ Insight Generation
✓ Multi-Agent Reasoning
✓ Temporal Reasoning
✓ Semantic Caching
✓ Domain Fine-Tuning
✓ Collaborative Knowledge Graph
✓ 3D Visualization Dashboard
✓ Browser Extension Integration
✓ Obsidian Vault Sync
✓ Mobile API
✓ Analytics & Monitoring
"""
    
    banner += "\nReady for Production! 🎉\n"
    print(banner)


if __name__ == '__main__':
    print_banner()
    version_info = get_version()
    print(f"Version: {version_info['version']}")
    print(f"Edition: {version_info['edition']}")
    print(f"Ultimate Features: {'Enabled' if version_info['ultimate_features'] else 'Disabled'}")
    print(f"\nActive Features: {sum(version_info['features'].values())}/{len(version_info['features'])}")

