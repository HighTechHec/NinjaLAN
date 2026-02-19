"""
NVIDIA Second Brain - HYBRID Edition
Complete AI-powered knowledge management system with NVIDIA + Google Cloud

Production-grade second brain leveraging:
- NVIDIA: GPU acceleration, local inference, real-time processing
- Google Cloud: Cloud storage, Gemini AI, Keep/Drive sync

Components:
- core: Multi-tier memory system + knowledge graph
- nvidia_inference: NVIDIA NIM + TensorRT integration
- nvidia_nemo: Advanced NLP with NeMo
- nvidia_rapids: GPU-accelerated data processing
- vector_db: Milvus GPU-accelerated vector database
- reasoning: 3-stage retrieval + chain-of-thought reasoning
- google_cloud: Keep, Drive, Gemini, Firebase integration
- hybrid_orchestrator: Intelligent routing between NVIDIA & Google
- auth: JWT authentication + RBAC + API keys
- advanced: Webhooks, auto-tagging, analytics, recommendations
- ultimate: Insights, multi-agent reasoning, temporal reasoning, semantic cache
- visualization: 3D knowledge graph dashboard
- integrations: Browser extension, Obsidian sync, mobile API
- server_ultimate: REST API with 40+ endpoints

Usage:
    from brain import HybridBrain
    
    brain = HybridBrain()
    brain.ingest("Your knowledge here")
    results = brain.retrieve("search query")
"""

__version__ = "3.0.0"
__edition__ = "HYBRID"
__author__ = "NVIDIA Second Brain Team"

from .core import SecondBrain, MemoryStore, KnowledgeGraph, MemoryNode
from .nvidia_inference import NVIDIAInferenceEngine, InferenceConfig
from .vector_db import MilvusVectorDB, VectorConfig
from .reasoning import RetrievalPipeline, ReasoningEngine, RetrievalResult

# NVIDIA Advanced Features
try:
    from .nvidia_nemo import NeMoNLPEngine, IntentType, SentimentType
    from .nvidia_rapids import RAPIDSDataProcessor, RAPIDSGraphProcessor
    
    NVIDIA_ADVANCED_AVAILABLE = True
except ImportError:
    NVIDIA_ADVANCED_AVAILABLE = False

# Google Cloud Features
try:
    from .google_cloud import GoogleCloudManager, CloudConfig, SyncStatus
    from .hybrid_orchestrator import HybridOrchestrator, HybridConfig, ProcessingMode
    
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

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

# Add NVIDIA advanced features if available
if NVIDIA_ADVANCED_AVAILABLE:
    __all__.extend([
        "NeMoNLPEngine",
        "IntentType",
        "SentimentType",
        "RAPIDSDataProcessor",
        "RAPIDSGraphProcessor",
    ])

# Add Google Cloud features if available
if GOOGLE_CLOUD_AVAILABLE:
    __all__.extend([
        "GoogleCloudManager",
        "CloudConfig",
        "SyncStatus",
        "HybridOrchestrator",
        "HybridConfig",
        "ProcessingMode",
    ])

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
        'nvidia_advanced': NVIDIA_ADVANCED_AVAILABLE,
        'google_cloud': GOOGLE_CLOUD_AVAILABLE,
        'ultimate_features': ULTIMATE_FEATURES_AVAILABLE,
        'features': {
            'core': True,
            # NVIDIA
            'nvidia_nim': True,
            'nvidia_nemo': NVIDIA_ADVANCED_AVAILABLE,
            'nvidia_rapids': NVIDIA_ADVANCED_AVAILABLE,
            'tensorrt': True,
            # Google Cloud
            'google_keep': GOOGLE_CLOUD_AVAILABLE,
            'google_drive': GOOGLE_CLOUD_AVAILABLE,
            'google_gemini': GOOGLE_CLOUD_AVAILABLE,
            'firebase': GOOGLE_CLOUD_AVAILABLE,
            'hybrid_orchestrator': GOOGLE_CLOUD_AVAILABLE,
            # Ultimate
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
║           🧠 SECOND BRAIN - {__edition__:^10} EDITION 🚀              ║
║                                                                  ║
║        NVIDIA GPU + Google Cloud AI Platform                     ║
║                   Version {__version__}                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

NVIDIA Stack:
✓ NVIDIA NIM + TensorRT Inference
✓ GPU-Accelerated Vector Search (Milvus)
✓ NVIDIA NeMo for Advanced NLP
✓ NVIDIA RAPIDS for Data Processing
✓ Multi-tier Memory System
✓ Knowledge Graph Analytics
"""
    
    if GOOGLE_CLOUD_AVAILABLE:
        banner += """
Google Cloud Integration:
✓ Google Keep - Rapid Capture
✓ Google Drive - Structured Storage
✓ Google Gemini (Vertex AI) - Advanced Reasoning
✓ Firebase/Firestore - Cloud Sync
✓ Hybrid Orchestrator - Intelligent Routing
"""
    
    if ULTIMATE_FEATURES_AVAILABLE:
        banner += """
Enterprise Features:
✓ JWT Authentication & RBAC
✓ Webhook Notifications
✓ Insight Generation
✓ Multi-Agent Reasoning
✓ Temporal Reasoning
✓ Semantic Caching
✓ 3D Visualization Dashboard
✓ Browser Extension
✓ Obsidian Sync
✓ Mobile API
✓ Analytics & Monitoring
"""
    
    banner += "\n🎉 Best of Both Worlds: NVIDIA + Google Cloud! 🎉\n"
    print(banner)


if __name__ == '__main__':
    print_banner()
    version_info = get_version()
    print(f"Version: {version_info['version']}")
    print(f"Edition: {version_info['edition']}")
    print(f"Ultimate Features: {'Enabled' if version_info['ultimate_features'] else 'Disabled'}")
    print(f"\nActive Features: {sum(version_info['features'].values())}/{len(version_info['features'])}")

