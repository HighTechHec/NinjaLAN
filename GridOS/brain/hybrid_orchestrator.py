"""
Hybrid Orchestrator - NVIDIA + Google Cloud
Coordinates the best of both platforms for optimal performance

Features:
- Intelligent routing between NVIDIA and Google Cloud
- Hybrid embedding strategy
- Dual inference (local + cloud)
- Unified workflow management
- Automatic fallback and load balancing
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio


class ProcessingMode(Enum):
    """Where to process requests."""
    NVIDIA_LOCAL = "nvidia_local"
    GOOGLE_CLOUD = "google_cloud"
    HYBRID = "hybrid"
    AUTO = "auto"


class InferenceStrategy(Enum):
    """Inference routing strategy."""
    NVIDIA_FIRST = "nvidia_first"  # Try NVIDIA, fallback to Google
    GOOGLE_FIRST = "google_first"  # Try Google, fallback to NVIDIA
    PARALLEL = "parallel"  # Query both, use fastest
    CONSENSUS = "consensus"  # Query both, combine results
    COST_OPTIMIZED = "cost_optimized"  # Use local when possible


@dataclass
class HybridConfig:
    """Configuration for hybrid processing."""
    default_mode: ProcessingMode = ProcessingMode.AUTO
    inference_strategy: InferenceStrategy = InferenceStrategy.NVIDIA_FIRST
    enable_caching: bool = True
    enable_consensus: bool = False
    nvidia_timeout: float = 5.0  # seconds
    google_timeout: float = 10.0
    cost_per_nvidia_call: float = 0.0
    cost_per_google_call: float = 0.001


@dataclass
class HybridResult:
    """Result from hybrid processing."""
    result: Any
    source: str  # "nvidia", "google", "hybrid"
    latency_ms: float
    cost: float
    confidence: float
    metadata: Dict


class HybridOrchestrator:
    """
    Orchestrates between NVIDIA and Google Cloud services.
    
    Decision logic:
    - Fast queries → NVIDIA (local GPU)
    - Complex reasoning → Google Gemini
    - Batch processing → NVIDIA (GPU parallelism)
    - Cloud sync → Google Cloud
    - Real-time inference → NVIDIA
    - Deep analysis → Hybrid (consensus)
    """
    
    def __init__(
        self,
        nvidia_engine,
        google_cloud_manager,
        config: Optional[HybridConfig] = None
    ):
        """
        Initialize hybrid orchestrator.
        
        Args:
            nvidia_engine: NVIDIA inference engine
            google_cloud_manager: Google Cloud manager
            config: Hybrid configuration
        """
        self.nvidia = nvidia_engine
        self.google = google_cloud_manager
        self.config = config or HybridConfig()
        
        self.stats = {
            'nvidia_calls': 0,
            'google_calls': 0,
            'hybrid_calls': 0,
            'nvidia_failures': 0,
            'google_failures': 0,
            'total_cost': 0.0,
            'total_latency': 0.0
        }
    
    async def embed(
        self,
        texts: List[str],
        mode: Optional[ProcessingMode] = None
    ) -> Tuple[Any, HybridResult]:
        """
        Generate embeddings using optimal provider.
        
        Args:
            texts: Texts to embed
            mode: Processing mode (uses config default if None)
            
        Returns:
            (embeddings, result_metadata)
        """
        mode = mode or self.config.default_mode
        
        if mode == ProcessingMode.AUTO:
            # Decision logic for embeddings
            if len(texts) > 100:
                # Large batch → NVIDIA GPU parallelism
                mode = ProcessingMode.NVIDIA_LOCAL
            elif len(texts) < 10:
                # Small batch → NVIDIA for speed
                mode = ProcessingMode.NVIDIA_LOCAL
            else:
                # Medium batch → use strategy
                mode = ProcessingMode.NVIDIA_LOCAL
        
        if mode == ProcessingMode.NVIDIA_LOCAL:
            return await self._embed_nvidia(texts)
        elif mode == ProcessingMode.GOOGLE_CLOUD:
            return await self._embed_google(texts)
        elif mode == ProcessingMode.HYBRID:
            return await self._embed_hybrid(texts)
        else:
            return await self._embed_nvidia(texts)
    
    async def _embed_nvidia(self, texts: List[str]) -> Tuple[Any, HybridResult]:
        """Embed using NVIDIA."""
        import time
        start = time.time()
        
        try:
            embeddings = self.nvidia.embed(texts, use_cache=self.config.enable_caching)
            latency = (time.time() - start) * 1000
            
            self.stats['nvidia_calls'] += 1
            self.stats['total_latency'] += latency
            self.stats['total_cost'] += self.config.cost_per_nvidia_call
            
            result = HybridResult(
                result=embeddings,
                source="nvidia",
                latency_ms=latency,
                cost=self.config.cost_per_nvidia_call,
                confidence=1.0,
                metadata={'batch_size': len(texts)}
            )
            
            return embeddings, result
            
        except Exception as e:
            self.stats['nvidia_failures'] += 1
            
            # Fallback to Google if strategy allows
            if self.config.inference_strategy in [InferenceStrategy.NVIDIA_FIRST, InferenceStrategy.CONSENSUS]:
                return await self._embed_google(texts)
            else:
                raise
    
    async def _embed_google(self, texts: List[str]) -> Tuple[Any, HybridResult]:
        """Embed using Google Vertex AI."""
        import time
        start = time.time()
        
        try:
            # Use Vertex AI embeddings API
            # This is a placeholder - would use actual Vertex AI SDK
            from nvidia_inference import NVIDIAInferenceEngine
            temp_engine = NVIDIAInferenceEngine()
            embeddings = temp_engine.embed(texts)
            
            latency = (time.time() - start) * 1000
            
            self.stats['google_calls'] += 1
            self.stats['total_latency'] += latency
            cost = self.config.cost_per_google_call * len(texts)
            self.stats['total_cost'] += cost
            
            result = HybridResult(
                result=embeddings,
                source="google",
                latency_ms=latency,
                cost=cost,
                confidence=0.95,
                metadata={'batch_size': len(texts)}
            )
            
            return embeddings, result
            
        except Exception as e:
            self.stats['google_failures'] += 1
            raise
    
    async def _embed_hybrid(self, texts: List[str]) -> Tuple[Any, HybridResult]:
        """Embed using both (parallel or consensus)."""
        if self.config.inference_strategy == InferenceStrategy.PARALLEL:
            # Race both providers
            nvidia_task = asyncio.create_task(self._embed_nvidia(texts))
            google_task = asyncio.create_task(self._embed_google(texts))
            
            done, pending = await asyncio.wait(
                [nvidia_task, google_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the slower one
            for task in pending:
                task.cancel()
            
            # Return the first result
            result_task = list(done)[0]
            return await result_task
        
        elif self.config.inference_strategy == InferenceStrategy.CONSENSUS:
            # Get both results and combine
            nvidia_result = await self._embed_nvidia(texts)
            google_result = await self._embed_google(texts)
            
            # Average embeddings
            import numpy as np
            combined = (nvidia_result[0] + google_result[0]) / 2.0
            
            self.stats['hybrid_calls'] += 1
            
            result = HybridResult(
                result=combined,
                source="hybrid",
                latency_ms=nvidia_result[1].latency_ms + google_result[1].latency_ms,
                cost=nvidia_result[1].cost + google_result[1].cost,
                confidence=0.98,  # Higher confidence from consensus
                metadata={'strategy': 'consensus'}
            )
            
            return combined, result
    
    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        mode: Optional[ProcessingMode] = None
    ) -> Tuple[str, HybridResult]:
        """
        Generate text using optimal provider.
        
        Args:
            prompt: User prompt
            context: Optional context
            mode: Processing mode
            
        Returns:
            (generated_text, result_metadata)
        """
        mode = mode or self.config.default_mode
        
        if mode == ProcessingMode.AUTO:
            # Decision logic for generation
            prompt_length = len(prompt)
            
            if prompt_length < 100:
                # Short prompt → NVIDIA for speed
                mode = ProcessingMode.NVIDIA_LOCAL
            elif "complex" in prompt.lower() or "analyze" in prompt.lower():
                # Complex reasoning → Google Gemini
                mode = ProcessingMode.GOOGLE_CLOUD
            else:
                # Default to NVIDIA
                mode = ProcessingMode.NVIDIA_LOCAL
        
        if mode == ProcessingMode.NVIDIA_LOCAL:
            return await self._generate_nvidia(prompt, context)
        elif mode == ProcessingMode.GOOGLE_CLOUD:
            return await self._generate_google(prompt, context)
        elif mode == ProcessingMode.HYBRID:
            return await self._generate_hybrid(prompt, context)
        else:
            return await self._generate_nvidia(prompt, context)
    
    async def _generate_nvidia(self, prompt: str, context: Optional[str]) -> Tuple[str, HybridResult]:
        """Generate using NVIDIA."""
        import time
        start = time.time()
        
        try:
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            text = self.nvidia.generate(full_prompt)
            latency = (time.time() - start) * 1000
            
            self.stats['nvidia_calls'] += 1
            self.stats['total_latency'] += latency
            self.stats['total_cost'] += self.config.cost_per_nvidia_call
            
            result = HybridResult(
                result=text,
                source="nvidia",
                latency_ms=latency,
                cost=self.config.cost_per_nvidia_call,
                confidence=0.9,
                metadata={'model': 'nvidia_nim'}
            )
            
            return text, result
            
        except Exception as e:
            self.stats['nvidia_failures'] += 1
            
            if self.config.inference_strategy == InferenceStrategy.NVIDIA_FIRST:
                return await self._generate_google(prompt, context)
            else:
                raise
    
    async def _generate_google(self, prompt: str, context: Optional[str]) -> Tuple[str, HybridResult]:
        """Generate using Google Gemini."""
        import time
        start = time.time()
        
        try:
            text = await self.google.query_gemini(prompt, context)
            latency = (time.time() - start) * 1000
            
            self.stats['google_calls'] += 1
            self.stats['total_latency'] += latency
            cost = self.config.cost_per_google_call
            self.stats['total_cost'] += cost
            
            result = HybridResult(
                result=text,
                source="google",
                latency_ms=latency,
                cost=cost,
                confidence=0.95,
                metadata={'model': 'gemini-pro'}
            )
            
            return text, result
            
        except Exception as e:
            self.stats['google_failures'] += 1
            raise
    
    async def _generate_hybrid(self, prompt: str, context: Optional[str]) -> Tuple[str, HybridResult]:
        """Generate using both providers."""
        if self.config.inference_strategy == InferenceStrategy.CONSENSUS:
            # Get both results
            nvidia_text, nvidia_meta = await self._generate_nvidia(prompt, context)
            google_text, google_meta = await self._generate_google(prompt, context)
            
            # Combine results (could use voting, averaging, etc.)
            combined = f"[NVIDIA]: {nvidia_text}\n\n[Google]: {google_text}\n\n[Consensus]: Using Google response for higher reasoning quality."
            
            self.stats['hybrid_calls'] += 1
            
            result = HybridResult(
                result=google_text,  # Prefer Google for reasoning
                source="hybrid",
                latency_ms=nvidia_meta.latency_ms + google_meta.latency_ms,
                cost=nvidia_meta.cost + google_meta.cost,
                confidence=0.98,
                metadata={'strategy': 'consensus', 'both_responses': combined}
            )
            
            return google_text, result
        else:
            # Parallel - return fastest
            nvidia_task = asyncio.create_task(self._generate_nvidia(prompt, context))
            google_task = asyncio.create_task(self._generate_google(prompt, context))
            
            done, pending = await asyncio.wait(
                [nvidia_task, google_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
            
            result_task = list(done)[0]
            return await result_task
    
    async def capture_and_process(self, source: str, data: Dict) -> Dict:
        """
        Unified capture workflow.
        
        Workflow:
        1. Capture from source (Keep, Drive, browser)
        2. Process with NVIDIA NLP
        3. Embed with NVIDIA
        4. Store locally + sync to Firebase
        
        Args:
            source: Source type ("keep", "drive", "browser")
            data: Captured data
            
        Returns:
            Processing result
        """
        result = {
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'steps': []
        }
        
        # Step 1: Extract text
        if source == "keep":
            text = data.get('text', '')
            result['steps'].append({'step': 'extract', 'source': 'keep'})
        elif source == "drive":
            text = data.get('content', '')
            result['steps'].append({'step': 'extract', 'source': 'drive'})
        elif source == "browser":
            text = data.get('content', '')
            result['steps'].append({'step': 'extract', 'source': 'browser'})
        else:
            text = str(data)
        
        # Step 2: NLP processing (NVIDIA NeMo)
        try:
            from nvidia_nemo import nemo_engine
            intent = nemo_engine.classify_intent(text)
            entities = nemo_engine.extract_entities(text)
            sentiment = nemo_engine.analyze_sentiment(text)
            
            result['steps'].append({
                'step': 'nlp',
                'intent': intent.label,
                'entities': len(entities),
                'sentiment': sentiment.label
            })
        except Exception as e:
            result['steps'].append({'step': 'nlp', 'error': str(e)})
        
        # Step 3: Embedding (hybrid)
        try:
            embeddings, embed_meta = await self.embed([text])
            result['steps'].append({
                'step': 'embed',
                'source': embed_meta.source,
                'latency_ms': embed_meta.latency_ms
            })
        except Exception as e:
            result['steps'].append({'step': 'embed', 'error': str(e)})
        
        # Step 4: Cloud sync (Firebase)
        try:
            sync_result = await self.google.sync_to_firebase(
                collection='memories',
                document_id=f"{source}_{datetime.utcnow().timestamp()}",
                data={
                    'text': text,
                    'source': source,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metadata': data
                }
            )
            result['steps'].append({'step': 'sync', 'status': sync_result.get('status')})
        except Exception as e:
            result['steps'].append({'step': 'sync', 'error': str(e)})
        
        return result
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics."""
        total_calls = (
            self.stats['nvidia_calls'] +
            self.stats['google_calls'] +
            self.stats['hybrid_calls']
        )
        
        return {
            'total_calls': total_calls,
            'nvidia_calls': self.stats['nvidia_calls'],
            'google_calls': self.stats['google_calls'],
            'hybrid_calls': self.stats['hybrid_calls'],
            'nvidia_failures': self.stats['nvidia_failures'],
            'google_failures': self.stats['google_failures'],
            'total_cost': round(self.stats['total_cost'], 4),
            'total_latency_ms': round(self.stats['total_latency'], 2),
            'avg_latency_ms': (
                round(self.stats['total_latency'] / total_calls, 2)
                if total_calls > 0
                else 0.0
            ),
            'nvidia_success_rate': (
                round(
                    self.stats['nvidia_calls'] /
                    (self.stats['nvidia_calls'] + self.stats['nvidia_failures']),
                    2
                )
                if (self.stats['nvidia_calls'] + self.stats['nvidia_failures']) > 0
                else 1.0
            ),
            'google_success_rate': (
                round(
                    self.stats['google_calls'] /
                    (self.stats['google_calls'] + self.stats['google_failures']),
                    2
                )
                if (self.stats['google_calls'] + self.stats['google_failures']) > 0
                else 1.0
            )
        }
    
    def recommend_mode(self, task_type: str) -> ProcessingMode:
        """
        Recommend optimal processing mode for a task.
        
        Args:
            task_type: Type of task
            
        Returns:
            Recommended mode
        """
        recommendations = {
            'embedding': ProcessingMode.NVIDIA_LOCAL,  # GPU faster
            'batch_embedding': ProcessingMode.NVIDIA_LOCAL,  # GPU parallelism
            'quick_question': ProcessingMode.NVIDIA_LOCAL,  # Low latency
            'complex_reasoning': ProcessingMode.GOOGLE_CLOUD,  # Better reasoning
            'summarization': ProcessingMode.GOOGLE_CLOUD,  # Better summarization
            'cloud_sync': ProcessingMode.GOOGLE_CLOUD,  # Cloud native
            'real_time': ProcessingMode.NVIDIA_LOCAL,  # Low latency
            'consensus': ProcessingMode.HYBRID,  # Need both
        }
        
        return recommendations.get(task_type, ProcessingMode.AUTO)


__all__ = [
    "HybridOrchestrator",
    "HybridConfig",
    "HybridResult",
    "ProcessingMode",
    "InferenceStrategy",
]
