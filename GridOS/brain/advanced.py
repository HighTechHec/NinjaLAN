"""
Advanced Features Module

Enterprise-grade features:
- Webhook notifications
- Auto-tagging with NLP
- Smart recommendations
- Duplicate detection
- Analytics and insights
- Background job processing
"""

import time
import hashlib
import re
from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import requests
from enum import Enum


class EventType(Enum):
    """Webhook event types"""
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    SEARCH_PERFORMED = "search.performed"
    QUESTION_ANSWERED = "question.answered"
    USER_CREATED = "user.created"
    API_KEY_CREATED = "api_key.created"


@dataclass
class Webhook:
    """Webhook configuration"""
    webhook_id: str
    url: str
    events: Set[EventType]
    secret: str
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    trigger_count: int = 0


class WebhookManager:
    """
    Webhook notification system
    
    Sends HTTP POST notifications for events
    """
    
    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.event_log: List[Dict] = []
    
    def register_webhook(self, url: str, events: List[EventType], secret: str) -> str:
        """Register a new webhook"""
        webhook_id = hashlib.sha256(f"{url}{time.time()}".encode()).hexdigest()[:16]
        
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            events=set(events),
            secret=secret
        )
        
        self.webhooks[webhook_id] = webhook
        return webhook_id
    
    def trigger_event(self, event_type: EventType, data: Dict):
        """Trigger webhooks for an event"""
        event = {
            'event_type': event_type.value,
            'data': data,
            'timestamp': time.time()
        }
        
        self.event_log.append(event)
        
        # Trigger matching webhooks
        for webhook in self.webhooks.values():
            if webhook.is_active and event_type in webhook.events:
                self._send_webhook(webhook, event)
    
    def _send_webhook(self, webhook: Webhook, event: Dict):
        """Send webhook notification"""
        try:
            # Create signature
            signature = hashlib.sha256(
                f"{webhook.secret}{json.dumps(event)}".encode()
            ).hexdigest()
            
            headers = {
                'Content-Type': 'application/json',
                'X-Webhook-Signature': signature
            }
            
            response = requests.post(
                webhook.url,
                json=event,
                headers=headers,
                timeout=10
            )
            
            webhook.last_triggered = time.time()
            webhook.trigger_count += 1
            
        except Exception as e:
            print(f"Webhook failed: {e}")
    
    def list_webhooks(self) -> List[Webhook]:
        """List all webhooks"""
        return list(self.webhooks.values())
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False


class AutoTagger:
    """
    Automatic tagging using NLP techniques
    
    Features:
    - Keyword extraction
    - Entity recognition
    - Topic detection
    - Category classification
    """
    
    def __init__(self):
        # Common stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which'
        }
        
        # Technology keywords
        self.tech_keywords = {
            'python', 'java', 'javascript', 'api', 'database', 'sql', 'nosql',
            'machine learning', 'ai', 'neural network', 'deep learning',
            'gpu', 'cpu', 'cloud', 'docker', 'kubernetes', 'microservices',
            'rest', 'graphql', 'redis', 'mongodb', 'postgresql', 'mysql'
        }
    
    def extract_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Extract relevant tags from text"""
        tags = set()
        text_lower = text.lower()
        
        # Extract technology keywords
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                tags.add(keyword.replace(' ', '-'))
        
        # Extract capitalized words (potential entities/proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for word in words:
            if len(word) > 2 and word.lower() not in self.stop_words:
                tags.add(word.lower())
        
        # Extract frequent words
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        word_freq = Counter(words)
        
        for word, freq in word_freq.most_common(10):
            if word not in self.stop_words and freq > 1:
                tags.add(word)
        
        # Return top tags
        return list(tags)[:max_tags]
    
    def suggest_category(self, text: str) -> str:
        """Suggest a category based on content"""
        text_lower = text.lower()
        
        categories = {
            'technology': ['code', 'software', 'api', 'database', 'programming'],
            'science': ['research', 'study', 'experiment', 'theory', 'hypothesis'],
            'business': ['market', 'strategy', 'revenue', 'customer', 'sales'],
            'education': ['learn', 'teach', 'course', 'lesson', 'tutorial'],
            'health': ['medical', 'health', 'disease', 'treatment', 'patient']
        }
        
        scores = defaultdict(int)
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'general'


class RecommendationEngine:
    """
    Smart recommendation system
    
    Recommends related memories based on:
    - Content similarity
    - Tag overlap
    - Access patterns
    - Temporal proximity
    """
    
    def __init__(self, vector_db, memory_store):
        self.vector_db = vector_db
        self.memory_store = memory_store
        self.access_patterns = defaultdict(list)
    
    def record_access(self, user_id: str, memory_id: str):
        """Record memory access for pattern analysis"""
        self.access_patterns[user_id].append({
            'memory_id': memory_id,
            'timestamp': time.time()
        })
    
    def recommend_similar(self, memory_id: str, top_k: int = 5) -> List[Dict]:
        """Recommend similar memories"""
        memory = self.memory_store.get_memory(memory_id)
        if not memory:
            return []
        
        # Use vector similarity (would need embedding)
        # For now, use tag-based similarity
        recommendations = []
        
        for mid, mem in self.memory_store.memories.items():
            if mid == memory_id:
                continue
            
            # Calculate tag overlap
            tag_overlap = len(set(memory.tags) & set(mem.tags))
            if tag_overlap > 0:
                recommendations.append({
                    'memory_id': mid,
                    'content': mem.content[:100],
                    'score': tag_overlap / max(len(memory.tags), len(mem.tags)),
                    'reason': f"{tag_overlap} shared tags"
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_k]
    
    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """Recommend memories based on user's access patterns"""
        recent_accesses = self.access_patterns.get(user_id, [])
        
        if not recent_accesses:
            return []
        
        # Get recently accessed memory IDs
        recent_memory_ids = [acc['memory_id'] for acc in recent_accesses[-10:]]
        
        # Find similar memories
        all_recommendations = []
        for memory_id in recent_memory_ids:
            similar = self.recommend_similar(memory_id, top_k=3)
            all_recommendations.extend(similar)
        
        # Deduplicate and score
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec['memory_id'] not in seen:
                seen.add(rec['memory_id'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:top_k]


class DuplicateDetector:
    """
    Duplicate and near-duplicate detection
    
    Features:
    - Exact duplicate detection
    - Fuzzy matching
    - Similarity threshold-based detection
    """
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.content_hashes: Dict[str, str] = {}
    
    def compute_hash(self, text: str) -> str:
        """Compute hash for exact duplicate detection"""
        normalized = ' '.join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def is_exact_duplicate(self, text: str) -> Optional[str]:
        """Check if text is an exact duplicate"""
        text_hash = self.compute_hash(text)
        return self.content_hashes.get(text_hash)
    
    def register_content(self, memory_id: str, text: str):
        """Register content for duplicate detection"""
        text_hash = self.compute_hash(text)
        self.content_hashes[text_hash] = memory_id
    
    def find_similar(self, text: str, candidates: List[str]) -> List[tuple]:
        """Find similar content using Jaccard similarity"""
        text_words = set(text.lower().split())
        
        similarities = []
        for candidate in candidates:
            candidate_words = set(candidate.lower().split())
            
            intersection = text_words & candidate_words
            union = text_words | candidate_words
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity >= self.threshold:
                    similarities.append((candidate, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)


class AnalyticsEngine:
    """
    Advanced analytics and insights
    
    Tracks:
    - Usage metrics
    - Memory health
    - Retention patterns
    - Search effectiveness
    - User behavior
    """
    
    def __init__(self):
        self.metrics: Dict[str, List] = defaultdict(list)
        self.events: List[Dict] = []
    
    def record_event(self, event_type: str, data: Dict):
        """Record an event for analytics"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.events.append(event)
        self.metrics[event_type].append(event)
    
    def get_memory_health_score(self, memory_store) -> float:
        """Calculate overall memory health score"""
        if not memory_store.memories:
            return 0.0
        
        total_retention = 0.0
        for memory in memory_store.memories.values():
            total_retention += memory.calculate_retention()
        
        avg_retention = total_retention / len(memory_store.memories)
        
        # Factor in access patterns
        avg_access_count = sum(m.access_count for m in memory_store.memories.values()) / len(memory_store.memories)
        
        # Combined score
        health_score = (avg_retention * 0.7) + (min(avg_access_count / 10, 1.0) * 0.3)
        
        return health_score
    
    def get_usage_stats(self, time_range: int = 86400) -> Dict:
        """Get usage statistics for time range (default: 24 hours)"""
        cutoff = time.time() - time_range
        recent_events = [e for e in self.events if e['timestamp'] > cutoff]
        
        event_counts = Counter(e['type'] for e in recent_events)
        
        return {
            'time_range_hours': time_range / 3600,
            'total_events': len(recent_events),
            'event_breakdown': dict(event_counts),
            'events_per_hour': len(recent_events) / (time_range / 3600)
        }
    
    def get_retention_insights(self, memory_store) -> Dict:
        """Analyze memory retention patterns"""
        if not memory_store.memories:
            return {}
        
        retention_scores = [m.calculate_retention() for m in memory_store.memories.values()]
        
        return {
            'avg_retention': sum(retention_scores) / len(retention_scores),
            'min_retention': min(retention_scores),
            'max_retention': max(retention_scores),
            'memories_at_risk': sum(1 for r in retention_scores if r < 0.3),
            'memories_strong': sum(1 for r in retention_scores if r > 0.8)
        }
    
    def get_search_effectiveness(self) -> Dict:
        """Analyze search effectiveness"""
        search_events = self.metrics.get('search', [])
        
        if not search_events:
            return {'searches': 0}
        
        return {
            'total_searches': len(search_events),
            'avg_results': sum(e['data'].get('result_count', 0) for e in search_events) / len(search_events),
            'searches_with_results': sum(1 for e in search_events if e['data'].get('result_count', 0) > 0)
        }


class BackgroundJobProcessor:
    """
    Background job processing system
    
    Handles:
    - Scheduled tasks
    - Periodic cleanup
    - Maintenance operations
    - Async processing
    """
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.job_history: List[Dict] = []
    
    def register_job(
        self,
        name: str,
        function: Callable,
        schedule: str = "daily",
        enabled: bool = True
    ):
        """Register a background job"""
        self.jobs[name] = {
            'function': function,
            'schedule': schedule,
            'enabled': enabled,
            'last_run': None,
            'next_run': self._calculate_next_run(schedule)
        }
    
    def _calculate_next_run(self, schedule: str) -> float:
        """Calculate next run time based on schedule"""
        schedules = {
            'hourly': 3600,
            'daily': 86400,
            'weekly': 604800
        }
        
        interval = schedules.get(schedule, 86400)
        return time.time() + interval
    
    def run_due_jobs(self):
        """Run jobs that are due"""
        current_time = time.time()
        
        for name, job in self.jobs.items():
            if job['enabled'] and job['next_run'] <= current_time:
                try:
                    result = job['function']()
                    job['last_run'] = current_time
                    job['next_run'] = self._calculate_next_run(job['schedule'])
                    
                    self.job_history.append({
                        'job': name,
                        'timestamp': current_time,
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    self.job_history.append({
                        'job': name,
                        'timestamp': current_time,
                        'status': 'error',
                        'error': str(e)
                    })
    
    def get_job_status(self) -> List[Dict]:
        """Get status of all jobs"""
        status = []
        for name, job in self.jobs.items():
            status.append({
                'name': name,
                'schedule': job['schedule'],
                'enabled': job['enabled'],
                'last_run': job['last_run'],
                'next_run': job['next_run']
            })
        return status


if __name__ == '__main__':
    # Demo
    print("=== Advanced Features Demo ===\n")
    
    # Auto-tagging
    print("1. Auto-Tagging:")
    tagger = AutoTagger()
    text = "Machine learning with Python using neural networks for AI applications"
    tags = tagger.extract_tags(text)
    print(f"   Text: {text}")
    print(f"   Tags: {tags}")
    print(f"   Category: {tagger.suggest_category(text)}")
    
    # Analytics
    print("\n2. Analytics:")
    analytics = AnalyticsEngine()
    analytics.record_event('search', {'query': 'AI', 'result_count': 5})
    analytics.record_event('search', {'query': 'ML', 'result_count': 3})
    stats = analytics.get_usage_stats()
    print(f"   Usage stats: {json.dumps(stats, indent=2)}")
    
    # Duplicate detection
    print("\n3. Duplicate Detection:")
    detector = DuplicateDetector(threshold=0.8)
    text1 = "This is a test document about AI"
    text2 = "This is a test document about ML"
    detector.register_content("mem1", text1)
    is_dup = detector.is_exact_duplicate(text2)
    print(f"   Text 1: {text1}")
    print(f"   Text 2: {text2}")
    print(f"   Is duplicate: {is_dup is not None}")
    
    print("\n✅ Advanced features demo complete!")
