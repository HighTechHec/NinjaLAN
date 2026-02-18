"""
Core Memory System + Knowledge Graph

Multi-tier memory architecture:
- Long-term: Persistent semantic knowledge
- Short-term: Working memory (TTL: 1 hour)
- Episodic: Time-stamped events with context
- Semantic: Concepts and relationships

Features:
- Ebbinghaus forgetting curve (exponential decay)
- Spaced repetition scheduling
- Multi-hop knowledge graph (Neo4j)
- Entity linking and semantic relationships
"""

import time
import math
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import uuid

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


@dataclass
class MemoryNode:
    """Represents a single memory unit with metadata."""
    id: str
    content: str
    memory_type: str  # 'long_term', 'short_term', 'episodic', 'semantic'
    timestamp: float
    last_accessed: float
    access_count: int = 0
    strength: float = 1.0  # Memory strength (0-1)
    decay_rate: float = 0.1  # Decay constant for forgetting curve
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def calculate_retention(self, current_time: Optional[float] = None) -> float:
        """
        Calculate memory retention using Ebbinghaus forgetting curve.
        R = e^(-t/S)
        where t = time elapsed, S = memory strength
        """
        if current_time is None:
            current_time = time.time()
        
        time_elapsed = (current_time - self.last_accessed) / 86400  # days
        retention = math.exp(-time_elapsed * self.decay_rate / self.strength)
        return max(0.0, min(1.0, retention))
    
    def access(self):
        """Update access metadata and strengthen memory."""
        self.last_accessed = time.time()
        self.access_count += 1
        # Strengthen memory with each access (logarithmic)
        self.strength = min(1.0, self.strength + 0.1 * math.log(self.access_count + 1))
    
    def should_review(self) -> Tuple[bool, int]:
        """
        Determine if memory needs review based on spaced repetition.
        Returns: (should_review, days_until_next_review)
        
        Spaced repetition intervals: 1, 3, 7, 14, 30, 90 days
        """
        days_since_access = (time.time() - self.last_accessed) / 86400
        
        # Define intervals based on access count
        intervals = [1, 3, 7, 14, 30, 90]
        interval_index = min(self.access_count, len(intervals) - 1)
        target_interval = intervals[interval_index]
        
        should_review = days_since_access >= target_interval
        days_until = max(0, target_interval - days_since_access)
        
        return should_review, int(days_until)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MemoryStore:
    """
    Multi-tier memory storage with automatic decay and spaced repetition.
    """
    
    def __init__(self):
        self.memories: Dict[str, MemoryNode] = {}
        self.long_term: Set[str] = set()
        self.short_term: Set[str] = set()
        self.episodic: Set[str] = set()
        self.semantic: Set[str] = set()
        
        # Index for fast lookups
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.time_index: List[Tuple[float, str]] = []
        
    def add_memory(
        self,
        content: str,
        memory_type: str = 'long_term',
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new memory to the store."""
        memory_id = self._generate_id(content)
        
        if memory_id in self.memories:
            # Memory exists, just access it
            self.memories[memory_id].access()
            return memory_id
        
        node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            last_accessed=time.time(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.memories[memory_id] = node
        
        # Add to appropriate tier
        if memory_type == 'long_term':
            self.long_term.add(memory_id)
        elif memory_type == 'short_term':
            self.short_term.add(memory_id)
        elif memory_type == 'episodic':
            self.episodic.add(memory_id)
        elif memory_type == 'semantic':
            self.semantic.add(memory_id)
        
        # Update indexes
        for tag in node.tags:
            self.tag_index[tag].add(memory_id)
        self.time_index.append((node.timestamp, memory_id))
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory by ID and update access metadata."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            return memory
        return None
    
    def search_by_tag(self, tag: str) -> List[MemoryNode]:
        """Find all memories with a specific tag."""
        memory_ids = self.tag_index.get(tag, set())
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]
    
    def search_by_time_range(self, start: float, end: float) -> List[MemoryNode]:
        """Find memories within a time range."""
        results = []
        for timestamp, memory_id in self.time_index:
            if start <= timestamp <= end and memory_id in self.memories:
                results.append(self.memories[memory_id])
        return results
    
    def get_memories_for_review(self, limit: int = 10) -> List[MemoryNode]:
        """Get memories that need review based on spaced repetition."""
        review_candidates = []
        
        for memory_id, memory in self.memories.items():
            should_review, _ = memory.should_review()
            if should_review:
                review_candidates.append((memory.calculate_retention(), memory))
        
        # Sort by retention (lowest first - most forgotten)
        review_candidates.sort(key=lambda x: x[0])
        
        return [memory for _, memory in review_candidates[:limit]]
    
    def cleanup_weak_memories(self, threshold: float = 0.1):
        """Remove memories with retention below threshold."""
        to_remove = []
        current_time = time.time()
        
        for memory_id, memory in self.memories.items():
            if memory.calculate_retention(current_time) < threshold:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self._remove_memory(memory_id)
        
        return len(to_remove)
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory and clean up indexes."""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Remove from tiers
        self.long_term.discard(memory_id)
        self.short_term.discard(memory_id)
        self.episodic.discard(memory_id)
        self.semantic.discard(memory_id)
        
        # Remove from indexes
        for tag in memory.tags:
            self.tag_index[tag].discard(memory_id)
        
        # Remove from time index
        self.time_index = [(t, mid) for t, mid in self.time_index if mid != memory_id]
        
        del self.memories[memory_id]
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict:
        """Get memory store statistics."""
        return {
            'total_memories': len(self.memories),
            'long_term': len(self.long_term),
            'short_term': len(self.short_term),
            'episodic': len(self.episodic),
            'semantic': len(self.semantic),
            'unique_tags': len(self.tag_index),
        }


class KnowledgeGraph:
    """
    Neo4j-backed knowledge graph for semantic relationships.
    Supports entity linking, multi-hop traversal, and semantic search.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self._create_indexes()
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}")
                self.driver = None
        else:
            print("Warning: neo4j package not available. KnowledgeGraph will run in mock mode.")
    
    def _create_indexes(self):
        """Create indexes for fast lookups."""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            # Create indexes
            try:
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")
            except Exception as e:
                print(f"Index creation warning: {e}")
    
    def add_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None) -> str:
        """Add an entity to the knowledge graph."""
        if not self.driver:
            return f"mock_{uuid.uuid4().hex[:8]}"
        
        props = properties or {}
        props.update({'name': name, 'type': entity_type, 'created': time.time()})
        
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e += $props
                RETURN e.name as name
                """,
                name=name,
                props=props
            )
            record = result.single()
            return record['name'] if record else name
    
    def add_relationship(self, entity1: str, relation: str, entity2: str, properties: Optional[Dict] = None):
        """Create a relationship between two entities."""
        if not self.driver:
            return
        
        props = properties or {}
        props['created'] = time.time()
        
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e1:Entity {name: $entity1})
                MATCH (e2:Entity {name: $entity2})
                MERGE (e1)-[r:RELATES {type: $relation}]->(e2)
                SET r += $props
                """,
                entity1=entity1,
                entity2=entity2,
                relation=relation,
                props=props
            )
    
    def find_paths(self, entity1: str, entity2: str, max_depth: int = 3) -> List[List[str]]:
        """Find all paths between two entities (multi-hop traversal)."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (e1:Entity {name: $entity1})-[*1..$max_depth]-(e2:Entity {name: $entity2})
                RETURN [node IN nodes(path) | node.name] as path_nodes
                LIMIT 10
                """,
                entity1=entity1,
                entity2=entity2,
                max_depth=max_depth
            )
            
            paths = []
            for record in result:
                paths.append(record['path_nodes'])
            return paths
    
    def get_neighbors(self, entity: str, depth: int = 1) -> List[Dict]:
        """Get neighboring entities (BFS traversal)."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $entity})-[r*1..$depth]-(neighbor:Entity)
                RETURN DISTINCT neighbor.name as name, neighbor.type as type
                LIMIT 50
                """,
                entity=entity,
                depth=depth
            )
            
            neighbors = []
            for record in result:
                neighbors.append({
                    'name': record['name'],
                    'type': record['type']
                })
            return neighbors
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for entities matching the query."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query OR e.type CONTAINS $query
                RETURN e.name as name, e.type as type
                LIMIT $limit
                """,
                query=query,
                limit=limit
            )
            
            results = []
            for record in result:
                results.append({
                    'name': record['name'],
                    'type': record['type']
                })
            return results
    
    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        if not self.driver:
            return {'entities': 0, 'relationships': 0}
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT e) as entities, count(DISTINCT r) as relationships
                """
            )
            record = result.single()
            return {
                'entities': record['entities'],
                'relationships': record['relationships']
            } if record else {'entities': 0, 'relationships': 0}
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()


class SecondBrain:
    """
    Main orchestrator combining memory store and knowledge graph.
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.memory_store = MemoryStore()
        self.knowledge_graph = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
    
    def ingest(self, content: str, memory_type: str = 'long_term', tags: Optional[List[str]] = None, entities: Optional[List[str]] = None):
        """
        Ingest new information into the second brain.
        Stores in memory and extracts entities for knowledge graph.
        """
        # Add to memory store
        memory_id = self.memory_store.add_memory(content, memory_type, tags)
        
        # Extract and link entities
        if entities:
            for entity in entities:
                self.knowledge_graph.add_entity(entity, 'concept')
                # Link entity to content
                self.knowledge_graph.add_relationship(
                    entity,
                    'mentioned_in',
                    memory_id,
                    {'content_preview': content[:100]}
                )
        
        return memory_id
    
    def retrieve(self, query: str, method: str = 'tag') -> List[MemoryNode]:
        """
        Retrieve memories based on query.
        Methods: 'tag', 'semantic', 'temporal'
        """
        if method == 'tag':
            return self.memory_store.search_by_tag(query)
        elif method == 'semantic':
            # Use knowledge graph for semantic search
            entities = self.knowledge_graph.semantic_search(query)
            # For now, return all memories (would integrate with vector search)
            return list(self.memory_store.memories.values())[:10]
        elif method == 'temporal':
            # Last 24 hours
            end = time.time()
            start = end - 86400
            return self.memory_store.search_by_time_range(start, end)
        return []
    
    def get_review_queue(self, limit: int = 10) -> List[MemoryNode]:
        """Get memories that need review."""
        return self.memory_store.get_memories_for_review(limit)
    
    def cleanup(self, threshold: float = 0.1):
        """Remove weak memories."""
        return self.memory_store.cleanup_weak_memories(threshold)
    
    def get_comprehensive_stats(self) -> Dict:
        """Get statistics from all components."""
        return {
            'memory': self.memory_store.get_stats(),
            'knowledge_graph': self.knowledge_graph.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def close(self):
        """Close connections."""
        self.knowledge_graph.close()


if __name__ == '__main__':
    # Demo usage
    brain = SecondBrain()
    
    # Ingest some knowledge
    brain.ingest(
        "Python is a high-level programming language known for readability.",
        memory_type='semantic',
        tags=['python', 'programming'],
        entities=['Python', 'programming language']
    )
    
    brain.ingest(
        "Machine learning uses statistical techniques to give computers the ability to learn.",
        memory_type='semantic',
        tags=['machine-learning', 'AI'],
        entities=['machine learning', 'AI']
    )
    
    # Retrieve and display stats
    stats = brain.get_comprehensive_stats()
    print(json.dumps(stats, indent=2))
    
    brain.close()
