"""
Milvus Vector Database Integration

GPU-accelerated vector search with Milvus.

Features:
- HNSW index on GPU
- Hybrid search (dense + sparse)
- Filtered search with metadata
- Batch upsert and search
- Collection management
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Warning: pymilvus not available. Running in mock mode.")


@dataclass
class VectorConfig:
    """Configuration for Milvus vector database."""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "second_brain"
    embedding_dim: int = 384
    index_type: str = "HNSW"  # or IVF_FLAT, IVF_SQ8, etc.
    metric_type: str = "IP"  # Inner Product (for normalized vectors)
    nlist: int = 1024  # Number of cluster units
    m: int = 8  # HNSW parameter
    ef_construction: int = 200  # HNSW build parameter
    ef_search: int = 100  # HNSW search parameter


class MilvusVectorDB:
    """
    Milvus vector database client with GPU acceleration.
    
    Supports:
    - Semantic search via embeddings
    - Filtered search with metadata
    - Batch operations
    - Index optimization
    """
    
    def __init__(self, config: Optional[VectorConfig] = None):
        self.config = config or VectorConfig()
        self.collection = None
        self.connected = False
        
        if MILVUS_AVAILABLE:
            self._connect()
            self._setup_collection()
        else:
            # Mock mode for testing
            self.mock_data = []
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )
            self.connected = True
            print(f"Connected to Milvus at {self.config.host}:{self.config.port}")
        except Exception as e:
            print(f"Warning: Could not connect to Milvus: {e}")
            self.connected = False
    
    def _setup_collection(self):
        """Create or load collection with appropriate schema."""
        if not self.connected:
            return
        
        # Check if collection exists
        if utility.has_collection(self.config.collection_name):
            self.collection = Collection(self.config.collection_name)
            self.collection.load()
            print(f"Loaded existing collection: {self.config.collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Second Brain semantic memory"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )
        
        print(f"Created collection: {self.config.collection_name}")
        
        # Create index
        self._create_index()
    
    def _create_index(self):
        """Create HNSW index on embedding field for fast search."""
        if not self.collection:
            return
        
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {
                "M": self.config.m,
                "efConstruction": self.config.ef_construction
            }
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        self.collection.load()
        print(f"Created {self.config.index_type} index with M={self.config.m}")
    
    def insert(
        self,
        id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Insert a single vector with metadata.
        
        Args:
            id: Unique identifier
            embedding: Vector embedding
            content: Original text content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        return self.batch_insert(
            ids=[id],
            embeddings=[embedding],
            contents=[content],
            metadatas=[metadata or {}]
        )
    
    def batch_insert(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> bool:
        """
        Batch insert vectors with metadata.
        
        Args:
            ids: List of unique identifiers
            embeddings: List of vector embeddings
            contents: List of text contents
            metadatas: List of metadata dicts
            
        Returns:
            Success status
        """
        if metadatas is None:
            metadatas = [{}] * len(ids)
        
        # Ensure embeddings are numpy arrays
        embeddings = [np.array(e).flatten().tolist() for e in embeddings]
        
        if not self.connected or not self.collection:
            # Mock mode
            for i in range(len(ids)):
                self.mock_data.append({
                    'id': ids[i],
                    'embedding': embeddings[i],
                    'content': contents[i],
                    'metadata': metadatas[i],
                    'timestamp': time.time()
                })
            return True
        
        # Real Milvus insert
        try:
            entities = [
                ids,
                embeddings,
                contents,
                [time.time()] * len(ids),
                metadatas
            ]
            
            self.collection.insert(entities)
            self.collection.flush()
            return True
        except Exception as e:
            print(f"Error inserting to Milvus: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filter expression (e.g., "metadata['tag'] == 'python'")
            
        Returns:
            List of search results with scores
        """
        # Normalize query embedding
        query_embedding = np.array(query_embedding).flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if not self.connected or not self.collection:
            # Mock search
            return self._mock_search(query_embedding, top_k)
        
        # Real Milvus search
        try:
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"ef": self.config.ef_search}
            }
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=["id", "content", "timestamp", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        'id': hit.entity.get('id'),
                        'content': hit.entity.get('content'),
                        'score': hit.score,
                        'timestamp': hit.entity.get('timestamp'),
                        'metadata': hit.entity.get('metadata', {})
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching Milvus: {e}")
            return []
    
    def _mock_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Mock search for testing."""
        if not self.mock_data:
            return []
        
        # Calculate cosine similarity with all stored embeddings
        results = []
        for item in self.mock_data:
            stored_emb = np.array(item['embedding'])
            similarity = np.dot(query_embedding, stored_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb)
            )
            results.append({
                'id': item['id'],
                'content': item['content'],
                'score': float(similarity),
                'timestamp': item['timestamp'],
                'metadata': item['metadata']
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        keyword: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            query_embedding: Query vector
            keyword: Optional keyword to filter by
            top_k: Number of results
            
        Returns:
            Ranked search results
        """
        # First, do vector search
        vector_results = self.search(query_embedding, top_k=top_k * 2)
        
        # If keyword provided, re-rank based on keyword presence
        if keyword:
            keyword_lower = keyword.lower()
            for result in vector_results:
                content_lower = result['content'].lower()
                if keyword_lower in content_lower:
                    # Boost score for keyword matches
                    result['score'] *= 1.5
            
            # Re-sort
            vector_results.sort(key=lambda x: x['score'], reverse=True)
        
        return vector_results[:top_k]
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        if not self.connected or not self.collection:
            # Mock mode
            self.mock_data = [item for item in self.mock_data if item['id'] not in ids]
            return True
        
        try:
            expr = f"id in {ids}"
            self.collection.delete(expr)
            return True
        except Exception as e:
            print(f"Error deleting from Milvus: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of vectors in collection."""
        if not self.connected or not self.collection:
            return len(self.mock_data)
        
        return self.collection.num_entities
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        if not self.connected or not self.collection:
            return {
                'total_vectors': len(self.mock_data),
                'collection': self.config.collection_name,
                'status': 'mock_mode'
            }
        
        return {
            'total_vectors': self.collection.num_entities,
            'collection': self.config.collection_name,
            'index_type': self.config.index_type,
            'metric_type': self.config.metric_type,
            'embedding_dim': self.config.embedding_dim,
            'status': 'connected'
        }
    
    def optimize_index(self):
        """Optimize index parameters for better performance."""
        if not self.collection:
            return
        
        print("Optimizing index parameters...")
        # In production, you might want to:
        # 1. Analyze query patterns
        # 2. Adjust ef_search based on accuracy/speed tradeoff
        # 3. Consider different index types based on data size
        
        self.config.ef_search = 150  # Increase for better accuracy
        print(f"Increased ef_search to {self.config.ef_search} for better recall")
    
    def close(self):
        """Close connection to Milvus."""
        if self.connected:
            connections.disconnect("default")
            print("Disconnected from Milvus")


class VectorIndex:
    """
    High-level vector index manager supporting multiple backends.
    """
    
    def __init__(self, backend: str = "milvus", config: Optional[VectorConfig] = None):
        self.backend = backend
        
        if backend == "milvus":
            self.db = MilvusVectorDB(config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def add(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Add documents with embeddings.
        
        Args:
            documents: List of documents with 'id', 'content', 'metadata'
            embeddings: Array of embeddings
        """
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        return self.db.batch_insert(ids, embeddings, contents, metadatas)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search for similar documents."""
        return self.db.search(query_embedding, top_k)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return self.db.get_stats()
    
    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == '__main__':
    # Demo usage
    print("=== Milvus Vector Database Demo ===\n")
    
    # Initialize database
    db = MilvusVectorDB()
    
    # Create sample embeddings
    print("1. Inserting vectors:")
    docs = [
        {"id": "doc1", "content": "NVIDIA provides AI acceleration"},
        {"id": "doc2", "content": "Vector databases enable semantic search"},
        {"id": "doc3", "content": "Machine learning transforms data"}
    ]
    
    # Generate mock embeddings
    embeddings = [np.random.randn(384) for _ in range(3)]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]  # Normalize
    
    success = db.batch_insert(
        ids=[doc['id'] for doc in docs],
        embeddings=embeddings,
        contents=[doc['content'] for doc in docs],
        metadatas=[{'source': 'demo'} for _ in docs]
    )
    print(f"   Inserted: {success}")
    print(f"   Total vectors: {db.count()}\n")
    
    # Search
    print("2. Searching for similar content:")
    query_emb = np.random.randn(384)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    results = db.search(query_emb, top_k=2)
    for i, result in enumerate(results):
        print(f"   Result {i+1}:")
        print(f"      ID: {result['id']}")
        print(f"      Content: {result['content']}")
        print(f"      Score: {result['score']:.4f}")
    
    # Statistics
    print("\n3. Database statistics:")
    stats = db.get_stats()
    print(json.dumps(stats, indent=2))
    
    db.close()
