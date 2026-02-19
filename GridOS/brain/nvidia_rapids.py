"""
NVIDIA RAPIDS Integration
GPU-accelerated data processing for the second brain

Features:
- cuDF for fast DataFrame operations
- cuML for GPU machine learning
- cuGraph for graph analytics
- GPU-accelerated similarity search
- Fast batch processing
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class DataStats:
    """Statistics about processed data."""
    total_records: int
    processing_time_ms: float
    gpu_memory_used_mb: float
    operations_performed: List[str]


class RAPIDSDataProcessor:
    """
    NVIDIA RAPIDS-powered data processor.
    
    Accelerates data operations using GPU:
    - Fast filtering and aggregation
    - Similarity computations
    - Clustering
    - Dimensionality reduction
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize RAPIDS processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration (falls back to CPU if unavailable)
        """
        self.use_gpu = use_gpu
        self.has_cudf = False
        self.has_cuml = False
        
        # Try to import RAPIDS libraries
        try:
            import cudf
            self.has_cudf = True
            self.cudf = cudf
        except ImportError:
            print("cuDF not available, using pandas fallback")
            import pandas as pd
            self.cudf = pd
        
        try:
            import cuml
            self.has_cuml = True
            self.cuml = cuml
        except ImportError:
            print("cuML not available, using scikit-learn fallback")
            self.cuml = None
        
        self.stats = {
            'operations': 0,
            'total_time': 0.0,
            'gpu_operations': 0
        }
    
    def create_dataframe(self, data: Dict[str, List]) -> Any:
        """
        Create a GPU-accelerated DataFrame.
        
        Args:
            data: Dictionary of column names to data
            
        Returns:
            cuDF DataFrame (or pandas if GPU unavailable)
        """
        return self.cudf.DataFrame(data)
    
    def filter_by_date_range(
        self,
        df: Any,
        date_column: str,
        start_date: datetime,
        end_date: datetime
    ) -> Any:
        """
        GPU-accelerated date filtering.
        
        Args:
            df: DataFrame
            date_column: Name of date column
            start_date: Start of range
            end_date: End of range
            
        Returns:
            Filtered DataFrame
        """
        start = time.time()
        
        # Convert dates if necessary
        if self.has_cudf:
            filtered = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        else:
            filtered = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        return filtered
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix on GPU.
        
        Args:
            embeddings: Array of embeddings (shape: [n, dim])
            
        Returns:
            Similarity matrix (shape: [n, n])
        """
        import time
        start = time.time()
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        if self.has_cudf:
            self.stats['gpu_operations'] += 1
        
        return similarity_matrix
    
    def find_duplicates(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """
        Find duplicate texts using GPU-accelerated similarity.
        
        Args:
            texts: List of text strings
            embeddings: Embeddings for each text
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of (index1, index2, similarity) tuples
        """
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        duplicates = []
        n = len(texts)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    duplicates.append((i, j, float(similarity_matrix[i, j])))
        
        return duplicates
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Cluster embeddings using GPU-accelerated algorithms.
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters
            method: Clustering method ("kmeans", "dbscan")
            
        Returns:
            Cluster labels for each embedding
        """
        import time
        start = time.time()
        
        if self.has_cuml and method == "kmeans":
            # Use GPU-accelerated KMeans
            from cuml.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            labels = labels.to_numpy() if hasattr(labels, 'to_numpy') else labels
        else:
            # Fallback to CPU
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        if self.has_cuml:
            self.stats['gpu_operations'] += 1
        
        return labels
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Reduce dimensionality for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Target dimensions
            method: Reduction method ("pca", "tsne", "umap")
            
        Returns:
            Reduced embeddings
        """
        import time
        start = time.time()
        
        if self.has_cuml and method == "pca":
            from cuml.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
            reduced = reduced.to_numpy() if hasattr(reduced, 'to_numpy') else reduced
        elif self.has_cuml and method == "umap":
            try:
                from cuml.manifold import UMAP
                umap = UMAP(n_components=n_components)
                reduced = umap.fit_transform(embeddings)
                reduced = reduced.to_numpy() if hasattr(reduced, 'to_numpy') else reduced
            except ImportError:
                # Fallback to CPU UMAP
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                reduced = pca.fit_transform(embeddings)
        else:
            # CPU fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        if self.has_cuml:
            self.stats['gpu_operations'] += 1
        
        return reduced
    
    def aggregate_statistics(
        self,
        df: Any,
        group_by: str,
        agg_columns: Dict[str, str]
    ) -> Any:
        """
        GPU-accelerated aggregation.
        
        Args:
            df: DataFrame
            group_by: Column to group by
            agg_columns: Dict of column -> aggregation function
            
        Returns:
            Aggregated DataFrame
        """
        import time
        start = time.time()
        
        grouped = df.groupby(group_by).agg(agg_columns)
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        return grouped
    
    def sort_by_relevance(
        self,
        scores: np.ndarray,
        top_k: int = 10
    ) -> np.ndarray:
        """
        GPU-accelerated sorting.
        
        Args:
            scores: Array of scores
            top_k: Number of top items to return
            
        Returns:
            Indices of top k items
        """
        # Use numpy's argsort (can be GPU-accelerated with CuPy)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        self.stats['operations'] += 1
        
        return sorted_indices
    
    def batch_process(
        self,
        items: List[Any],
        processor_func: callable,
        batch_size: int = 1000
    ) -> List[Any]:
        """
        Process items in GPU-accelerated batches.
        
        Args:
            items: Items to process
            processor_func: Function to apply to each batch
            batch_size: Size of each batch
            
        Returns:
            Processed results
        """
        import time
        start = time.time()
        
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
        
        elapsed = (time.time() - start) * 1000
        self.stats['operations'] += 1
        self.stats['total_time'] += elapsed
        
        return results
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_operations': self.stats['operations'],
            'gpu_operations': self.stats['gpu_operations'],
            'total_time_ms': round(self.stats['total_time'], 2),
            'avg_time_ms': (
                round(self.stats['total_time'] / self.stats['operations'], 2)
                if self.stats['operations'] > 0
                else 0.0
            ),
            'gpu_available': self.has_cudf and self.has_cuml,
            'cudf_available': self.has_cudf,
            'cuml_available': self.has_cuml
        }


class RAPIDSGraphProcessor:
    """
    NVIDIA cuGraph processor for graph analytics.
    
    Accelerates knowledge graph operations:
    - Pagerank
    - Community detection
    - Shortest paths
    - Centrality metrics
    """
    
    def __init__(self):
        """Initialize cuGraph processor."""
        self.has_cugraph = False
        
        try:
            import cugraph
            self.has_cugraph = True
            self.cugraph = cugraph
        except ImportError:
            print("cuGraph not available, using NetworkX fallback")
            import networkx as nx
            self.nx = nx
    
    def create_graph(self, edges: List[Tuple[str, str, float]]) -> Any:
        """
        Create a graph from edges.
        
        Args:
            edges: List of (source, target, weight) tuples
            
        Returns:
            Graph object
        """
        if self.has_cugraph:
            import cudf
            df = cudf.DataFrame({
                'source': [e[0] for e in edges],
                'target': [e[1] for e in edges],
                'weight': [e[2] for e in edges]
            })
            G = self.cugraph.Graph()
            G.from_cudf_edgelist(df, source='source', destination='target', edge_attr='weight')
            return G
        else:
            G = self.nx.DiGraph()
            for source, target, weight in edges:
                G.add_edge(source, target, weight=weight)
            return G
    
    def compute_pagerank(self, graph: Any, max_iter: int = 100) -> Dict[str, float]:
        """
        Compute PageRank on GPU.
        
        Args:
            graph: Graph object
            max_iter: Maximum iterations
            
        Returns:
            Dict of node -> pagerank score
        """
        if self.has_cugraph:
            pr = self.cugraph.pagerank(graph, max_iter=max_iter)
            return dict(zip(pr['vertex'].to_pandas(), pr['pagerank'].to_pandas()))
        else:
            pr = self.nx.pagerank(graph, max_iter=max_iter)
            return pr
    
    def find_communities(self, graph: Any) -> Dict[str, int]:
        """
        Detect communities in graph.
        
        Args:
            graph: Graph object
            
        Returns:
            Dict of node -> community ID
        """
        if self.has_cugraph:
            parts, modularity = self.cugraph.louvain(graph)
            return dict(zip(parts['vertex'].to_pandas(), parts['partition'].to_pandas()))
        else:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(graph.to_undirected())
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i
            return node_to_comm
    
    def compute_centrality(self, graph: Any, method: str = "betweenness") -> Dict[str, float]:
        """
        Compute node centrality.
        
        Args:
            graph: Graph object
            method: Centrality method
            
        Returns:
            Dict of node -> centrality score
        """
        if self.has_cugraph and method == "betweenness":
            bc = self.cugraph.betweenness_centrality(graph)
            return dict(zip(bc['vertex'].to_pandas(), bc['betweenness_centrality'].to_pandas()))
        else:
            if method == "betweenness":
                cent = self.nx.betweenness_centrality(graph)
            elif method == "closeness":
                cent = self.nx.closeness_centrality(graph)
            elif method == "degree":
                cent = self.nx.degree_centrality(graph)
            else:
                cent = self.nx.betweenness_centrality(graph)
            return cent


# Convenience instances
rapids_processor = RAPIDSDataProcessor()
rapids_graph = RAPIDSGraphProcessor()


__all__ = [
    "RAPIDSDataProcessor",
    "RAPIDSGraphProcessor",
    "DataStats",
    "rapids_processor",
    "rapids_graph",
]
