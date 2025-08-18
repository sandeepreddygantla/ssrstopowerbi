"""
Embedding Manager for Query Similarity
=====================================
Efficient embedding generation and caching system for semantic similarity analysis.
Provides cost-effective pre-filtering before expensive LLM analysis.

Features:
- Multiple embedding providers (OpenAI, HuggingFace, local models)
- Intelligent caching to avoid redundant API calls
- Batch processing for optimal API usage
- Semantic similarity calculation with cosine distance
- Token cost tracking and optimization
"""

import os
import json
import hashlib
import logging
import pickle
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Optional imports - will fall back to basic implementations if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available, using fallback embedding")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-transformers not available, using fallback")

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text_hash: str
    embedding: List[float]
    provider: str
    model: str
    token_count: int
    cost_estimate: float
    created_at: datetime

@dataclass
class SimilarityMatch:
    """Similarity match result"""
    query1_id: str
    query2_id: str
    similarity_score: float
    embedding_method: str
    computation_time: float

class EmbeddingManager:
    """
    Manages embedding generation, caching, and similarity calculations
    with cost optimization for large-scale query analysis.
    """
    
    def __init__(self, cache_dir: str = "embeddings_cache", provider: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache database
        self.cache_db_path = self.cache_dir / "embeddings.db"
        self._init_cache_database()
        
        # Initialize embedding provider
        self.provider = self._select_provider(provider)
        self.model = None
        self._init_embedding_provider()
        
        # Statistics tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_generated': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0
        }
        
        logger.info(f"EmbeddingManager initialized with provider: {self.provider}")
    
    def generate_embeddings(self, texts: Dict[str, str], batch_size: int = 50) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings for multiple texts with intelligent caching.
        
        Args:
            texts: Dictionary mapping text_id to text content
            batch_size: Number of texts to process in each batch
            
        Returns:
            Dictionary mapping text_id to EmbeddingResult
        """
        results = {}
        cache_misses = {}
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache first
        for text_id, text in texts.items():
            text_hash = self._get_text_hash(text)
            cached_result = self._get_cached_embedding(text_hash)
            
            if cached_result:
                results[text_id] = cached_result
                self.stats['cache_hits'] += 1
            else:
                cache_misses[text_id] = text
                self.stats['cache_misses'] += 1
        
        logger.info(f"Cache hits: {self.stats['cache_hits']}, Cache misses: {len(cache_misses)}")
        
        # Generate embeddings for cache misses
        if cache_misses:
            new_embeddings = self._generate_batch_embeddings(cache_misses, batch_size)
            results.update(new_embeddings)
            
            # Cache new results
            for text_id, embedding_result in new_embeddings.items():
                self._cache_embedding(embedding_result)
        
        return results
    
    def calculate_similarity_matrix(self, embeddings: Dict[str, EmbeddingResult], 
                                  threshold: float = 0.5) -> List[SimilarityMatch]:
        """
        Calculate pairwise similarity matrix for embeddings.
        
        Args:
            embeddings: Dictionary of embeddings from generate_embeddings
            threshold: Minimum similarity threshold to include in results
            
        Returns:
            List of SimilarityMatch objects above threshold
        """
        start_time = datetime.now()
        
        ids = list(embeddings.keys())
        similarities = []
        
        logger.info(f"Calculating similarity matrix for {len(ids)} embeddings")
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                emb1 = embeddings[id1].embedding
                emb2 = embeddings[id2].embedding
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)
                
                if similarity >= threshold:
                    match = SimilarityMatch(
                        query1_id=id1,
                        query2_id=id2,
                        similarity_score=similarity,
                        embedding_method=embeddings[id1].provider,
                        computation_time=(datetime.now() - start_time).total_seconds()
                    )
                    similarities.append(match)
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Found {len(similarities)} similar pairs above threshold {threshold} in {computation_time:.2f}s")
        
        return similarities
    
    def find_most_similar(self, query_embedding: EmbeddingResult, 
                         candidate_embeddings: Dict[str, EmbeddingResult], 
                         top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Target embedding to find matches for
            candidate_embeddings: Pool of candidate embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        similarities = []
        query_emb = query_embedding.embedding
        
        for candidate_id, candidate_result in candidate_embeddings.items():
            candidate_emb = candidate_result.embedding
            similarity = self._cosine_similarity(query_emb, candidate_emb)
            similarities.append((candidate_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def optimize_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Optimize cache by removing old entries and duplicate embeddings.
        
        Args:
            max_age_days: Maximum age of cached embeddings to keep
            
        Returns:
            Statistics about cache optimization
        """
        logger.info("Optimizing embedding cache...")
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            
            # Count total entries before optimization
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_before = cursor.fetchone()[0]
            
            # Remove old entries
            cursor.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff_date.isoformat(),))
            old_removed = cursor.rowcount
            
            # Count entries after cleanup
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_after = cursor.fetchone()[0]
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
        
        stats = {
            'entries_before': total_before,
            'old_entries_removed': old_removed,
            'entries_after': total_after,
            'space_saved_percent': (old_removed / total_before * 100) if total_before > 0 else 0
        }
        
        logger.info(f"Cache optimization complete: {stats}")
        return stats
    
    def _select_provider(self, provider: str) -> str:
        """Select the best available embedding provider"""
        if provider == "auto":
            if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
                return "openai"
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                return "sentence_transformers"
            else:
                return "fallback"
        return provider
    
    def _init_embedding_provider(self):
        """Initialize the embedding provider"""
        try:
            if self.provider == "openai":
                openai.api_key = os.getenv('OPENAI_API_KEY')
                if not openai.api_key:
                    logger.warning("OpenAI API key not found, falling back to local models")
                    self.provider = "sentence_transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "fallback"
            
            if self.provider == "sentence_transformers":
                # Use a lightweight but effective model for SQL queries
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
            
            elif self.provider == "fallback":
                logger.warning("Using fallback embedding provider (feature-based)")
        
        except Exception as e:
            logger.error(f"Error initializing embedding provider: {e}")
            self.provider = "fallback"
    
    def _init_cache_database(self):
        """Initialize SQLite cache database"""
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    token_count INTEGER,
                    cost_estimate REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)")
            conn.commit()
    
    def _generate_batch_embeddings(self, texts: Dict[str, str], 
                                  batch_size: int) -> Dict[str, EmbeddingResult]:
        """Generate embeddings in batches for efficiency"""
        results = {}
        text_items = list(texts.items())
        
        for i in range(0, len(text_items), batch_size):
            batch_items = text_items[i:i + batch_size]
            batch_dict = dict(batch_items)
            
            try:
                batch_results = self._generate_embeddings_batch(batch_dict)
                results.update(batch_results)
                self.stats['api_calls'] += 1
                
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Fall back to individual processing
                for text_id, text in batch_dict.items():
                    try:
                        result = self._generate_single_embedding(text_id, text)
                        results[text_id] = result
                    except Exception as e2:
                        logger.error(f"Error generating embedding for {text_id}: {e2}")
        
        return results
    
    def _generate_embeddings_batch(self, texts: Dict[str, str]) -> Dict[str, EmbeddingResult]:
        """Generate embeddings for a batch of texts"""
        results = {}
        
        if self.provider == "openai":
            results = self._generate_openai_embeddings(texts)
        elif self.provider == "sentence_transformers":
            results = self._generate_sentence_transformer_embeddings(texts)
        else:
            results = self._generate_fallback_embeddings(texts)
        
        return results
    
    def _generate_openai_embeddings(self, texts: Dict[str, str]) -> Dict[str, EmbeddingResult]:
        """Generate embeddings using OpenAI API"""
        results = {}
        text_list = list(texts.values())
        text_ids = list(texts.keys())
        
        try:
            response = openai.Embedding.create(
                input=text_list,
                model="text-embedding-ada-002"
            )
            
            for i, (text_id, text) in enumerate(texts.items()):
                embedding_data = response['data'][i]
                token_count = response['usage']['total_tokens']
                cost_estimate = token_count * 0.0004 / 1000  # OpenAI pricing
                
                result = EmbeddingResult(
                    text_hash=self._get_text_hash(text),
                    embedding=embedding_data['embedding'],
                    provider="openai",
                    model="text-embedding-ada-002",
                    token_count=token_count,
                    cost_estimate=cost_estimate,
                    created_at=datetime.now()
                )
                
                results[text_id] = result
                self.stats['total_tokens'] += token_count
                self.stats['total_cost'] += cost_estimate
        
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Fallback to local model
            return self._generate_fallback_embeddings(texts)
        
        return results
    
    def _generate_sentence_transformer_embeddings(self, texts: Dict[str, str]) -> Dict[str, EmbeddingResult]:
        """Generate embeddings using sentence-transformers"""
        results = {}
        
        try:
            text_list = list(texts.values())
            embeddings = self.model.encode(text_list, convert_to_numpy=True)
            
            for i, (text_id, text) in enumerate(texts.items()):
                embedding = embeddings[i].tolist()
                
                result = EmbeddingResult(
                    text_hash=self._get_text_hash(text),
                    embedding=embedding,
                    provider="sentence_transformers",
                    model="all-MiniLM-L6-v2",
                    token_count=len(text.split()),  # Approximate
                    cost_estimate=0.0,  # Free local model
                    created_at=datetime.now()
                )
                
                results[text_id] = result
        
        except Exception as e:
            logger.error(f"Sentence-transformers embedding error: {e}")
            return self._generate_fallback_embeddings(texts)
        
        return results
    
    def _generate_fallback_embeddings(self, texts: Dict[str, str]) -> Dict[str, EmbeddingResult]:
        """Generate simple feature-based embeddings as fallback"""
        results = {}
        
        for text_id, text in texts.items():
            # Create a simple feature vector based on text characteristics
            embedding = self._create_feature_embedding(text)
            
            result = EmbeddingResult(
                text_hash=self._get_text_hash(text),
                embedding=embedding,
                provider="fallback",
                model="feature_based",
                token_count=len(text.split()),
                cost_estimate=0.0,
                created_at=datetime.now()
            )
            
            results[text_id] = result
        
        return results
    
    def _create_feature_embedding(self, text: str) -> List[float]:
        """Create a feature-based embedding for fallback"""
        # Normalize text
        text_lower = text.lower()
        
        # Create 128-dimensional feature vector
        features = [0.0] * 128
        
        # SQL keywords features
        sql_keywords = [
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
            'group by', 'order by', 'having', 'union', 'insert', 'update', 'delete',
            'create', 'alter', 'drop', 'distinct', 'count', 'sum', 'avg', 'max', 'min',
            'case', 'when', 'then', 'else', 'end', 'in', 'exists', 'between', 'like'
        ]
        
        for i, keyword in enumerate(sql_keywords[:64]):  # First 64 features for keywords
            if keyword in text_lower:
                features[i] = 1.0
        
        # Text statistics features
        features[64] = min(1.0, len(text) / 1000.0)  # Normalized length
        features[65] = min(1.0, len(text.split()) / 100.0)  # Normalized word count
        features[66] = text_lower.count('(') / max(1, len(text))  # Parentheses density
        features[67] = text_lower.count(',') / max(1, len(text))  # Comma density
        features[68] = text_lower.count('=') / max(1, len(text))  # Equals density
        
        # Table/column patterns
        table_patterns = ['customers', 'orders', 'products', 'sales', 'employees', 'items']
        for i, pattern in enumerate(table_patterns):
            if pattern in text_lower:
                features[70 + i] = 1.0
        
        # Function patterns
        func_patterns = ['sum(', 'count(', 'avg(', 'max(', 'min(', 'cast(', 'convert(']
        for i, pattern in enumerate(func_patterns):
            if pattern in text_lower:
                features[80 + i] = 1.0
        
        # Normalize vector
        vector_norm = sum(f * f for f in features) ** 0.5
        if vector_norm > 0:
            features = [f / vector_norm for f in features]
        
        return features
    
    def _generate_single_embedding(self, text_id: str, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        return self._generate_embeddings_batch({text_id: text})[text_id]
    
    def _get_cached_embedding(self, text_hash: str) -> Optional[EmbeddingResult]:
        """Retrieve embedding from cache"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM embeddings WHERE text_hash = ?", (text_hash,))
                row = cursor.fetchone()
                
                if row:
                    text_hash, embedding_blob, provider, model, token_count, cost_estimate, created_at = row
                    embedding = pickle.loads(embedding_blob)
                    
                    return EmbeddingResult(
                        text_hash=text_hash,
                        embedding=embedding,
                        provider=provider,
                        model=model,
                        token_count=token_count or 0,
                        cost_estimate=cost_estimate or 0.0,
                        created_at=datetime.fromisoformat(created_at)
                    )
        except Exception as e:
            logger.error(f"Error retrieving cached embedding: {e}")
        
        return None
    
    def _cache_embedding(self, result: EmbeddingResult):
        """Cache embedding result"""
        try:
            embedding_blob = pickle.dumps(result.embedding)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (text_hash, embedding, provider, model, token_count, cost_estimate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.text_hash,
                    embedding_blob,
                    result.provider,
                    result.model,
                    result.token_count,
                    result.cost_estimate,
                    result.created_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        # Normalize text for consistent hashing
        normalized = ' '.join(text.split()).strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics"""
        cache_hit_rate = (
            self.stats['cache_hits'] / 
            max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        )
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'provider': self.provider,
            'model': getattr(self.model, '_modules', {}).get('0', 'N/A') if self.model else 'N/A'
        }
    
    def export_embeddings(self, filepath: str, format: str = 'json'):
        """Export cached embeddings to file"""
        embeddings_data = []
        
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM embeddings")
            
            for row in cursor.fetchall():
                text_hash, embedding_blob, provider, model, token_count, cost_estimate, created_at = row
                embedding = pickle.loads(embedding_blob)
                
                embeddings_data.append({
                    'text_hash': text_hash,
                    'embedding': embedding,
                    'provider': provider,
                    'model': model,
                    'token_count': token_count,
                    'cost_estimate': cost_estimate,
                    'created_at': created_at
                })
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings_data, f)
        
        logger.info(f"Exported {len(embeddings_data)} embeddings to {filepath}")

# Usage example and testing
if __name__ == "__main__":
    # Test with sample SQL queries
    sample_queries = {
        'query1': 'SELECT CustomerID, OrderDate, SUM(Amount) FROM Orders WHERE OrderDate >= @StartDate GROUP BY CustomerID, OrderDate',
        'query2': 'SELECT o.CustomerID, o.OrderDate, SUM(o.TotalAmount) FROM Orders o WHERE o.OrderDate >= @StartDate GROUP BY o.CustomerID, o.OrderDate',
        'query3': 'SELECT ProductID, ProductName, CategoryID FROM Products WHERE Active = 1',
        'query4': 'SELECT p.ProductID, p.Name, p.Category FROM Products p WHERE p.IsActive = 1',
        'query5': 'SELECT COUNT(*) FROM Employees WHERE DepartmentID = @DeptID'
    }
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_manager.generate_embeddings(sample_queries)
    
    # Calculate similarity matrix
    print("\nCalculating similarity matrix...")
    similarities = embedding_manager.calculate_similarity_matrix(embeddings, threshold=0.3)
    
    # Print results
    print(f"\nFound {len(similarities)} similar pairs:")
    for match in similarities:
        print(f"  {match.query1_id} ↔ {match.query2_id}: {match.similarity_score:.3f}")
    
    # Print statistics
    stats = embedding_manager.get_stats()
    print(f"\nEmbedding Manager Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")