"""
Similarity Analysis Caching System
=================================
Advanced caching system for similarity analysis results to avoid redundant
AI processing and reduce token costs for repeated analyses.

Features:
- Multi-level caching (memory, disk, database)
- Intelligent cache invalidation based on content changes
- Compression for large result sets
- Cache warming for common patterns
- Performance monitoring and optimization
- Thread-safe operations for concurrent access
"""

import os
import json
import pickle
import sqlite3
import hashlib
import logging
import gzip
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    content_hash: str
    expires_at: Optional[datetime] = None
    tags: List[str] = None

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_entries: int
    memory_hits: int
    disk_hits: int
    db_hits: int
    misses: int
    total_size_bytes: int
    hit_rate: float
    avg_access_time_ms: float

class SimilarityCacheManager:
    """
    Multi-level caching system for similarity analysis results.
    Provides efficient storage and retrieval with intelligent eviction policies.
    """
    
    def __init__(self, cache_dir: str = "similarity_cache", 
                 max_memory_entries: int = 1000,
                 max_disk_size_mb: int = 500,
                 default_ttl_hours: int = 24):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.max_disk_size_mb = max_disk_size_mb
        self.default_ttl_hours = default_ttl_hours
        
        # Multi-level cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = self.cache_dir / "disk"
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        # Database cache for persistence
        self.db_path = self.cache_dir / "similarity_cache.db"
        self._init_database()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'db_hits': 0,
            'misses': 0,
            'total_requests': 0,
            'cache_writes': 0,
            'evictions': 0,
            'avg_access_time': []
        }
        
        logger.info(f"SimilarityCacheManager initialized: {cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache with multi-level lookup.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached data if found, None otherwise
        """
        start_time = datetime.now()
        
        with self._lock:
            self.stats['total_requests'] += 1
            
            # Level 1: Memory cache (fastest)
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid_entry(entry):
                    entry.accessed_at = datetime.now()
                    entry.access_count += 1
                    self.stats['memory_hits'] += 1
                    self._record_access_time(start_time)
                    logger.debug(f"Memory cache hit: {key}")
                    return entry.data
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
            
            # Level 2: Disk cache (medium speed)
            disk_data = self._get_from_disk(key)
            if disk_data is not None:
                # Promote to memory cache
                self._put_memory(key, disk_data, access_count=1)
                self.stats['disk_hits'] += 1
                self._record_access_time(start_time)
                logger.debug(f"Disk cache hit: {key}")
                return disk_data
            
            # Level 3: Database cache (slower, persistent)
            db_data = self._get_from_database(key)
            if db_data is not None:
                # Promote to memory and disk cache
                self._put_memory(key, db_data, access_count=1)
                self._put_disk(key, db_data)
                self.stats['db_hits'] += 1
                self._record_access_time(start_time)
                logger.debug(f"Database cache hit: {key}")
                return db_data
            
            # Cache miss
            self.stats['misses'] += 1
            self._record_access_time(start_time)
            logger.debug(f"Cache miss: {key}")
            return None
    
    def put(self, key: str, data: Any, ttl_hours: Optional[int] = None, 
            tags: List[str] = None) -> bool:
        """
        Store item in cache with automatic tier management.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_hours: Time to live in hours (optional)
            tags: Tags for cache invalidation (optional)
            
        Returns:
            True if successfully cached
        """
        try:
            with self._lock:
                # Calculate expiration
                expires_at = None
                if ttl_hours or self.default_ttl_hours:
                    hours = ttl_hours or self.default_ttl_hours
                    expires_at = datetime.now() + timedelta(hours=hours)
                
                # Store in all cache levels
                self._put_memory(key, data, expires_at=expires_at, tags=tags)
                self._put_disk(key, data, expires_at=expires_at, tags=tags)
                self._put_database(key, data, expires_at=expires_at, tags=tags)
                
                self.stats['cache_writes'] += 1
                logger.debug(f"Cached item: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Error caching item {key}: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cached item across all levels.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if item was found and invalidated
        """
        invalidated = False
        
        with self._lock:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                invalidated = True
            
            # Remove from disk cache
            disk_file = self.disk_cache_dir / f"{self._safe_filename(key)}.gz"
            if disk_file.exists():
                disk_file.unlink()
                invalidated = True
            
            # Remove from database cache
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    if cursor.rowcount > 0:
                        invalidated = True
            except Exception as e:
                logger.error(f"Error invalidating database cache: {e}")
        
        if invalidated:
            logger.debug(f"Invalidated cache entry: {key}")
        
        return invalidated
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Invalidate all cached items with specified tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            Number of items invalidated
        """
        invalidated_count = 0
        
        with self._lock:
            # Invalidate from memory cache
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if entry.tags and any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated_count += 1
            
            # Invalidate from database (includes disk cache metadata)
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    placeholders = ', '.join('?' for _ in tags)
                    cursor.execute(f"""
                        SELECT key FROM cache_entries 
                        WHERE tags IS NOT NULL AND (
                            {' OR '.join(f'tags LIKE ?' for _ in tags)}
                        )
                    """, [f'%{tag}%' for tag in tags])
                    
                    keys_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    for key in keys_to_delete:
                        self.invalidate(key)
                        invalidated_count += 1
                        
            except Exception as e:
                logger.error(f"Error invalidating by tags: {e}")
        
        logger.info(f"Invalidated {invalidated_count} entries with tags: {tags}")
        return invalidated_count
    
    def clear_expired(self) -> int:
        """
        Remove expired entries from all cache levels.
        
        Returns:
            Number of expired entries removed
        """
        removed_count = 0
        current_time = datetime.now()
        
        with self._lock:
            # Clear expired from memory cache
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if not self._is_valid_entry(entry, current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                removed_count += 1
            
            # Clear expired from database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM cache_entries 
                        WHERE expires_at IS NOT NULL AND expires_at < ?
                    """, (current_time.isoformat(),))
                    removed_count += cursor.rowcount
            except Exception as e:
                logger.error(f"Error clearing expired entries: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} expired cache entries")
        
        return removed_count
    
    def optimize_cache(self) -> Dict[str, int]:
        """
        Optimize cache performance by cleaning up and reorganizing.
        
        Returns:
            Optimization statistics
        """
        stats = {
            'expired_removed': 0,
            'lru_evicted': 0,
            'disk_cleaned': 0,
            'db_vacuumed': 0
        }
        
        with self._lock:
            # Clear expired entries
            stats['expired_removed'] = self.clear_expired()
            
            # Evict LRU entries if memory cache is too large
            if len(self.memory_cache) > self.max_memory_entries:
                lru_count = len(self.memory_cache) - self.max_memory_entries
                lru_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].accessed_at
                )[:lru_count]
                
                for key in lru_keys:
                    del self.memory_cache[key]
                    stats['lru_evicted'] += 1
                    self.stats['evictions'] += 1
            
            # Clean up disk cache if too large
            disk_size_mb = self._calculate_disk_cache_size()
            if disk_size_mb > self.max_disk_size_mb:
                stats['disk_cleaned'] = self._cleanup_disk_cache()
            
            # Vacuum database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("VACUUM")
                    stats['db_vacuumed'] = 1
            except Exception as e:
                logger.error(f"Error vacuuming database: {e}")
        
        logger.info(f"Cache optimization complete: {stats}")
        return stats
    
    def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_requests = self.stats['total_requests']
            total_hits = (self.stats['memory_hits'] + 
                         self.stats['disk_hits'] + 
                         self.stats['db_hits'])
            
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            avg_access_time = (
                sum(self.stats['avg_access_time']) / len(self.stats['avg_access_time'])
                if self.stats['avg_access_time'] else 0
            )
            
            total_size = sum(
                len(pickle.dumps(entry.data)) 
                for entry in self.memory_cache.values()
            )
            
            return CacheStats(
                total_entries=len(self.memory_cache),
                memory_hits=self.stats['memory_hits'],
                disk_hits=self.stats['disk_hits'],
                db_hits=self.stats['db_hits'],
                misses=self.stats['misses'],
                total_size_bytes=total_size,
                hit_rate=hit_rate,
                avg_access_time_ms=avg_access_time
            )
    
    def warm_cache(self, common_queries: Dict[str, Any]) -> int:
        """
        Pre-populate cache with commonly used queries.
        
        Args:
            common_queries: Dictionary of common query results
            
        Returns:
            Number of entries warmed
        """
        warmed_count = 0
        
        for key, data in common_queries.items():
            if self.put(key, data, tags=['warmed']):
                warmed_count += 1
        
        logger.info(f"Warmed cache with {warmed_count} entries")
        return warmed_count
    
    def export_cache_data(self, filepath: str) -> bool:
        """
        Export cache data for backup or analysis.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'memory_cache': {},
                'cache_metadata': []
            }
            
            # Export memory cache
            for key, entry in self.memory_cache.items():
                export_data['memory_cache'][key] = {
                    'created_at': entry.created_at.isoformat(),
                    'accessed_at': entry.accessed_at.isoformat(),
                    'access_count': entry.access_count,
                    'tags': entry.tags,
                    'data_size': len(pickle.dumps(entry.data))
                }
            
            # Export database metadata
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT key, created_at, accessed_at, access_count, 
                               size_bytes, tags FROM cache_entries
                    """)
                    
                    for row in cursor.fetchall():
                        export_data['cache_metadata'].append({
                            'key': row[0],
                            'created_at': row[1],
                            'accessed_at': row[2],
                            'access_count': row[3],
                            'size_bytes': row[4],
                            'tags': row[5]
                        })
            except Exception as e:
                logger.error(f"Error exporting database metadata: {e}")
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Cache data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting cache data: {e}")
            return False
    
    def _init_database(self):
        """Initialize SQLite database for persistent cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create cache entries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        data BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        accessed_at TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        size_bytes INTEGER,
                        content_hash TEXT,
                        expires_at TEXT,
                        tags TEXT,
                        compressed BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON cache_entries(tags)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _put_memory(self, key: str, data: Any, expires_at: Optional[datetime] = None,
                   access_count: int = 0, tags: List[str] = None):
        """Store item in memory cache"""
        data_size = len(pickle.dumps(data))
        content_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
        
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=access_count,
            size_bytes=data_size,
            content_hash=content_hash,
            expires_at=expires_at,
            tags=tags or []
        )
        
        self.memory_cache[key] = entry
        
        # Evict if necessary
        if len(self.memory_cache) > self.max_memory_entries:
            self._evict_lru_memory()
    
    def _put_disk(self, key: str, data: Any, expires_at: Optional[datetime] = None,
                 tags: List[str] = None):
        """Store item in disk cache"""
        try:
            filename = self._safe_filename(key) + '.gz'
            filepath = self.disk_cache_dir / filename
            
            # Compress and store
            serialized = pickle.dumps(data)
            with gzip.open(filepath, 'wb') as f:
                f.write(serialized)
                
        except Exception as e:
            logger.error(f"Error storing to disk cache: {e}")
    
    def _put_database(self, key: str, data: Any, expires_at: Optional[datetime] = None,
                     tags: List[str] = None):
        """Store item in database cache"""
        try:
            # Compress large data
            serialized = pickle.dumps(data)
            compressed = len(serialized) > 1024  # Compress if > 1KB
            
            if compressed:
                serialized = gzip.compress(serialized)
            
            content_hash = hashlib.md5(serialized).hexdigest()
            now = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, data, created_at, accessed_at, access_count,
                     size_bytes, content_hash, expires_at, tags, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, serialized, now.isoformat(), now.isoformat(), 0,
                    len(serialized), content_hash,
                    expires_at.isoformat() if expires_at else None,
                    json.dumps(tags) if tags else None, compressed
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing to database cache: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve item from disk cache"""
        try:
            filename = self._safe_filename(key) + '.gz'
            filepath = self.disk_cache_dir / filename
            
            if filepath.exists():
                with gzip.open(filepath, 'rb') as f:
                    return pickle.loads(f.read())
                    
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
        
        return None
    
    def _get_from_database(self, key: str) -> Optional[Any]:
        """Retrieve item from database cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data, expires_at, compressed FROM cache_entries 
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if row:
                    data_blob, expires_at, compressed = row
                    
                    # Check expiration
                    if expires_at:
                        expire_time = datetime.fromisoformat(expires_at)
                        if datetime.now() > expire_time:
                            return None
                    
                    # Decompress if needed
                    if compressed:
                        data_blob = gzip.decompress(data_blob)
                    
                    # Update access statistics
                    cursor.execute("""
                        UPDATE cache_entries 
                        SET accessed_at = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (datetime.now().isoformat(), key))
                    
                    return pickle.loads(data_blob)
                    
        except Exception as e:
            logger.error(f"Error reading from database cache: {e}")
        
        return None
    
    def _is_valid_entry(self, entry: CacheEntry, current_time: Optional[datetime] = None) -> bool:
        """Check if cache entry is still valid"""
        if current_time is None:
            current_time = datetime.now()
        
        if entry.expires_at and current_time > entry.expires_at:
            return False
        
        return True
    
    def _evict_lru_memory(self):
        """Evict least recently used entry from memory cache"""
        if not self.memory_cache:
            return
        
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k].accessed_at
        )
        
        del self.memory_cache[lru_key]
        self.stats['evictions'] += 1
    
    def _safe_filename(self, key: str) -> str:
        """Convert cache key to safe filename"""
        # Use hash for long or unsafe keys
        if len(key) > 200 or any(c in key for c in '<>:"/\\|?*'):
            return hashlib.md5(key.encode()).hexdigest()
        return key.replace('/', '_').replace('\\', '_')
    
    def _calculate_disk_cache_size(self) -> float:
        """Calculate total disk cache size in MB"""
        total_size = 0
        try:
            for file_path in self.disk_cache_dir.glob('*.gz'):
                total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating disk cache size: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _cleanup_disk_cache(self) -> int:
        """Clean up old disk cache files"""
        removed_count = 0
        try:
            # Get all cache files with their access times
            cache_files = []
            for file_path in self.disk_cache_dir.glob('*.gz'):
                stat = file_path.stat()
                cache_files.append((file_path, stat.st_atime))
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under size limit
            current_size_mb = self._calculate_disk_cache_size()
            target_size_mb = self.max_disk_size_mb * 0.8  # Remove to 80% of limit
            
            for file_path, _ in cache_files:
                if current_size_mb <= target_size_mb:
                    break
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink()
                current_size_mb -= file_size_mb
                removed_count += 1
                
        except Exception as e:
            logger.error(f"Error cleaning up disk cache: {e}")
        
        return removed_count
    
    def _record_access_time(self, start_time: datetime):
        """Record access time for performance monitoring"""
        access_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.stats['avg_access_time'].append(access_time_ms)
        
        # Keep only last 1000 access times
        if len(self.stats['avg_access_time']) > 1000:
            self.stats['avg_access_time'] = self.stats['avg_access_time'][-1000:]

# Usage example and testing
if __name__ == "__main__":
    # Test the caching system
    cache_manager = SimilarityCacheManager()
    
    # Test data
    test_data = {
        'query_similarity_1': {'similarity': 0.85, 'details': 'High similarity between queries'},
        'query_similarity_2': {'similarity': 0.42, 'details': 'Medium similarity'},
        'embedding_results': [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # Larger data for compression test
    }
    
    print("Testing Similarity Cache Manager:")
    print("=" * 40)
    
    # Test cache storage
    for key, data in test_data.items():
        success = cache_manager.put(key, data, ttl_hours=1, tags=['test'])
        print(f"Cached '{key}': {success}")
    
    # Test cache retrieval
    print("\nTesting cache retrieval:")
    for key in test_data.keys():
        cached_data = cache_manager.get(key)
        hit = cached_data is not None
        print(f"Cache {'HIT' if hit else 'MISS'}: {key}")
    
    # Test cache miss
    missing_data = cache_manager.get('non_existent_key')
    print(f"Cache {'HIT' if missing_data else 'MISS'}: non_existent_key")
    
    # Show statistics
    stats = cache_manager.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit Rate: {stats.hit_rate:.1f}%")
    print(f"  Total Entries: {stats.total_entries}")
    print(f"  Memory Hits: {stats.memory_hits}")
    print(f"  Disk Hits: {stats.disk_hits}")
    print(f"  Database Hits: {stats.db_hits}")
    print(f"  Misses: {stats.misses}")
    
    # Test tag-based invalidation
    invalidated = cache_manager.invalidate_by_tags(['test'])
    print(f"\nInvalidated {invalidated} entries with 'test' tag")
    
    # Test optimization
    optimization_stats = cache_manager.optimize_cache()
    print(f"\nOptimization results: {optimization_stats}")
    
    # Export cache data
    export_path = "cache_export.json"
    if cache_manager.export_cache_data(export_path):
        print(f"\nCache data exported to {export_path}")