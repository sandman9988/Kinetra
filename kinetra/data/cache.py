"""
Cache Management
================

Feature caching and deduplication for performance.
Extracted from data_management.py.

Features:
- Cache computed features by checksum
- Automatic cache invalidation
- LRU-style cache management
- Thread-safe operations
"""

import hashlib
import json
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """Cache entry for computed features."""
    key: str
    value_path: str
    checksum: str
    created_at: str
    last_accessed: str
    size_bytes: int
    hit_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class CacheManager:
    """
    Manages feature caching with automatic invalidation.
    
    Features:
    - Content-based caching (checksum keys)
    - Automatic expiration
    - Cache statistics
    - Thread-safe
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / 'cache_index.json'
        self.lock = threading.Lock()
        
        # Load existing index
        self.index: Dict[str, CacheEntry] = {}
        self._load_index()
        
    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.index = {
                        k: CacheEntry(**v) for k, v in data.items()
                    }
            except Exception:
                # Corrupted index - start fresh
                self.index = {}
                
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.index.items()},
                f,
                indent=2
            )
            
    def get(self, key: str) -> Optional[Path]:
        """
        Get cached value path.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cached file, or None if not found
        """
        with self.lock:
            if key in self.index:
                entry = self.index[key]
                entry.hit_count += 1
                entry.last_accessed = datetime.now().isoformat()
                self._save_index()
                
                path = Path(entry.value_path)
                if path.exists():
                    return path
                    
        return None
        
    def put(self, key: str, value_path: Path, checksum: str) -> None:
        """
        Add entry to cache.
        
        Args:
            key: Cache key
            value_path: Path to cached file
            checksum: Data checksum
        """
        with self.lock:
            entry = CacheEntry(
                key=key,
                value_path=str(value_path),
                checksum=checksum,
                created_at=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                size_bytes=value_path.stat().st_size if value_path.exists() else 0,
                hit_count=0
            )
            self.index[key] = entry
            self._save_index()
            
    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        with self.lock:
            if key in self.index:
                entry = self.index[key]
                # Delete cached file
                path = Path(entry.value_path)
                if path.exists():
                    path.unlink()
                # Remove from index
                del self.index[key]
                self._save_index()
                
    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            # Delete all cached files
            for entry in self.index.values():
                path = Path(entry.value_path)
                if path.exists():
                    path.unlink()
            # Clear index
            self.index = {}
            self._save_index()
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(e.size_bytes for e in self.index.values())
            total_hits = sum(e.hit_count for e in self.index.values())
            
            return {
                'entries': len(self.index),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'total_hits': total_hits,
                'avg_hits_per_entry': total_hits / len(self.index) if self.index else 0
            }
