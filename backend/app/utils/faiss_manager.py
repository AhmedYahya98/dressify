"""
FAISS Index Manager with disk persistence.
Builds, saves, and loads the vector database for fashion product search.
"""

import os
import pickle
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import faiss
from PIL import Image

from ..core.config import config
from .embeddings import get_image_embedding


class FAISSManager:
    """
    Manages the FAISS index with persistence support.
    Saves index to disk to avoid rebuilding on every startup.
    """
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded and ready."""
        return self._is_loaded and self.index is not None and self.metadata_df is not None
    
    @property
    def size(self) -> int:
        """Get number of items in the index."""
        if self.index is not None:
            return self.index.ntotal
        return 0
    
    def load_from_disk(self) -> bool:
        """
        Load FAISS index and metadata from disk if they exist.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(config.FAISS_INDEX_FILE) or not os.path.exists(config.METADATA_FILE):
            print("📂 No saved FAISS index found on disk")
            return False
        
        try:
            print(f"\n📂 Loading FAISS index from {config.FAISS_INDEX_FILE}...")
            self.index = faiss.read_index(config.FAISS_INDEX_FILE)
            
            print(f"📂 Loading metadata from {config.METADATA_FILE}...")
            with open(config.METADATA_FILE, 'rb') as f:
                self.metadata_df = pickle.load(f)
            
            self._is_loaded = True
            print(f"✅ Loaded FAISS index: {self.index.ntotal} items")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            self.index = None
            self.metadata_df = None
            self._is_loaded = False
            return False
    
    def save_to_disk(self) -> bool:
        """
        Save FAISS index and metadata to disk.
        
        Returns:
            True if successfully saved, False otherwise
        """
        if not self.is_loaded:
            print("❌ Cannot save: Index not loaded")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
            
            print(f"\n💾 Saving FAISS index to {config.FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, config.FAISS_INDEX_FILE)
            
            print(f"💾 Saving metadata to {config.METADATA_FILE}...")
            with open(config.METADATA_FILE, 'wb') as f:
                pickle.dump(self.metadata_df, f)
            
            print(f"✅ Saved FAISS index: {self.index.ntotal} items")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save FAISS index: {e}")
            return False
    
    def build_index(self, df: pd.DataFrame) -> bool:
        """
        Build FAISS index from dataset.
        Ported from build_faiss_index function in FP.ipynb.
        
        Args:
            df: DataFrame with 'image_path' column and product metadata
            
        Returns:
            True if successfully built, False otherwise
        """
        print(f"\n🔨 Building FAISS index from {len(df)} items...")
        embeddings = []
        metadata = []
        
        max_items = min(len(df), config.FAISS_MAX_ITEMS)
        
        for idx, row in df.head(max_items).iterrows():
            try:
                image_path = row.get('image_path')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                img = Image.open(image_path).convert('RGB')
                emb = get_image_embedding(img)
                embeddings.append(emb)
                
                metadata.append({
                    'id': len(metadata),
                    'image_id': row.get('id', idx),
                    'title': f"{row.get('articleType', 'Item')} - {row.get('baseColour', 'Color')}",
                    'brand': row.get('brandName', 'Brand'),
                    'price': str(row.get('price', 'N/A')),
                    'thumbnail_url': image_path,
                    'source_path': image_path,
                    'snippet': f"{row.get('gender', '')} {row.get('articleType', '')} in {row.get('baseColour', '')}".strip(),
                    'gender': row.get('gender', 'N/A'),
                    'article_type': row.get('articleType', 'N/A'),
                    'color': row.get('baseColour', 'N/A')
                })
                
                if len(embeddings) % config.INDEX_BATCH_SIZE == 0:
                    print(f"  ✓ {len(embeddings)} items processed")
                    
            except Exception as e:
                continue
        
        if not embeddings:
            print("❌ No embeddings generated")
            return False
        
        # Build FAISS index
        emb_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(emb_array.shape[1])  # Inner product for cosine similarity
        self.index.add(emb_array)
        self.metadata_df = pd.DataFrame(metadata)
        self._is_loaded = True
        
        print(f"✅ Built FAISS index: {self.index.ntotal} items, {emb_array.shape[1]}D")
        return True
    
    def search(self, embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for similar items.
        
        Args:
            embedding: Query embedding (normalized)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if not self.is_loaded:
            raise RuntimeError("FAISS index not loaded")
        
        emb_normalized = embedding / np.linalg.norm(embedding)
        emb_arr = np.array([emb_normalized]).astype('float32')
        
        distances, indices = self.index.search(emb_arr, k)
        return distances[0], indices[0]
    
    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a given index."""
        if not self.is_loaded or self.metadata_df is None:
            raise RuntimeError("Metadata not loaded")
        
        if idx < 0 or idx >= len(self.metadata_df):
            raise IndexError(f"Index {idx} out of range")
        
        return self.metadata_df.iloc[idx].to_dict()


# Global FAISS manager instance
faiss_manager = FAISSManager()
