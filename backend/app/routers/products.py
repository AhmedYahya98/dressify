from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import random
import pandas as pd

from ..utils.faiss_manager import faiss_manager
from ..core.config import config

router = APIRouter(
    prefix="/api/products",
    tags=["products"]
)

@router.get("/featured")
async def get_featured_products(limit: int = 20):
    """
    Get random featured products from the catalog.
    Uses the metadata loaded in FAISS manager.
    """
    if not faiss_manager.is_loaded or faiss_manager.metadata_df is None:
        # Try to ensure index is loaded
        if not faiss_manager.load_from_disk():
             raise HTTPException(status_code=503, detail="Product catalog not available")
    
    df = faiss_manager.metadata_df
    total_items = len(df)
    
    if total_items == 0:
        return []
    
    # Get random sample
    # Use config constants if available for better control, but defaults are fine here
    sample_size = min(limit, total_items)
    # Simple random sampling
    indices = random.sample(range(total_items), sample_size)
    
    products = []
    for idx in indices:
        try:
             products.append(faiss_manager.get_metadata(idx))
        except:
             continue
             
    return products

@router.get("/by-category")
async def get_products_by_category(items_per_category: int = 8):
    """
    Get products grouped by master category.
    Returns a dict where keys are category names and values are product lists.
    """
    if not faiss_manager.is_loaded or faiss_manager.metadata_df is None:
        if not faiss_manager.load_from_disk():
            raise HTTPException(status_code=503, detail="Product catalog not available")
    
    df = faiss_manager.metadata_df
    
    # Get unique article_types as categories (more granular than master_category)
    categories = df['article_type'].dropna().unique().tolist()
    
    result = {}
    for cat in categories[:10]:  # Limit to 10 categories for performance
        cat_items = df[df['article_type'] == cat].head(items_per_category)
        if len(cat_items) > 0:
            result[cat] = [row.to_dict() for _, row in cat_items.iterrows()]
    
    return result

@router.get("/{product_id}")
async def get_product_details(product_id: str):
    """
    Get full details for a specific product by its image_id (string ID).
    Example: 1533
    """
    if not faiss_manager.is_loaded or faiss_manager.metadata_df is None:
         if not faiss_manager.load_from_disk():
             raise HTTPException(status_code=503, detail="Product catalog not available")
             
    df = faiss_manager.metadata_df
    
    # Filter by image_id (which is the main ID used in file names)
    # Ensure type compatibility (df id might be int or str)
    try:
        # Search efficiently
        # In a real DB this would be an index lookup. Here we do a pandas scan.
        # Check if column is string or int
        if pd.api.types.is_integer_dtype(df['image_id']):
             product = df[df['image_id'] == int(product_id)]
        else:
             product = df[df['image_id'] == str(product_id)]
             
        if product.empty:
            raise HTTPException(status_code=404, detail="Product not found")
            
        # Return first match as dict
        return product.iloc[0].to_dict()
        
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid product ID")
    except Exception as e:
        print(f"Error fetching product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
