"""
CLIP Embedding utilities for Fashion AI Chatbot.
Ported from FP.ipynb - get_image_embedding and get_text_embedding functions.
"""

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Optional

from ..core.config import config

# Global model instances (loaded once)
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None


def load_clip_model() -> tuple:
    """Load Fashion-CLIP model and processor."""
    global _clip_model, _clip_processor
    
    if _clip_model is None or _clip_processor is None:
        print(f"\n🎨 Loading Fashion-CLIP from {config.CLIP_MODEL}...")
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(config.DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        print("✅ Fashion-CLIP loaded")
    
    return _clip_model, _clip_processor


def get_clip_model() -> CLIPModel:
    """Get the loaded CLIP model instance."""
    global _clip_model
    if _clip_model is None:
        load_clip_model()
    return _clip_model


def get_clip_processor() -> CLIPProcessor:
    """Get the loaded CLIP processor instance."""
    global _clip_processor
    if _clip_processor is None:
        load_clip_model()
    return _clip_processor


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Get CLIP embedding for an image.
    Ported exactly from FP.ipynb.
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        Normalized embedding as float32 numpy array
    """
    clip_model = get_clip_model()
    clip_processor = get_clip_processor()
    
    inputs = clip_processor(images=image, return_tensors="pt").to(config.DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        embedding = features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype('float32')


def get_text_embedding(text: str) -> np.ndarray:
    """
    Get CLIP embedding for text.
    Ported exactly from FP.ipynb.
    
    Args:
        text: Text query string
        
    Returns:
        Normalized embedding as float32 numpy array
    """
    clip_model = get_clip_model()
    clip_processor = get_clip_processor()
    
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(config.DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        embedding = features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype('float32')


def validate_image_fashion(image: Image.Image, fashion_categories: list, non_fashion_categories: list) -> dict:
    """
    Validate if an image is fashion-related using CLIP.
    Uses the same logic as image_fashion_validator_agent.
    
    Args:
        image: PIL Image
        fashion_categories: List of fashion category strings
        non_fashion_categories: List of non-fashion category strings
        
    Returns:
        Dict with is_fashion, scores, and top predictions
    """
    clip_model = get_clip_model()
    clip_processor = get_clip_processor()
    
    inputs = clip_processor(images=image, return_tensors="pt").to(config.DEVICE)
    all_cats = fashion_categories + non_fashion_categories
    text_inputs = clip_processor(text=all_cats, return_tensors="pt", padding=True).to(config.DEVICE)
    
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
        txt_feat = clip_model.get_text_features(**text_inputs)
        sim = (img_feat @ txt_feat.T).softmax(dim=-1)
        top_idx = sim[0].topk(10).indices.cpu().numpy()
        top_scores = sim[0].topk(10).values.cpu().numpy()
        top_cats = [all_cats[i] for i in top_idx]
    
    # Calculate fashion vs non-fashion scores
    f_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in fashion_categories)
    nf_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in non_fashion_categories)
    
    is_fashion = f_score > config.FASHION_SCORE_THRESHOLD and f_score > nf_score
    
    return {
        "is_fashion": is_fashion,
        "fashion_score": float(f_score),
        "non_fashion_score": float(nf_score),
        "top_predictions": [f"{top_cats[i]} ({top_scores[i]:.2f})" for i in range(min(3, len(top_cats)))]
    }
