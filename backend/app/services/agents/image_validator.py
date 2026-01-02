"""
Image Fashion Validator Agent.
Ported exactly from FP.ipynb - image_fashion_validator_agent function.
Validates if uploaded image is fashion-related and extracts embedding.
"""

import torch
from PIL import Image

from ...models.schemas import AgentState
from ...core.config import config
from ...utils.embeddings import get_image_embedding, get_clip_model, get_clip_processor


def image_fashion_validator_agent(state: AgentState) -> AgentState:
    """
    Validates if uploaded image is fashion-related and extracts embedding.
    Ported exactly from FP.ipynb.
    """
    if not state.get('image_input'):
        state['is_fashion_image'] = None
        state['image_embedding'] = None
        state['text_query'] = state.get('user_input', '').strip() or "hello"
        state['messages'].append("⭐ No image - text mode")
        state['next_agent'] = 'intent_classifier'
        return state
    
    try:
        image = Image.open(state['image_input']).convert('RGB')
        
        # EXTRACT IMAGE EMBEDDING IMMEDIATELY
        img_embedding = get_image_embedding(image)
        state['image_embedding'] = img_embedding
        
        clip_model = get_clip_model()
        clip_processor = get_clip_processor()
        
        inputs = clip_processor(images=image, return_tensors="pt").to(config.DEVICE)
        
        # Dynamic fashion categories from dataset
        fashion_cats = list(config.DYNAMIC_FASHION_ITEMS)[:30]
        generic_fashion = [
            "clothing", "fashion", "apparel", "footwear", "accessory",
            "garment", "outfit", "attire", "wear"
        ]
        fashion_cats = list(set(fashion_cats + generic_fashion))
        
        # Non-fashion categories
        non_fashion_cats = [
            "animal", "car", "vehicle", "building", "architecture",
            "food", "meal", "landscape", "nature", "plant", "tree",
            "electronics", "furniture", "tool", "instrument"
        ]
        
        all_cats = fashion_cats + non_fashion_cats
        
        # CLIP Classification
        text_inputs = clip_processor(text=all_cats, return_tensors="pt", padding=True).to(config.DEVICE)
        
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs)
            txt_feat = clip_model.get_text_features(**text_inputs)
            sim = (img_feat @ txt_feat.T).softmax(dim=-1)
            top_idx = sim[0].topk(10).indices.cpu().numpy()
            top_scores = sim[0].topk(10).values.cpu().numpy()
            top_cats = [all_cats[i] for i in top_idx]
        
        # Calculate fashion vs non-fashion scores
        f_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in fashion_cats)
        nf_score = sum(float(top_scores[i]) for i, c in enumerate(top_cats) if c in non_fashion_cats)
        
        is_fashion = f_score > config.FASHION_SCORE_THRESHOLD and f_score > nf_score
        
        top_3_cats = [f"{top_cats[i]} ({top_scores[i]:.2f})" for i in range(min(3, len(top_cats)))]
        reason = f"F={f_score:.2f}, NF={nf_score:.2f} | Top: {', '.join(top_3_cats)}"
        
        state['is_fashion_image'] = is_fashion
        state['image_validation_reason'] = reason
        state['debug_info'] = {
            'fashion_score': float(f_score),
            'non_fashion_score': float(nf_score),
            'top_predictions': top_3_cats,
            'fashion_categories_used': len(fashion_cats),
            'dynamic_categories': fashion_cats[:10],
            'image_embedding_extracted': True
        }
        
        state['messages'].append(
            f"{'✅ Fashion image' if is_fashion else '❌ Non-fashion'}: {reason}"
        )
        state['next_agent'] = 'image_to_description' if is_fashion else 'non_relevant_image_agent'
        
    except Exception as e:
        state['is_fashion_image'] = False
        state['image_embedding'] = None
        state['image_validation_reason'] = f"Error: {str(e)}"
        state['messages'].append(f"⚠️ Image error: {str(e)[:50]}")
        state['next_agent'] = 'non_relevant_image_agent'
    
    return state
