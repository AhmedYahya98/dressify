"""
Image to Description Agent.
Ported exactly from FP.ipynb - image_to_description_agent function.
Converts fashion image to text description.
"""

import torch
from PIL import Image

from ...models.schemas import AgentState
from ...core.config import config
from ...utils.embeddings import get_clip_model, get_clip_processor


def image_to_description_agent(state: AgentState) -> AgentState:
    """
    Converts fashion image to text description (lightweight for context).
    Ported exactly from FP.ipynb.
    """
    user_text = state.get('user_input', '').strip()
    
    if state.get('image_input') and state.get('is_fashion_image'):
        try:
            image = Image.open(state['image_input']).convert('RGB')
            
            clip_model = get_clip_model()
            clip_processor = get_clip_processor()
            
            inputs = clip_processor(images=image, return_tensors="pt").to(config.DEVICE)
            
            # Get top attributes from dataset vocabulary
            sample_items = list(config.DYNAMIC_FASHION_ITEMS)[:50]
            sample_colors = list(config.DYNAMIC_COLORS)[:30]
            
            attrs = sample_items + sample_colors + ["casual", "formal", "summer", "winter"]
            text_inputs = clip_processor(text=attrs, return_tensors="pt", padding=True).to(config.DEVICE)
            
            with torch.no_grad():
                img_feat = clip_model.get_image_features(**inputs)
                txt_feat = clip_model.get_text_features(**text_inputs)
                sim = (img_feat @ txt_feat.T).softmax(dim=-1)
                top_idx = sim[0].topk(5).indices.cpu().numpy()
                detected = [attrs[i] for i in top_idx]
            
            # Build richer description
            image_desc = " ".join(detected[:5])
            state['image_description'] = image_desc
            
            if user_text:
                state['text_query'] = user_text
                state['messages'].append(f"📝 Text: '{user_text}' + Image: {image_desc}")
            else:
                state['text_query'] = image_desc  # Fallback description
                state['messages'].append(f"📸 Image query: {image_desc}")
                
        except Exception as e:
            state['text_query'] = user_text or "fashion items"
            state['messages'].append(f"⚠️ Image desc error")
    else:
        state['text_query'] = user_text or "hello"
        state['messages'].append(f"💬 Text mode")
    
    state['next_agent'] = 'intent_classifier'
    return state
