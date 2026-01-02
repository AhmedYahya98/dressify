"""
Search Executor Agent.
Ported exactly from FP.ipynb - search_executor_agent function.
Handles image-only, text-only, and hybrid searches with gender filtering.
"""

import os
import numpy as np

from ...models.schemas import AgentState
from ...core.config import config
from ...utils.embeddings import get_text_embedding
from ...utils.faiss_manager import faiss_manager


def search_executor_agent(state: AgentState) -> AgentState:
    """
    FIXED: Properly handles image-only, text-only, and hybrid searches with gender filtering.
    Ported exactly from FP.ipynb.
    """
    search_mode = state.get('search_mode', 'text_only')
    queries = state.get('search_queries', [])
    categories = state.get('query_categories', [])
    intent_type = state.get('intent_type', 'direct_search')
    image_embedding = state.get('image_embedding')
    detected_gender = state.get('detected_gender', 'both')
    
    if not queries and image_embedding is None:
        state['final_response'] = "❌ No search criteria available"
        state['search_results_data'] = []
        state['next_agent'] = 'end'
        return state
    
    state['messages'].append(f"🔍 Search mode: {search_mode}, Gender: {detected_gender.upper()}")
    
    all_grouped_results = []
    
    # ========================================================================
    # IMAGE-ONLY SEARCH (100% visual similarity)
    # ========================================================================
    if search_mode == 'image_only' and image_embedding is not None:
        try:
            img_emb_normalized = image_embedding / np.linalg.norm(image_embedding)
            
            distances, indices = faiss_manager.search(img_emb_normalized, k=15)
            
            valid_items = []
            for faiss_idx, score in zip(indices, distances):
                if score < 0.5:
                    continue
                
                meta = faiss_manager.get_metadata(int(faiss_idx))
                img_path = meta.get('source_path') or meta.get('thumbnail_url')
                
                if img_path and os.path.exists(img_path):
                    valid_items.append({
                        'id': int(meta.get('image_id', meta['id'])),  # Use actual image ID from dataset
                        'title': meta['title'],
                        'brand': meta['brand'],
                        'price': meta['price'],
                        'color': meta['color'],
                        'article_type': meta['article_type'],
                        'snippet': meta['snippet'],
                        'source_path': img_path,
                        'thumbnail_url': img_path,
                        'score': float(score),
                        'gender': meta.get('gender', 'N/A')
                    })
                    
                    if len(valid_items) >= 5:
                        break
            
            all_grouped_results.append({
                "query_number": 1,
                "query_text": "Similar items (visual search)",
                "category": "similar",
                "items": valid_items,
                "item_count": len(valid_items)
            })
            
            state['messages'].append(f"  ✓ Visual search: {len(valid_items)} similar items")
            
        except Exception as e:
            state['messages'].append(f"  ⚠️ Image search failed: {str(e)[:50]}")
    
    # ========================================================================
    # TEXT-ONLY or HYBRID SEARCH WITH GENDER FILTERING
    # ========================================================================
    else:
        for idx, query_text in enumerate(queries):
            category = categories[idx] if idx < len(categories) else 'general'
            
            try:
                # Get text embedding
                text_emb = get_text_embedding(query_text)
                text_emb = text_emb / np.linalg.norm(text_emb)
                
                # HYBRID: Combine text and image embeddings
                if search_mode == 'hybrid' and image_embedding is not None:
                    img_emb_normalized = image_embedding / np.linalg.norm(image_embedding)
                    combined_emb = (config.TEXT_WEIGHT * text_emb + 
                                   config.IMAGE_WEIGHT * img_emb_normalized)
                    combined_emb = combined_emb / np.linalg.norm(combined_emb)
                    search_emb = combined_emb
                else:
                    search_emb = text_emb
                
                # Search FAISS
                distances, indices = faiss_manager.search(search_emb, k=20)
                
                # Collect valid results with gender filtering
                valid_items = []
                for faiss_idx, score in zip(indices, distances):
                    meta = faiss_manager.get_metadata(int(faiss_idx))
                    item_gender = str(meta.get('gender', '')).lower()
                    
                    # Gender filtering logic
                    if detected_gender == "men" and item_gender not in ["men", "male", "boys"]:
                        continue
                    elif detected_gender == "women" and item_gender not in ["women", "female", "girls"]:
                        continue
                    # If detected_gender == "both", include all items
                    
                    img_path = meta.get('source_path') or meta.get('thumbnail_url')
                    
                    if img_path and os.path.exists(img_path):
                        valid_items.append({
                            'id': int(meta.get('image_id', meta['id'])),  # Use actual image ID from dataset
                            'title': meta['title'],
                            'brand': meta['brand'],
                            'price': meta['price'],
                            'color': meta['color'],
                            'article_type': meta['article_type'],
                            'snippet': meta['snippet'],
                            'source_path': img_path,
                            'thumbnail_url': img_path,
                            'score': float(score),
                            'gender': item_gender
                        })
                        
                        if len(valid_items) >= 5:
                            break
                
                all_grouped_results.append({
                    "query_number": idx + 1,
                    "query_text": query_text,
                    "category": category,
                    "items": valid_items,
                    "item_count": len(valid_items),
                    "gender_filter": detected_gender
                })
                
                mode_str = "hybrid" if search_mode == 'hybrid' else "text"
                state['messages'].append(f"  ✓ Q{idx+1} ({mode_str}) [{category}]: '{query_text}' → {len(valid_items)} items ({detected_gender})")
                
            except Exception as e:
                state['messages'].append(f"  ⚠️ Query {idx+1} failed: {str(e)[:50]}")
                all_grouped_results.append({
                    "query_number": idx + 1,
                    "query_text": query_text,
                    "category": category,
                    "items": [],
                    "item_count": 0,
                    "gender_filter": detected_gender
                })
    
    # ========================================================================
    # BUILD FINAL RESPONSE WITH GENDER INFO
    # ========================================================================
    
    total_items = sum(g['item_count'] for g in all_grouped_results)
    state['search_results_data'] = all_grouped_results
    
    gender_info = ""
    if detected_gender == "men":
        gender_info = "\n🚹 **Showing: Men's Fashion Only**"
    elif detected_gender == "women":
        gender_info = "\n🚺 **Showing: Women's Fashion Only**"
    elif detected_gender == "both":
        gender_info = "\n⚧ **Showing: Both Men's and Women's Fashion**"
    
    if search_mode == 'image_only':
        state['final_response'] = f"""📸 **Similar Fashion Items**

**Found {total_items} visually similar items**{gender_info}

---

**💡 Tip:** These items match the style, color, and type of your uploaded image!"""
    
    elif intent_type == 'recommendation':
        state['final_response'] = f"""✨ **Complete Outfit Recommendation**

**{len(queries)} Items Curated for Your Occasion**
**{total_items} Total Products Found**{gender_info}

---

**💡 Styling Tip:** Mix and match these pieces for a complete look!"""
        
        categories_dict = {}
        category_emojis = {
            'top': '👕',
            'bottom': '👖',
            'footwear': '👟',
            'accessories': '👜',
            'watches': '⌚'
        }
        
        for group in all_grouped_results:
            cat = group['category']
            if cat not in categories_dict:
                categories_dict[cat] = []
            categories_dict[cat].append(group)
        
        # Display in logical order
        category_order = ['top', 'bottom', 'footwear', 'accessories', 'watches']
        
        for cat_name in category_order:
            if cat_name in categories_dict:
                cat_groups = categories_dict[cat_name]
                total_cat_items = sum(g['item_count'] for g in cat_groups)
                emoji = category_emojis.get(cat_name, '📦')
                state['final_response'] += f"\n\n**{emoji} {cat_name.upper()}** ({total_cat_items} items)"
                for group in cat_groups:
                    if group['item_count'] > 0:
                        state['final_response'] += f"\n  └─ {group['query_text']}: {group['item_count']} options"
    
    elif search_mode == 'hybrid':
        state['final_response'] = f"""🎨 **Smart Matching Results**

**These items match your image and text!**
**{total_items} Total Matches**{gender_info}

---

**💡 Tip:** Results prioritize your text (70%) with visual hints."""
        
        for group in all_grouped_results:
            if group['item_count'] > 0:
                state['final_response'] += f"\n\n**{group['query_text']}**"
                state['final_response'] += f"\n└─ {group['item_count']} items"
    
    else:
        state['final_response'] = f"""🔍 **Search Results**

**Found {total_items} items**{gender_info}

---"""
        
        for group in all_grouped_results:
            if group['item_count'] > 0:
                state['final_response'] += f"\n\n**{group['query_text']}**"
                state['final_response'] += f"\n└─ {group['item_count']} items"
    
    state['next_agent'] = 'end'
    state['messages'].append(f"✅ Complete: {total_items} items, mode={search_mode}, gender={detected_gender}")
    
    return state
