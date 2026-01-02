"""
Smart Query Understanding Agent.
Ported exactly from FP.ipynb - smart_query_understanding_agent function.
Gender-aware query understanding with Gemini API.
"""

import json
from google import genai
from google.genai import types

from ...models.schemas import AgentState
from ...core.config import config
from ..memory_service import get_memory_service


def smart_query_understanding_agent(state: AgentState) -> AgentState:
    """
    ENHANCED: Gender-aware query understanding WITH MEMORY.
    - Retrieves session context
    - Merges current query with context (e.g. "white" + context="tshirt" -> "white tshirt")
    - Detects gender from text (rule-based)
    
    Ported exactly from FP.ipynb.
    """
    
    query = state.get('text_query', '').strip().lower()
    session_id = state.get('session_id', '')
    has_image = state.get('is_fashion_image') == True
    image_desc = state.get('image_description', '')
    debug_info = state.get('debug_info', {})
    
    matched_items = debug_info.get('matched_items', [])
    matched_colors = debug_info.get('matched_colors', [])
    
    # ========================================================================
    # STEP 0: MEMORY & CONTEXT MERGING
    #Check if query is a follow-up (e.g., just a color or attribute like "white", "cheap", "cotton")
    # ========================================================================
    
    memory_service = get_memory_service()
    context = memory_service.get_context(session_id) if session_id else None
    
    # Words that suggest a follow-up query
    attribute_keywords = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 
                          'beige', 'brown', 'grey', 'gray', 'orange', 'gold', 'silver',
                          'cheap', 'expensive', 'cotton', 'silk', 'denim', 'leather', 
                          'casual', 'formal', 'summer', 'winter']
    
    is_short_query = len(query.split()) <= 2
    is_attribute = any(kw in query for kw in attribute_keywords)
    
    previous_item = context.last_item_type if context else None
    
    if is_short_query and is_attribute and previous_item and not has_image:
        # It's a follow-up! Merge with previous item
        # e.g., "white" + last="tshirt" -> "white tshirt"
        original_query = query
        query = f"{query} {previous_item}"
        state['messages'].append(f"🧠 Memory: Merged '{original_query}' + '{previous_item}' → '{query}'")
        state['text_query'] = query  # Update state query
    
    # ========================================================================
    # STEP 1: GENDER DETECTION (Query Text > Dropdown > Default)
    # Priority: 1) Extract from query text, 2) Use dropdown if set, 3) Default both
    # ========================================================================
    
    detected_gender_text = None
    final_gender = None
    gender_source = "none"
    
    # Get user's dropdown selection (lower priority)
    user_selected_gender = state.get('detected_gender')
    user_gender_source = state.get('gender_source')
    
    # FIRST: Try to detect gender from query text (highest priority)
    male_keywords = ['men', 'man', 'male', 'guy', 'boy', 'gentleman', 'his', 'he', 'him']
    female_keywords = ['women', 'woman', 'female', 'girl', 'lady', 'her', 'she']
    
    query_words = query.split()
    has_male = any(kw in query_words for kw in male_keywords)
    has_female = any(kw in query_words for kw in female_keywords)
    
    if has_male and not has_female:
        detected_gender_text = "men"
        state['messages'].append(f"🚹 Query text: Detected MALE gender")
    elif has_female and not has_male:
        detected_gender_text = "women"
        state['messages'].append(f"🚺 Query text: Detected FEMALE gender")
    elif has_male and has_female:
        detected_gender_text = "both"
        state['messages'].append(f"⚧ Query text: Both genders mentioned")
    else:
        state['messages'].append(f"⚪ Query text: No gender detected")
    
    # Determine final gender based on priority
    if detected_gender_text:
        # Priority 1: Gender found in query text - use it
        final_gender = detected_gender_text
        gender_source = "query_text"
        state['messages'].append(f"✅ Using gender from query: {final_gender.upper()}")
    elif user_selected_gender and user_selected_gender != "both" and user_gender_source == "user_selection":
        # Priority 2: User explicitly selected a specific gender (not "both")
        final_gender = user_selected_gender
        gender_source = "user_selection"
        state['messages'].append(f"� Using dropdown selection: {final_gender.upper()}")
    else:
        # Priority 3: Default to both
        final_gender = "both"
        gender_source = "default_both"
        state['messages'].append(f"🌐 No gender specified → Showing BOTH genders")
    
    state['detected_gender'] = final_gender
    state['gender_source'] = gender_source
    
    # ========================================================================
    # STEP 2: SCENARIO DETECTION
    # ========================================================================
    
    # Scenario 1: IMAGE ONLY
    # Scenario 1: IMAGE ONLY
    if has_image and (not query or query in ['similar', 'like this', 'same', ''] or len(query.split()) <= 2):
        state['search_mode'] = 'image_only'
        state['intent_type'] = 'image_search'
        state['messages'].append(f"🎯 Image-only search mode: Generating diverse queries from image")
        # Don't return early! Continue to Gemini to generate text queries from image description
        # We will add 'visual_search' to the queries later or let search_executor handle it
    
    # Scenario 4: IMAGE + TEXT (hybrid)
    elif has_image and query and len(query.split()) > 2:
        state['search_mode'] = 'hybrid'
        state['messages'].append(f"🎯 Hybrid mode: Text + Image (70% text, 30% image)")
    
    # Scenario 2, 3, 5: TEXT ONLY
    else:
        state['search_mode'] = 'text_only'
        state['messages'].append(f"🎯 Text-only mode")
    
    # ========================================================================
    # STEP 3: GEMINI QUERY GENERATION WITH GENDER AWARENESS
    # ========================================================================
    
    context_parts = []
    if has_image:
        context_parts.append(f"User uploaded image: {image_desc}")
    if matched_items:
        context_parts.append(f"Detected items: {', '.join(matched_items[:5])}")
    if matched_colors:
        context_parts.append(f"Detected colors: {', '.join(matched_colors[:5])}")
    
    context_parts.append(f"Target gender: {final_gender.upper()} (source: {gender_source})")
    context_str = "\n".join(context_parts)
    
    available_items = list(config.DYNAMIC_FASHION_ITEMS)[:40]
    available_colors = list(config.DYNAMIC_COLORS)[:25]
    available_genders = list(config.DYNAMIC_GENDERS)
    
    system_instruction = f"""You are an Expert Fashion Search Query Generator with GENDER AWARENESS.

**AVAILABLE INVENTORY:**
- Items: {', '.join(available_items)}
- Colors: {', '.join(available_colors)}
- Genders: {', '.join(available_genders)}

**GENDER HANDLING (IMPORTANT):**
- Target gender: {final_gender.upper()}
- If target is "MEN" → Include ONLY men's fashion queries
- If target is "WOMEN" → Include ONLY women's fashion queries
- If target is "BOTH" → Include queries for BOTH men AND women (duplicate queries for each gender)

**YOUR TASK:** Generate search queries based on these scenarios:

---

**SCENARIO 1: IMAGE ONLY - DIVERSE SEARCH**
- The user uploaded an image without text.
- Generate 3-4 diverse queries based on the image description to find similar items.
- Examples (if image is blue floral dress):
  * "women blue floral dress"
  * "women casual summer dress"
  * "women floral print midi dress"

**SCENARIO 2: TEXT ONLY - SPECIFIC ITEM**
- Generate: 1-2 queries for that specific item
- Examples:
  * "blue shirt" + BOTH gender → ["men blue shirt", "women blue shirt"]
  * "black jeans" + MEN gender → ["men black jeans"]
  * "red dress" + WOMEN gender → ["women red dress"]

**SCENARIO 3: TEXT ONLY - RECOMMENDATION REQUEST (COMPLETE OUTFIT)**
- Generate: Multiple queries covering ALL OUTFIT CATEGORIES
- **REQUIRED CATEGORIES:** top, bottom, footwear, accessories, watches
- Gender awareness: Duplicate for each gender if target is BOTH
- **ALWAYS include watches in recommendations for complete looks**

**SCENARIO 4: IMAGE + TEXT - MATCHING REQUEST**
- Examples:
  * Text: "black shirt" + MEN → ["men black shirt"]
  * Text: "black shirt" + BOTH → ["men black shirt", "women black shirt"]

**SCENARIO 5: TEXT - ITEM FOR ITEM**
- Examples:
  * "black shirt for blue pants" + BOTH → ["men black shirt", "women black shirt"]

---

**OUTPUT FORMAT - STRICT JSON:**

For Scenario 2, 4, 5 (Direct/Matching):
```json
{{
  "intent": "direct_search",
  "queries": ["query1", "query2"],
  "results_per_query": 5,
  "gender_aware": true
}}
```

For Scenario 3 (Recommendations with WATCHES):
```json
{{
  "intent": "outfit_recommendation",
  "categories": [
    {{"category": "top", "queries": ["men formal shirt", "women formal shirt"]}},
    {{"category": "bottom", "queries": ["men dress pants", "women dress pants"]}},
    {{"category": "footwear", "queries": ["men formal shoes", "women heels"]}},
    {{"category": "accessories", "queries": ["men leather belt", "women clutch"]}},
    {{"category": "watches", "queries": ["men formal watch", "women elegant watch"]}}
  ],
  "results_per_query": 5,
  "gender_aware": true
}}
```

**CRITICAL RULES:**
- ALWAYS include gender prefix in queries: "men [item]" or "women [item]"
- If target gender is BOTH → Generate queries for BOTH genders
- If target gender is MEN or WOMEN → Generate queries for ONLY that gender
- **ALWAYS include "watches" category for outfit recommendations**
- 2-6 words per query
- Return ONLY valid JSON"""

    user_prompt = f"""Analyze this fashion search request:

User Query: "{query}"

Context:
{context_str}

Target Gender: {final_gender.upper()}

Generate appropriate search queries with gender awareness. Return ONLY JSON."""

    # ========================================================================
    # CALL GEMINI API
    # ========================================================================
    
    try:
        client = genai.Client(api_key=config.get_gemini_api_key())
        
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(response_text)
        
        intent_type = parsed_response.get('intent', 'direct_search')
        
        if intent_type == "outfit_recommendation":
            all_queries = []
            categories_info = []
            
            for cat in parsed_response.get('categories', []):
                category_name = cat.get('category', 'items')
                category_queries = cat.get('queries', [])
                
                for q in category_queries:
                    all_queries.append(q)
                    categories_info.append(category_name)
            
            state['search_queries'] = all_queries
            state['query_categories'] = categories_info
            state['intent_type'] = 'recommendation'
            state['messages'].append(f"✅ Gemini: {len(all_queries)} gender-aware queries")
        
        else:
            queries = parsed_response.get('queries', [])
            if not queries:
                queries = [query if query else "fashion items"]
            
            state['search_queries'] = queries[:10]
            state['query_categories'] = ['general'] * len(state['search_queries'])
            state['intent_type'] = intent_type
            state['messages'].append(f"✅ Gemini: {len(state['search_queries'])} queries")
        
    except Exception as e:
        # ====================================================================
        # FALLBACK: Rule-based with gender awareness
        # ====================================================================
        state['messages'].append(f"⚠️ Gemini failed → Rule-based fallback")
        
        gender_prefixes = []
        if final_gender == "both":
            gender_prefixes = ["men", "women"]
        elif final_gender == "men":
            gender_prefixes = ["men"]
        elif final_gender == "women":
            gender_prefixes = ["women"]
        else:
            gender_prefixes = ["men", "women"]
        
        if any(word in query for word in ['wedding', 'party', 'office', 'recommend', 'outfit']):
            state['search_queries'] = []
            state['query_categories'] = []
            for gender_prefix in gender_prefixes:
                state['search_queries'].extend([
                    f"{gender_prefix} formal shirt",
                    f"{gender_prefix} dress pants",
                    f"{gender_prefix} formal shoes",
                    f"{gender_prefix} leather belt",
                    f"{gender_prefix} formal watch"
                ])
                state['query_categories'].extend(['top', 'bottom', 'footwear', 'accessories', 'watches'])
            state['intent_type'] = 'recommendation'
        
        else:
            state['search_queries'] = [f"{gp} {query}" for gp in gender_prefixes]
            state['query_categories'] = ['general'] * len(state['search_queries'])
            state['intent_type'] = 'direct_search'
    
    state['debug_info'].update({
        'search_mode': state.get('search_mode'),
        'detected_gender': final_gender,
        'gender_source': gender_source,
        'gender_from_text': detected_gender_text,
        'user_dropdown': user_selected_gender
    })
    
    state['next_agent'] = 'search_executor'
    
    # ========================================================================
    # STEP 4: MEMORY UPDATE
    # Extract item type from query to store in context
    # ========================================================================
    if session_id:
        # Simple heuristic to extract main item
        # In a real app, use NER or the detailed Gemini response
        item_types = ['shirt', 'tshirt', 'jeans', 'pants', 'trousers', 'dress', 'skirt', 
                      'shoes', 'sneakers', 'boots', 'heels', 'jacket', 'coat', 'blazer',
                      'watch', 'bag', 'handbag', 'belt', 'scarf', 'hat', 'cap',
                      'suit', 'tuxedo', 'shorts', 'leggings', 'sweater', 'hoodie']
        
        found_item = next((item for item in item_types if item in query), None)
        
        # Don't overwrite item type if we just used it in a merger and didn't find a new one
        if found_item:
            memory_service.update_context(
                session_id=session_id,
                query=query,
                item_type=found_item,
                gender=final_gender
            )
        elif context and context.last_item_type:
             # Keep previous item if this was just an attribute query
             memory_service.update_context(
                session_id=session_id,
                query=query,
                item_type=context.last_item_type,
                gender=final_gender
            )
    
    return state
