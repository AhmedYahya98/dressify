"""
Intent Classifier Agent.
Uses BERT classifier for fashion/non-fashion detection.
Welcome messages are detected via keyword patterns (only if no fashion intent).
"""

from ...models.schemas import AgentState
from ...core.config import config
from ..llm_service import classify_text


# Welcome/greeting patterns (only used if no fashion intent)
WELCOME_PATTERNS = [
    "hi", "hello", "hey", "hii", "hiii",
    "good morning", "good evening", "good afternoon", "good night",
    "what can you", "who are you", "what do you do", "help me",
    "greetings", "howdy", "sup", "yo", "hola",
    "thank you", "thanks", "bye", "goodbye", "see you",
    "how are you", "what's up", "whats up"
]


def has_fashion_signals(query: str) -> bool:
    """
    Check if query contains fashion-related words.
    Used to override welcome detection when user says "hello i want tshirt".
    """
    fashion_keywords = [
        # Clothing items
        "shirt", "tshirt", "t-shirt", "dress", "jeans", "pants", "shorts",
        "jacket", "coat", "sweater", "hoodie", "skirt", "blouse", "top",
        "kurta", "saree", "lehenga", "suit", "blazer", "trouser", "legging",
        # Accessories
        "watch", "bag", "shoe", "sandal", "heel", "sneaker", "boot",
        "sunglasses", "belt", "scarf", "hat", "cap", "jewelry",
        # Action words
        "want", "need", "looking for", "find", "show", "search", "buy",
        "recommend", "suggest", "get me", "i need", "give me",
        # Style words
        "casual", "formal", "party", "wedding", "summer", "winter", "outfit"
    ]
    
    for kw in fashion_keywords:
        if kw in query:
            return True
    return False


def is_pure_greeting(query: str) -> bool:
    """Check if query is ONLY a greeting with no fashion intent."""
    for pattern in WELCOME_PATTERNS:
        if query.startswith(pattern) or query == pattern:
            # Check if there's more than just the greeting
            if not has_fashion_signals(query):
                return True
    return False


def intent_classifier_agent(state: AgentState) -> AgentState:
    """
    Smart hybrid intent classification:
    1. Check for pure greetings (no fashion keywords)
    2. Fashion/non-fashion via BERT classifier
    """
    query = state.get('text_query', '').lower().strip()
    
    # If no text query but has fashion image, treat as fashion
    if not query and state.get('is_fashion_image'):
        state['intent'] = 'relevant_fashion'
        state['next_agent'] = 'fashion_classifier'
        state['messages'].append("🎯 Fashion detected (image only)")
        return state
    
    if not query:
        state['intent'] = 'non_relevant'
        state['next_agent'] = 'non_relevant_agent'
        state['messages'].append("🎯 No query provided")
        return state
    
    # STEP 1: Check for PURE greetings (no fashion keywords)
    if is_pure_greeting(query):
        state['intent'] = 'welcome'
        state['next_agent'] = 'welcome_agent'
        state['messages'].append("🎯 Welcome intent (pure greeting)")
        return state
    
    # STEP 2: Use BERT classifier for fashion/non-fashion
    label, confidence = classify_text(query)
    
    state['debug_info'] = state.get('debug_info', {})
    state['debug_info']['bert_label'] = label
    state['debug_info']['bert_confidence'] = confidence
    
    # Handle BERT output (fashion or non-fashion / LABEL_0 or LABEL_1)
    is_fashion = label.lower() in ['fashion', 'label_1', '1', 'true']
    
    if is_fashion:
        state['intent'] = 'relevant_fashion'
        state['next_agent'] = 'fashion_classifier'
        state['messages'].append(f"🎯 Fashion intent (BERT: {confidence:.2f})")
    else:
        # If classified as non-fashion but has fashion image, override
        if state.get('is_fashion_image'):
            state['intent'] = 'relevant_fashion'
            state['next_agent'] = 'fashion_classifier'
            state['messages'].append("🎯 Fashion (image override)")
        else:
            state['intent'] = 'non_relevant'
            state['next_agent'] = 'non_relevant_agent'
            state['messages'].append(f"🎯 Non-fashion (BERT: {confidence:.2f})")
    
    return state
