"""
Pydantic models and TypedDict for Fashion AI Chatbot.
AgentState is ported exactly from FP.ipynb.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field
import operator
import numpy as np


# =============================================================================
# API REQUEST/RESPONSE MODELS (Pydantic)
# =============================================================================

class SearchItem(BaseModel):
    """Individual product item in search results."""
    id: int
    title: str
    brand: str
    price: str
    color: str
    article_type: str
    snippet: str
    source_path: str
    thumbnail_url: str
    score: float
    gender: str


class SearchGroup(BaseModel):
    """Grouped search results by query."""
    query_number: int
    query_text: str
    category: str
    items: List[SearchItem]
    item_count: int
    gender_filter: Optional[str] = None


class SearchResponse(BaseModel):
    """Full API response for search endpoint."""
    success: bool = True
    final_response: str
    search_results_data: List[SearchGroup] = []
    search_mode: Optional[str] = None
    detected_gender: Optional[str] = None
    gender_source: Optional[str] = None
    intent_type: Optional[str] = None
    messages: List[str] = []
    debug_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_size: int
    vocabulary_items: int
    vocabulary_colors: int
    device: str


class SearchRequest(BaseModel):
    """Search request body (for JSON requests without file upload)."""
    text_query: Optional[str] = ""


# =============================================================================
# AGENT STATE (TypedDict - ported exactly from notebook)
# =============================================================================

class AgentState(TypedDict):
    """
    Agent state dictionary - ported exactly from FP.ipynb.
    Used by LangGraph workflow for passing state between agents.
    """
    user_input: str
    image_input: Optional[str]
    image_embedding: Optional[np.ndarray]  # Store image embedding
    is_fashion_image: Optional[bool]
    image_validation_reason: Optional[str]
    image_description: Optional[str]
    text_query: Optional[str]
    intent: Optional[str]
    intent_class: Optional[str]
    messages: Annotated[List[str], operator.add]
    final_response: Optional[str]
    next_agent: Optional[str]
    debug_info: Optional[dict]
    search_queries: Optional[List[str]]
    search_results_data: Optional[List[dict]]
    query_categories: Optional[List[str]]
    intent_type: Optional[str]
    search_mode: Optional[str]  # 'image_only', 'text_only', 'hybrid'
    detected_gender: Optional[str]  # Track detected gender
    gender_source: Optional[str]  # Track how gender was detected


def create_initial_state(user_text: str = "", image_path: Optional[str] = None, user_gender: str = "both") -> AgentState:
    """Create initial agent state for a new query.
    
    Args:
        user_text: User's text query
        image_path: Optional path to uploaded image
        user_gender: Gender filter from user selection (men, women, both)
    """
    return {
        "user_input": user_text or "",
        "image_input": image_path,
        "image_embedding": None,
        "is_fashion_image": None,
        "image_validation_reason": None,
        "image_description": None,
        "text_query": None,
        "intent": None,
        "intent_class": None,
        "messages": [],
        "final_response": None,
        "next_agent": None,
        "debug_info": {},
        "search_queries": [],
        "search_results_data": [],
        "query_categories": [],
        "intent_type": None,
        "search_mode": None,
        "detected_gender": user_gender,  # Always use user selection (men, women, or both)
        "gender_source": "user_selection"  # Mark as user-selected
    }
