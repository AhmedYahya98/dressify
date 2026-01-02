"""
LangGraph Workflow for Fashion AI Chatbot.
Ported from FP.ipynb - StateGraph workflow with conditional routing.
"""

from langgraph.graph import StateGraph, END

from ..models.schemas import AgentState
from .agents import (
    image_fashion_validator_agent,
    image_to_description_agent,
    non_relevant_image_agent,
    intent_classifier_agent,
    welcome_agent,
    non_relevant_agent,
    smart_query_understanding_agent,
    search_executor_agent
)

# Global workflow instance
_workflow = None
_compiled_app = None


def route_agent(state: AgentState) -> str:
    """Route to next agent based on state."""
    return state.get('next_agent', 'end')


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow.
    Ported exactly from FP.ipynb.
    """
    workflow = StateGraph(AgentState)
    
    # Add all agent nodes
    workflow.add_node("image_fashion_validator", image_fashion_validator_agent)
    workflow.add_node("image_to_description", image_to_description_agent)
    workflow.add_node("non_relevant_image_agent", non_relevant_image_agent)
    workflow.add_node("intent_classifier", intent_classifier_agent)
    workflow.add_node("welcome_agent", welcome_agent)
    workflow.add_node("non_relevant_agent", non_relevant_agent)
    workflow.add_node("smart_query_understanding", smart_query_understanding_agent)
    workflow.add_node("search_executor", search_executor_agent)
    
    # Set entry point
    workflow.set_entry_point("image_fashion_validator")
    
    # Add conditional edges from image_fashion_validator
    workflow.add_conditional_edges(
        "image_fashion_validator",
        route_agent,
        {
            "image_to_description": "image_to_description",
            "non_relevant_image_agent": "non_relevant_image_agent",
            "intent_classifier": "intent_classifier",
            "end": END
        }
    )
    
    # Add edges from non_relevant_image_agent
    workflow.add_conditional_edges(
        "non_relevant_image_agent",
        route_agent,
        {"end": END}
    )
    
    # Add edges from image_to_description
    workflow.add_conditional_edges(
        "image_to_description",
        route_agent,
        {"intent_classifier": "intent_classifier", "end": END}
    )
    
    # Add conditional edges from intent_classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_agent,
        {
            "welcome_agent": "welcome_agent",
            "non_relevant_agent": "non_relevant_agent",
            "fashion_classifier": "smart_query_understanding",
            "end": END
        }
    )
    
    # Add edges from welcome_agent
    workflow.add_conditional_edges(
        "welcome_agent",
        route_agent,
        {"end": END}
    )
    
    # Add edges from non_relevant_agent
    workflow.add_conditional_edges(
        "non_relevant_agent",
        route_agent,
        {"end": END}
    )
    
    # Add edge from smart_query_understanding to search_executor
    workflow.add_edge("smart_query_understanding", "search_executor")
    
    # Add edge from search_executor to END
    workflow.add_edge("search_executor", END)
    
    return workflow


def get_workflow():
    """Get or create the workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def get_compiled_app():
    """Get or create the compiled workflow app."""
    global _compiled_app
    if _compiled_app is None:
        workflow = get_workflow()
        _compiled_app = workflow.compile()
        print("✅ Workflow compiled and ready")
    return _compiled_app


def run_query(user_text: str = "", image_path: str = None, user_gender: str = "both", session_id: str = "") -> dict:
    """
    Run a query through the workflow.
    
    Args:
        user_text: User's text query
        image_path: Optional path to uploaded image
        user_gender: Gender filter from user (men, women, both)
        session_id: Session ID for chat memory
        
    Returns:
        Final state dictionary with results
    """
    from ..models.schemas import create_initial_state
    
    app = get_compiled_app()
    state = create_initial_state(user_text, image_path, user_gender)
    state['session_id'] = session_id  # Add session_id to state
    result = app.invoke(state)
    
    return result
