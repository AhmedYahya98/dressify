"""
LLM Service for Fashion AI Chatbot.
Handles BERT classifier initialization.
Generator LLM (T5) has been removed as it is no longer used.
"""

from typing import Optional, Tuple, Any
from transformers import pipeline

from ..core.config import config

# Global LLM instances
_classifier_pipeline: Optional[Any] = None  # BERT text-classification pipeline
_llms_initialized = False


def classify_text(text: str) -> Tuple[str, float]:
    """
    Classify text using BERT classifier.
    Returns: (label, confidence_score)
    Labels: 'fashion', 'welcome', 'non-fashion'
    """
    global _classifier_pipeline
    if _classifier_pipeline is None:
        return "fashion", 0.5  # Default fallback
    
    try:
        result = _classifier_pipeline(text)
        # Result format: [[{'label': 'fashion', 'score': 0.97}]]
        if result and len(result) > 0:
            top_result = result[0] if isinstance(result[0], dict) else result[0][0]
            return top_result['label'], top_result['score']
        return "fashion", 0.5
    except Exception as e:
        print(f"Classification error: {e}")
        return "fashion", 0.5


def initialize_llms() -> Any:
    """
    Initialize classifier LLM.
    Classifier uses BERT text-classification.
    """
    global _classifier_pipeline, _llms_initialized
    
    if _llms_initialized:
        return _classifier_pipeline
    
    print("\n🔧 Initializing LLMs...")
    
    # Classifier (BERT for intent classification: fashion/welcome/non-fashion)
    try:
        _classifier_pipeline = pipeline(
            "text-classification",
            model=config.CLASSIFIER_MODEL,
            device=0 if config.DEVICE == "cuda" else -1,
            top_k=1  # Return only top prediction
        )
        print(f"✅ Classifier: {config.CLASSIFIER_MODEL}")
    except Exception as e:
        print(f"⚠️ Classifier failed: {e}")
        _classifier_pipeline = None
    
    _llms_initialized = True
    return _classifier_pipeline


def get_classifier() -> Optional[Any]:
    """Get the BERT classifier pipeline instance."""
    global _classifier_pipeline, _llms_initialized
    if not _llms_initialized:
        initialize_llms()
    return _classifier_pipeline
