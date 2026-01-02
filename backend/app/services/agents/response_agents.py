"""
Response Agents.
Ported exactly from FP.ipynb - welcome_agent, non_relevant_agent, non_relevant_image_agent.
"""

from ...models.schemas import AgentState


def welcome_agent(state: AgentState) -> AgentState:
    """
    Welcomes user and provides examples.
    Ported exactly from FP.ipynb.
    """
    state['final_response'] = """✨ **Hello! I'm your Personal AI Stylist.**

I'm here to help you look your best! Here's what I can do for you:

🛍️ **Find Items:**
_"Show me red velvet dresses"_
_"I need white sneakers for running"_

🎨 **Style Advice & Matching:**
_"What goes well with blue jeans?"_
_"Find a shirt that matches this jacket"_ (upload photo)

👗 **Complete Outfits:**
_"Outfit for a summer wedding"_
_"Business casual look for men"_

📸 **Visual Search:**
Upload any fashion image, and I'll find similar items or potential matches!

**How can I help you today?**"""
    state['next_agent'] = 'end'
    state['messages'].append("✅ Ended: Welcome message")
    return state


def non_relevant_agent(state: AgentState) -> AgentState:
    """
    Handles non-fashion text queries.
    Ported exactly from FP.ipynb.
    """
    state['final_response'] = """🤔 **I'm focused on fashion and style!**

I can't help with that request, but I'd love to help you find your next great outfit.

**Try asking me about:**
• Clothing items (dresses, shirts, pants)
• Fashion accessories (shoes, bags, watches)
• Style advice and color matching
• Outfit recommendations

*What fashion item are you looking for?* ✨"""
    state['next_agent'] = 'end'
    state['messages'].append("❌ Ended: Non-fashion query")
    return state


def non_relevant_image_agent(state: AgentState) -> AgentState:
    """
    Handles non-fashion images.
    Ported exactly from FP.ipynb.
    """
    reason = state.get('image_validation_reason', 'Image is not fashion-related')
    state['final_response'] = f"""📸 **Non-Fashion Image Detected**

{reason}

Please upload fashion items (clothing, shoes, accessories) for search!"""
    state['next_agent'] = 'end'
    state['messages'].append("❌ Ended: Non-fashion image")
    return state
