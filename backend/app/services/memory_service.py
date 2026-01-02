"""
Memory Service for Chat Context.
Stores conversation context per session for follow-up queries.
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from threading import Lock

# Session TTL in seconds (30 minutes)
SESSION_TTL = 30 * 60

# Max history items to keep
MAX_HISTORY = 5


@dataclass
class SessionContext:
    """Context for a single session."""
    last_item_type: Optional[str] = None  # e.g., "tshirt", "dress", "jeans"
    last_color: Optional[str] = None      # e.g., "blue", "red"
    last_gender: Optional[str] = None     # e.g., "men", "women", "both"
    last_query: Optional[str] = None      # Full last query
    last_updated: float = field(default_factory=time.time)
    history: list = field(default_factory=list)  # List of past queries


class MemoryService:
    """Manages session-based chat memory."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = Lock()
    
    def get_context(self, session_id: str) -> Optional[SessionContext]:
        """Get context for a session, None if expired or not found."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is None:
                return None
            
            # Check TTL
            if time.time() - ctx.last_updated > SESSION_TTL:
                del self._sessions[session_id]
                return None
            
            return ctx
    
    def update_context(
        self,
        session_id: str,
        query: str,
        item_type: Optional[str] = None,
        color: Optional[str] = None,
        gender: Optional[str] = None
    ) -> SessionContext:
        """Update or create context for a session."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            
            if ctx is None:
                ctx = SessionContext()
                self._sessions[session_id] = ctx
            
            # Update fields if provided
            if item_type:
                ctx.last_item_type = item_type
            if color:
                ctx.last_color = color
            if gender:
                ctx.last_gender = gender
            
            ctx.last_query = query
            ctx.last_updated = time.time()
            
            # Add to history
            ctx.history.append(query)
            if len(ctx.history) > MAX_HISTORY:
                ctx.history = ctx.history[-MAX_HISTORY:]
            
            return ctx
    
    def clear_session(self, session_id: str):
        """Clear a session's context."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    def cleanup_expired(self):
        """Remove expired sessions."""
        current_time = time.time()
        with self._lock:
            expired = [
                sid for sid, ctx in self._sessions.items()
                if current_time - ctx.last_updated > SESSION_TTL
            ]
            for sid in expired:
                del self._sessions[sid]


# Global memory service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create the memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
        print("✅ Memory service initialized")
    return _memory_service
