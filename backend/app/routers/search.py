"""
Search API Router for Fashion AI Chatbot.
Handles search requests with text and/or image input.
"""

import os
import tempfile
import shutil
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from ..models.schemas import SearchResponse, HealthResponse, SearchItem, SearchGroup
from ..core.config import config
from ..services.workflow import run_query
from ..utils.faiss_manager import faiss_manager

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is ready."""
    return HealthResponse(
        status="ready" if faiss_manager.is_loaded else "initializing",
        index_size=faiss_manager.size,
        vocabulary_items=len(config.DYNAMIC_FASHION_ITEMS),
        vocabulary_colors=len(config.DYNAMIC_COLORS),
        device=config.DEVICE
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    text_query: Optional[str] = Form(default=""),
    gender_filter: Optional[str] = Form(default="both"),
    session_id: Optional[str] = Form(default=""),
    image: Optional[UploadFile] = File(default=None)
):
    """
    Search for fashion items.
    
    Args:
        text_query: Text search query
        gender_filter: Gender filter (men, women, both)
        session_id: Session ID for chat memory
        image: Optional image file upload
        
    Returns:
        SearchResponse with results
    """
    temp_image_path = None
    
    try:
        # Handle image upload
        if image and image.filename:
            # Save to temp file
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(image.file, tmp)
                temp_image_path = tmp.name
        
        # Run the workflow with gender filter and session
        result = run_query(
            user_text=text_query or "",
            image_path=temp_image_path,
            user_gender=gender_filter or "both",
            session_id=session_id or ""
        )
        
        # Convert search results to proper format
        search_results = []
        for group_data in result.get('search_results_data', []):
            items = []
            for item_data in group_data.get('items', []):
                items.append(SearchItem(
                    id=item_data.get('id', 0),
                    title=item_data.get('title', ''),
                    brand=item_data.get('brand', ''),
                    price=str(item_data.get('price', 'N/A')),
                    color=item_data.get('color', ''),
                    article_type=item_data.get('article_type', ''),
                    snippet=item_data.get('snippet', ''),
                    source_path=item_data.get('source_path', ''),
                    thumbnail_url=item_data.get('thumbnail_url', ''),
                    score=float(item_data.get('score', 0)),
                    gender=item_data.get('gender', 'N/A')
                ))
            
            search_results.append(SearchGroup(
                query_number=group_data.get('query_number', 0),
                query_text=group_data.get('query_text', ''),
                category=group_data.get('category', 'general'),
                items=items,
                item_count=group_data.get('item_count', 0),
                gender_filter=group_data.get('gender_filter')
            ))
        
        return SearchResponse(
            success=True,
            final_response=result.get('final_response', 'No response'),
            search_results_data=search_results,
            search_mode=result.get('search_mode'),
            detected_gender=result.get('detected_gender'),
            gender_source=result.get('gender_source'),
            intent_type=result.get('intent_type'),
            messages=result.get('messages', []),
            debug_info=result.get('debug_info')
        )
        
    except Exception as e:
        import traceback
        return SearchResponse(
            success=False,
            final_response=f"Error: {str(e)}",
            messages=[f"❌ Error: {str(e)}", traceback.format_exc()[:500]]
        )
    
    finally:
        # Clean up temp file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass


@router.get("/images/{image_id}")
async def get_image(image_id: str):
    """Serve product images."""
    image_path = os.path.join(config.IMAGES_PATH, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(image_path, media_type="image/jpeg")
