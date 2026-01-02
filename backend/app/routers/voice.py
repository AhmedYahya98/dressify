"""
Voice API Router for Speech-to-Text.
Handles audio file uploads, conversion, and transcription.
"""

import os
import tempfile
import shutil
import subprocess
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..services.stt_service import transcribe_audio, initialize_stt

router = APIRouter(prefix="/api/voice", tags=["voice"])


def convert_to_wav(input_path: str) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    Falls back to original if ffmpeg not available.
    """
    wav_path = input_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        # Try ffmpeg conversion
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback: try pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception:
        pass
    
    # Return original if conversion fails
    return input_path


@router.post("/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio to text using Whisper.
    
    Accepts: audio file (webm, wav, mp3, ogg, etc.)
    Returns: { "text": "transcribed text", "success": true }
    """
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Get file extension
    ext = os.path.splitext(audio.filename)[1] or ".webm"
    
    temp_path = None
    wav_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_path = tmp.name
        
        # Convert to WAV if not already
        if ext.lower() != '.wav':
            wav_path = convert_to_wav(temp_path)
        else:
            wav_path = temp_path
        
        # Transcribe
        text = transcribe_audio(wav_path)
        
        if text is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "text": "", "error": "Transcription failed. Make sure ffmpeg is installed."}
            )
        
        return {"success": True, "text": text}
    
    except Exception as e:
        print(f"❌ Voice API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "text": "", "error": str(e)}
        )
    
    finally:
        # Clean up temp files
        for path in [temp_path, wav_path]:
            if path and os.path.exists(path) and path != wav_path:
                try:
                    os.unlink(path)
                except:
                    pass


@router.get("/status")
async def stt_status():
    """Check if STT is initialized and ready."""
    from ..services.stt_service import get_stt_pipeline
    pipeline_obj = get_stt_pipeline()
    
    # Check ffmpeg availability
    ffmpeg_available = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        ffmpeg_available = result.returncode == 0
    except FileNotFoundError:
        pass
    
    return {
        "ready": pipeline_obj is not None,
        "model": "whisper-small" if pipeline_obj else None,
        "ffmpeg": ffmpeg_available
    }
