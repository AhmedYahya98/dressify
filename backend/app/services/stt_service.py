"""
Speech-to-Text Service using Whisper model.
Handles audio transcription for voice input with preprocessing.
"""

import os
import re
import wave
import numpy as np
from typing import Optional, Tuple
from transformers import pipeline

from ..core.config import config

# Global STT pipeline
_stt_pipeline = None
_stt_initialized = False

# Audio save directory
AUDIO_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_audio")


def initialize_stt():
    """Initialize Whisper STT pipeline."""
    global _stt_pipeline, _stt_initialized
    
    if _stt_initialized:
        return _stt_pipeline
    
    print("\n🎤 Initializing Whisper STT...")
    
    # Create audio save directory
    os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    
    try:
        _stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL,
            device=0 if config.DEVICE == "cuda" else -1,
            chunk_length_s=30,
        )
        print(f"✅ Whisper STT: {config.WHISPER_MODEL}")
        _stt_initialized = True
    except Exception as e:
        print(f"⚠️ Whisper STT failed: {e}")
        _stt_pipeline = None
        _stt_initialized = True
    
    return _stt_pipeline


def load_wav_file(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load WAV file using pure Python (no ffmpeg required).
    Returns audio data as numpy array and sample rate.
    """
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        
        # Read raw audio data
        raw_data = wav_file.readframes(n_frames)
        
        # Convert to numpy array based on sample width
        if sample_width == 1:
            audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        elif n_channels > 2:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        
        return audio, sample_rate


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Simple linear resampling (good enough for STT)."""
    if orig_sr == target_sr:
        return audio
    
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


def preprocess_transcription(text: str) -> str:
    """
    Clean and preprocess transcribed text.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove common filler words at start
    filler_patterns = [
        r'^(um+|uh+|er+|ah+|hmm+|well|so|like|okay|ok)\s*,?\s*',
        r'^(hey|hi|hello)\s+(there|siri|alexa|google)\s*,?\s*',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove repeated words (stuttering)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text


def transcribe_audio(audio_path: str, save_audio: bool = True) -> Optional[str]:
    """
    Transcribe audio file to text using Whisper.
    """
    global _stt_pipeline
    
    if _stt_pipeline is None:
        initialize_stt()
    
    if _stt_pipeline is None:
        return None
    
    try:
        # Save audio if requested
        if save_audio and os.path.exists(audio_path):
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(audio_path)[1]
            save_path = os.path.join(AUDIO_SAVE_DIR, f"audio_{timestamp}{ext}")
            shutil.copy2(audio_path, save_path)
            print(f"💾 Saved audio: {save_path}")
        
        # Load WAV file using pure Python
        if audio_path.lower().endswith('.wav'):
            audio, sample_rate = load_wav_file(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = resample_audio(audio, sample_rate, 16000)
            
            # Run transcription with raw audio
            result = _stt_pipeline({"raw": audio, "sampling_rate": 16000})
        else:
            # For other formats, try direct loading (requires ffmpeg)
            result = _stt_pipeline(audio_path)
        
        # Extract text
        if isinstance(result, dict) and 'text' in result:
            raw_text = result['text'].strip()
        else:
            raw_text = str(result).strip()
        
        # Preprocess the transcription
        processed_text = preprocess_transcription(raw_text)
        
        print(f"📝 Raw: {raw_text}")
        print(f"✅ Processed: {processed_text}")
        
        return processed_text
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_stt_pipeline():
    """Get the STT pipeline instance."""
    global _stt_pipeline, _stt_initialized
    if not _stt_initialized:
        initialize_stt()
    return _stt_pipeline
