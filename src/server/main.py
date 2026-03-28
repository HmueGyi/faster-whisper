import time
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from src.core import config
from src.core.logger import logger
from src.server.engine import TranscriptionEngine

app = FastAPI(title="Faster-Whisper STT Server")

# Global engine instance
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = TranscriptionEngine()
    return _engine

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket, language: str = "en"):
    await websocket.accept()
    
    engine = get_engine()
    audio_buffer = bytearray()
    consecutive_silence = 0

    try:
        logger.info(f"Client connected for streaming transcription (Language: {language})...")
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)
                
                # Check if we have enough audio to process (500ms)
                if len(audio_buffer) >= config.CHUNK_SIZE * 2:
                    # Get energy to detect silence
                    energy = await run_in_threadpool(engine.calculate_energy, bytes(audio_buffer))
                    
                    if energy < config.SILENCE_THRESHOLD:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0
                    
                    # Transcribe when we have enough audio
                    text = await run_in_threadpool(engine.transcribe, bytes(audio_buffer), language)
                    
                    if text.strip():
                        await websocket.send_text(text)
                        logger.info(f"✅ {text}")
                        consecutive_silence = 0
                    
                    # Clear buffer after processing
                    audio_buffer = bytearray()
                    
                    # Log long silence
                    if consecutive_silence > config.MAX_SILENCE_CHUNKS:
                        logger.info("⏸️ Silence detected - waiting for speech...")
                        consecutive_silence = 0

            except WebSocketDisconnect:
                logger.info("Client disconnected.")
                break
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}")
