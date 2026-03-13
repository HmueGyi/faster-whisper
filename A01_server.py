import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import time

app = FastAPI()

# Configuration - Optimized for streaming & real-time
MODEL_SIZE = "distil-large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
SAMPLE_RATE = 16000

print(f"Loading model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="./models")

# Streaming configuration for low latency
CHUNK_DURATION_MS = 500  # Process every 500ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 0.003  # RMS energy threshold for speech detection

def get_energy(audio_bytes):
    """Calculate RMS energy of audio chunk"""
    audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.sqrt(np.mean(audio_float ** 2))

def transcribe_streaming(audio_bytes, language="en"):
    """
    Streaming transcription with improved accuracy.
    Uses beam_size=5 for better accuracy vs speed trade-off.
    """
    try:
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, _ = model.transcribe(
            audio_float,
            language=language,
            beam_size=5,  # Balance between accuracy and speed
            vad_filter=True,
            condition_on_previous_text=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
                threshold=0.5
            ),
            without_timestamps=True,
        )
        
        return " ".join([segment.text.strip() for segment in segments])
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    last_transcribe_time = time.time()
    consecutive_silence = 0

    try:
        print("Client connected for streaming transcription...")
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)
                
                # Check if we have enough audio to process (500ms)
                if len(audio_buffer) >= CHUNK_SIZE * 2:
                    # Get energy to detect silence
                    energy = await run_in_threadpool(get_energy, bytes(audio_buffer))
                    
                    if energy < SILENCE_THRESHOLD:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0
                    
                    # Transcribe when we have enough audio
                    text = await run_in_threadpool(transcribe_streaming, bytes(audio_buffer))
                    
                    if text.strip():
                        await websocket.send_text(text)
                        print(f"✅ {text}")
                        consecutive_silence = 0
                    
                    # Clear buffer after processing
                    audio_buffer = bytearray()
                    last_transcribe_time = time.time()
                    
                    # Stop if long silence detected
                    if consecutive_silence > 6:  # ~3 seconds of silence
                        print("⏸️ Long silence detected - waiting for speech...")
                        consecutive_silence = 0

            except WebSocketDisconnect:
                print("Client disconnected.")
                break
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
