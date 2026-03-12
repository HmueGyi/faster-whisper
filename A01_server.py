
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import time
import threading
from collections import deque

app = FastAPI()

# Configuration - Optimized for  & real-time
MODEL_SIZE = "distil-large-v3" 
DEVICE = "cuda" 
COMPUTE_TYPE = "int8" 
SAMPLE_RATE = 16000

print(f"Loading model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="./models")

# Load Silero VAD for better speech detection
print("Loading Silero VAD...")
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad()
except ImportError:
    print("Warning: silero_vad not available. Using default VAD.")
    vad_model = None

# Real-time transcription queue for Moshi-like streaming
transcription_queue = deque(maxlen=100)
queue_lock = threading.Lock()

def detect_speech_activity(audio_bytes):
    """
    Detects speech activity in audio chunk.
    Uses VAD filter internally from faster-whisper.
    Returns True if likely contains speech, False if silent.
    """
    try:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Check if RMS energy is above threshold (indicates potential speech)
        rms = np.sqrt(np.mean(audio_float32 ** 2))
        return rms > 0.005  # Lower threshold for continuous speech
    except:
        return False

def transcribe_chunk(audio_bytes):
    """
    Fast, low-latency transcription for .
    Optimized for quick response time over long context.
    """
    try:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        segments, _ = model.transcribe(
            audio_float32, 
            language="en", 
            beam_size=1,  # Speed optimization (beam_size=1 is fastest)
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters=dict(
                min_silence_duration_ms=200,  # Fast silence detection
                speech_pad_ms=50,             # Minimal padding for low latency
                threshold=0.5                 
            ),
            without_timestamps=False,
            language_detection_threshold=0.5,
        )
        
        return " ".join([segment.text.strip() for segment in segments])
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    silence_counter = 0
    speech_detected_count = 0
    last_transcribe_time = time.time()
    
    # -optimized thresholds (Moshi-like streaming)
    CHUNK_THRESHOLD = 32000   # 1 second of audio for ultra-low latency
    SILENCE_THRESHOLD = 4     # 200ms of silence to trigger transcription
    MIN_AUDIO_LENGTH = 8000   # 0.5 seconds minimum
    SPEECH_TIMEOUT = 3        # 3 seconds timeout

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                # Check for speech activity in the received chunk
                has_speech = await run_in_threadpool(detect_speech_activity, data)
                
                if has_speech:
                    # Reset silence counter when speech is detected
                    silence_counter = 0
                    speech_detected_count += 1
                    last_transcribe_time = time.time()
                else:
                    # Increment silence counter only if we've detected speech before
                    if speech_detected_count > 0:
                        silence_counter += 1
                
                # Low-latency transcription triggers
                current_time = time.time()
                time_since_last_speech = current_time - last_transcribe_time
                
                should_transcribe = (
                    (len(audio_buffer) >= CHUNK_THRESHOLD) or  # 1 second accumulated
                    (silence_counter >= SILENCE_THRESHOLD and len(audio_buffer) >= MIN_AUDIO_LENGTH) or  # 200ms silence
                    (time_since_last_speech > SPEECH_TIMEOUT and len(audio_buffer) >= MIN_AUDIO_LENGTH and speech_detected_count > 0)
                )
                
                if should_transcribe and len(audio_buffer) >= MIN_AUDIO_LENGTH:
                    # Process the accumulated audio (non-blocking)
                    text = await run_in_threadpool(transcribe_chunk, bytes(audio_buffer))
                    
                    if text.strip():
                        # Send the transcribed text to the client
                        await websocket.send_text(text)
                        print(f"[Transcribed {len(audio_buffer)}B in {time.time()-current_time:.2f}s] {text}")
                    
                    # Clear the buffer after processing
                    audio_buffer = bytearray()
                    silence_counter = 0
                    speech_detected_count = 0
                    last_transcribe_time = time.time()

            except WebSocketDisconnect:
                print("Client disconnected.")
                break
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
