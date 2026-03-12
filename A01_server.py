import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import time
import torch

app = FastAPI()

# Configuration - Moshi-like streaming ASR
MODEL_SIZE = "distil-large-v3"  # Fast + accurate
DEVICE = "cuda" 
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000

print(f"Loading model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="./models")

# Load Silero VAD for Moshi-like speech detection
print("Loading Silero VAD...")
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad(model="silero_vad", jit=True, device=DEVICE)
    print("✓ Silero VAD loaded successfully")
except ImportError:
    print("Warning: silero_vad not available. Using RMS-based VAD.")
    vad_model = None

def detect_speech_activity(audio_bytes):
    """
    Moshi-like VAD: Detects if audio chunk contains speech using Silero VAD.
    Returns True if speech detected, False if silent.
    """
    try:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Use Silero VAD if available (more accurate than RMS)
        if vad_model is not None:
            with torch.no_grad():
                # Silero VAD expects audio as torch tensor
                audio_tensor = torch.from_numpy(audio_float32).to(DEVICE)
                confidence = vad_model(audio_tensor, SAMPLE_RATE).item()
                return confidence > 0.5  # Speech confidence threshold
        else:
            # Fallback to RMS-based detection
            rms = np.sqrt(np.mean(audio_float32 ** 2))
            return rms > 0.015
    except Exception as e:
        print(f"VAD Error: {e}")
        return False

def transcribe_streaming_audio(audio_bytes, start_time=None):
    """
    Moshi-like transcription: Fast transcription when silence is detected.
    Returns (text, latency_ms)
    """
    try:
        # Convert buffer to float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        transcribe_start = time.time()
        
        segments, _ = model.transcribe(
            audio_float32, 
            language="en", 
            beam_size=1,  # Speed optimized
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters=dict(
                min_silence_duration_ms=200,  # Moshi-like: quick silence detection
                speech_pad_ms=50,             # Minimal padding for fast response
                threshold=0.6
            ),
            without_timestamps=False,
        )
        
        text = " ".join([segment.text.strip() for segment in segments])
        latency_ms = (time.time() - transcribe_start) * 1000
        
        return text, latency_ms
    except Exception as e:
        print(f"Transcription Error: {e}")
        return "", 0

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    Moshi-like streaming ASR endpoint:
    - Detects speech activity in real-time
    - Transcribes when silence is detected (end of utterance)
    - Sends response immediately
    """
    await websocket.accept()
    print("✓ Client connected")
    
    audio_buffer = bytearray()
    silence_frame_count = 0
    speech_detected = False
    speech_start_time = None
    
    # Moshi-like parameters
    SILENCE_FRAMES_THRESHOLD = 8  # ~400ms of silence triggers transcription (50ms chunks)
    MIN_SPEECH_DURATION = 5  # At least 250ms of speech required
    MAX_SPEECH_LENGTH = 240000  # Max 15 seconds
    FRAME_SIZE = 512  # samples per frame at 16kHz ≈ 32ms

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)
                
                # Real-time speech detection
                has_speech = await run_in_threadpool(detect_speech_activity, data)
                
                if has_speech:
                    # Speech detected
                    if not speech_detected:
                        speech_detected = True
                        speech_start_time = time.time()
                        silence_frame_count = 0
                        print(f"🎤 Speech detected, recording...")
                    else:
                        # Reset silence counter during active speech
                        silence_frame_count = 0
                else:
                    # Silence/background noise detected
                    if speech_detected:
                        silence_frame_count += 1
                        
                        # Check if we should trigger transcription
                        if silence_frame_count >= SILENCE_FRAMES_THRESHOLD:
                            # Transcribe when silence is detected after speech
                            if len(audio_buffer) >= MIN_SPEECH_DURATION * SAMPLE_RATE // 1000:
                                print(f"⏸️  Silence detected, transcribing {len(audio_buffer)} bytes...")
                                
                                text, latency = await run_in_threadpool(
                                    transcribe_streaming_audio, 
                                    bytes(audio_buffer)
                                )
                                
                                if text.strip():
                                    # Send result with latency info
                                    import json
                                    response = {
                                        "text": text.strip(),
                                        "latency_ms": int(latency),
                                        "timestamp": time.time()
                                    }
                                    await websocket.send_text(json.dumps(response))
                                    print(f"✓ Response: {text} (latency: {latency:.0f}ms)")
                                
                                # Reset for next utterance
                                audio_buffer = bytearray()
                                silence_frame_count = 0
                                speech_detected = False
                                speech_start_time = None
                
                # Safety: reset if buffer gets too long
                if len(audio_buffer) >= MAX_SPEECH_LENGTH:
                    print(f"⚠️  Max buffer reached, transcribing...")
                    text, latency = await run_in_threadpool(
                        transcribe_streaming_audio,
                        bytes(audio_buffer)
                    )
                    if text.strip():
                        import json
                        response = {
                            "text": text.strip(),
                            "latency_ms": int(latency),
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(response))
                    audio_buffer = bytearray()
                    silence_frame_count = 0
                    speech_detected = False

            except WebSocketDisconnect:
                print("❌ Client disconnected")
                break
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)