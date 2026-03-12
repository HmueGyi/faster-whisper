import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import time

app = FastAPI()

# Configuration
MODEL_SIZE = "large-v3-turbo" 
DEVICE = "cuda" 
COMPUTE_TYPE = "float16" 
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
        return rms > 0.01  # Threshold for speech detection
    except:
        return False

def transcribe_streaming_audio(audio_bytes):
    """
    Transcribes longer audio chunks with optimized settings for continuous speech.
    """
    # Convert buffer to float32
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    segments, _ = model.transcribe(
        audio_float32, 
        language="en", 
        beam_size=5,  # Increased for better accuracy on longer sentences
        vad_filter=True,
        # Disable context conditioning to prevent repetitions
        condition_on_previous_text=False,
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Longer silence tolerance for natural pauses
            speech_pad_ms=150,            # More padding for better context
            threshold=0.5                 # Slightly lower threshold for continuous speech
        ),
        without_timestamps=False,
        language_detection_threshold=0.5,
    )
    
    return " ".join([segment.text.strip() for segment in segments])

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    silence_counter = 0
    speech_detected_count = 0
    last_transcribe_time = time.time()
    
    # Increased thresholds for longer continuous speech (5 seconds = 160000 bytes)
    CHUNK_THRESHOLD = 160000  # 5 seconds of audio for longer sentences
    # Silence threshold for end of sentence detection (1 second of silence)
    SILENCE_THRESHOLD = 20  # 20 chunks of ~50ms = ~1000ms (requires 1sec silence to stop)
    # Minimum pause for word break (0.3 seconds)
    MIN_PAUSE_THRESHOLD = 6  # For internal pauses between words
    # Minimum audio to process
    MIN_AUDIO_LENGTH = 16000  # Minimum 1 second
    # Timeout to finalize if no new speech (10 seconds)
    SPEECH_TIMEOUT = 10

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
                
                # Trigger transcription when:
                # 1. Buffer reaches size threshold (5 seconds) - for very long speeches
                # 2. We detect long silence (1 second) after speech - end of sentence
                # 3. Timeout reached without new speech
                current_time = time.time()
                time_since_last_speech = current_time - last_transcribe_time
                
                should_transcribe = (
                    (len(audio_buffer) >= CHUNK_THRESHOLD) or  # 5 seconds accumulated
                    (silence_counter >= SILENCE_THRESHOLD and len(audio_buffer) >= MIN_AUDIO_LENGTH and speech_detected_count > 0) or  # 1 second silence detected
                    (time_since_last_speech > SPEECH_TIMEOUT and len(audio_buffer) >= MIN_AUDIO_LENGTH and speech_detected_count > 0)  # Timeout
                )
                
                if should_transcribe and len(audio_buffer) >= MIN_AUDIO_LENGTH:
                    # Process the accumulated audio
                    text = await run_in_threadpool(transcribe_streaming_audio, bytes(audio_buffer))
                    
                    if text.strip():
                        # Send the transcribed text to the client
                        await websocket.send_text(text)
                        print(f"[Transcribed - {len(audio_buffer)} bytes] {text}")
                    
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