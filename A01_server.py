import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

# Configuration
MODEL_SIZE = "large-v3-turbo" 
DEVICE = "cuda" 
COMPUTE_TYPE = "float16" 

print(f"Loading model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="./models")

def transcribe_streaming_audio(audio_bytes):
    """
    Transcribes a chunk of audio bytes.
    """
    # Convert buffer to float32
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    segments, _ = model.transcribe(
        audio_float32, 
        language="en", 
        beam_size=5,
        vad_filter=True,
        # IMPORTANT: Disable context conditioning to stop repetitions
        condition_on_previous_text=False,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    return " ".join([segment.text.strip() for segment in segments])

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    # 1.5 seconds threshold (16000Hz * 2 bytes * 1.5s)
    CHUNK_THRESHOLD = 48000 

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                if len(audio_buffer) >= CHUNK_THRESHOLD:
                    # Process the current chunk
                    text = await run_in_threadpool(transcribe_streaming_audio, bytes(audio_buffer))
                    
                    if text.strip():
                        # Send the transcribed text to the client
                        await websocket.send_text(text)
                        # FIX: Clear the buffer entirely after a successful result 
                        # to prevent the "sliding window" from repeating words.
                        audio_buffer = bytearray()
                    else:
                        # If no speech was detected (silence), we keep a tiny bit 
                        # of the end to ensure we don't cut a word starting right now.
                        audio_buffer = audio_buffer[-4000:] 

            except WebSocketDisconnect:
                print("Client disconnected.")
                break
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)