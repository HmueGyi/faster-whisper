import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, Response
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

# Configuration
MODEL_SIZE = "distil-large-v3"
DEVICE = "cuda" 
COMPUTE_TYPE = "float16" 

print(f"Loading model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="./models")

def transcribe_streaming_audio(audio_bytes):
    """
    Real-time streaming အတွက် logic ။
    """
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Beam size ကို ၁ ထားတာက streaming အတွက် အမြန်ဆုံးဖြစ်စေပါတယ်
    segments, _ = model.transcribe(
        audio_float32, 
        language="en", 
        beam_size=5,             # 1 ကနေ 5 ကို တိုးလိုက်တာပါ
        best_of=5,               # Candidates ပိုရှာခိုင်းမယ်
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return " ".join([segment.text.strip() for segment in segments])

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # audio_buffer က အသံတွေ စုဆောင်းထားဖို့
    audio_buffer = bytearray()
    
    # စမ်းသပ်ချက်အရ ၁.၅ စက္ကန့်စာ chunk က streaming အတွက် အဆင်ပြေဆုံးပါ
    # 16000 (rate) * 2 (bytes) * 1.5 (sec) = 48000 bytes
    CHUNK_THRESHOLD = 48000 

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                if len(audio_buffer) >= CHUNK_THRESHOLD:
                    # လက်ရှိ buffer ထဲက အသံကို transcribe လုပ်မယ်
                    text = await run_in_threadpool(transcribe_streaming_audio, bytes(audio_buffer))
                    
                    if text.strip():
                        # စာသားထွက်လာရင် client ဆီ ချက်ချင်းပို့မယ်
                        await websocket.send_text(text)
                    
                    # Streaming style မှာ အရေးကြီးဆုံးက 'Sliding Window' ပါ။
                    # Buffer အကုန်မဖျက်ဘဲ နောက်ဆုံး 0.5 စက္ကန့်စာကို ချန်ထားမှ 
                    # စကားလုံးတွေ ပြတ်မသွားဘဲ context ဆက်မိမှာဖြစ်ပါတယ်။
                    # 0.5 sec = 16000 bytes
                    audio_buffer = audio_buffer[-16000:] 

            except WebSocketDisconnect:
                print("Client disconnected.")
                break
    except Exception as e:
        print(f"WebSocket Error: {e}")

# ... (transcribe_post function နဲ့ uvicorn run တာကို အရင်အတိုင်းထားနိုင်ပါတယ်) ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)