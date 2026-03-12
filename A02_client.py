import asyncio
import websockets
import pyaudio
import sys
import json

# Audio Settings - Optimized for real-time Moshi-like streaming
CHUNK = 512  # Smaller chunks for real-time detection (32ms at 16kHz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def send_audio(websocket, stream):
    """
    ⚡ Send audio from mic to server in real-time
    Moshi-like: streams audio continuously for immediate detection
    """
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001)  # Minimal delay for real-time response
    except Exception as e:
        print(f"\n❌ Sender Error: {e}")

async def receive_text(websocket):
    """
    💬 Receive transcriptions from server
    Moshi-like: Gets response immediately after speech ends
    """
    print("-" * 70)
    try:
        while True:
            message = await websocket.recv()
            
            try:
                # Parse JSON response with latency
                response = json.loads(message)
                text = response.get("text", "")
                latency_ms = response.get("latency_ms", 0)
                
                if text.strip():
                    sys.stdout.write(f"\n✅ Transcribed ({latency_ms}ms): {text}\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                # Fallback for plain text
                if message.strip():
                    sys.stdout.write(f"\n✅ {message}\n")
                    sys.stdout.flush()
                
    except Exception as e:
        print(f"\n❌ Receiver Error: {e}")

async def record_and_stream():
    # Server address
    uri = "ws://localhost:8080/ws/transcribe"
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("🎯 Moshi-like ASR - Real-time Speech Detection")
    print("✅ Connected! Listening...")
    print("🎤 Speak naturally. Response comes when you stop speaking.")
    print("-" * 70)

    try:
        async with websockets.connect(uri) as websocket:
            # Send and receive simultaneously for real-time streaming
            await asyncio.gather(
                send_audio(websocket, stream),
                receive_text(websocket)
            )

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped.")
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(record_and_stream())