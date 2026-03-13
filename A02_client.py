import asyncio
import websockets
import pyaudio
import sys

# Audio Settings - Optimized for streaming
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def send_audio(websocket, stream):
    """Send audio from microphone to server (streaming)"""
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001)  # Small delay for streaming
    except Exception as e:
        print(f"Sender Error: {e}")

async def receive_text(websocket):
    """Receive transcribed text from server (real-time)"""
    try:
        while True:
            message = await websocket.recv()
            if message.strip():
                sys.stdout.write(f"{message} ")
                sys.stdout.flush()
    except Exception as e:
        print(f"Receiver Error: {e}")

async def main():
    uri = "ws://localhost:8080/ws/transcribe"
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("\n🎙️  Streaming Real-time STT (Press Ctrl+C to stop)")
    print("-" * 60)

    try:
        async with websockets.connect(uri) as websocket:
            await asyncio.gather(
                send_audio(websocket, stream),
                receive_text(websocket)
            )
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped!")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(main())