import asyncio
import websockets
import pyaudio
import sys

# Audio Settings - Optimized for longer continuous speech
CHUNK = 1024  # Larger chunks to accumulate more audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def send_audio(websocket, stream):
    """Mic ကနေ အသံဖမ်းပြီး Server ဆီ တောက်လျှောက် ပို့နေမယ် (အရည်ကြီးစကား)"""
    try:
        while True:
            # Mic ကနေ data ဖတ်မယ်
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Server ဆီ ပို့မယ်
            await websocket.send(data)
            # loop မပိတ်အောင် asyncio ကို အသက်ရှူချိန်ပေးမယ်
            await asyncio.sleep(0.01) 
    except Exception as e:
        print(f"\nSender Error: {e}")

async def receive_text(websocket):
    """Server က စာသားအသစ်ပို့တာနဲ့ ဘေးတိုက် တန်းစီပြီး ပြပေးမယ်"""
    print("-" * 60)
    try:
        while True:
            # Server က စာသားအသစ်ကို စောင့်မယ်
            message = await websocket.recv()
            
            if message.strip():
                # စာသားများကို တောက်လျှောက် ပြပေးမယ်
                sys.stdout.write(f"\n✅ test: {message}\n")
                sys.stdout.flush()
                
    except Exception as e:
        print(f"\nReceiver Error: {e}")

async def record_and_stream():
    # Server address (Localhost)
    uri = "ws://localhost:8080/ws/transcribe"
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("✅ Connected! Starting audio stream...")
    print("🎙️  Speaking... (Press Ctrl+C to stop)")
    print("-" * 60)

    try:
        async with websockets.connect(uri) as websocket:
            # ပို့တဲ့ task နဲ့ လက်ခံတဲ့ task ကို ပြိုင်တူ run မယ်
            await asyncio.gather(
                send_audio(websocket, stream),
                receive_text(websocket)
            )

    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping...")
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(record_and_stream())