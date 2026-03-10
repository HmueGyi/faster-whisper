import asyncio
import websockets
import pyaudio
import sys

# Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def send_audio(websocket, stream):
    """Mic ကနေ အသံဖမ်းပြီး Server ဆီ တောက်လျှောက် ပို့နေမယ်"""
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
    print("Transcription: ", end="", flush=True) 
    try:
        while True:
            # Server က စာသားအသစ်ကို စောင့်မယ်
            message = await websocket.recv()
            
            if message.strip():
                # \r မသုံးတော့ဘဲ space လေးခြားပြီး ဘေးတိုက် ဆက်သွားမယ်
                # ဒါမှ စာသားတွေက ရိုက်စက်လိုမျိုး တောက်လျှောက် စီးဆင်းနေမှာပါ
                sys.stdout.write(f"{message} ")
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

    print("--- Connected! Audio streaming is live. ---")

    try:
        async with websockets.connect(uri) as websocket:
            # ပို့တဲ့ task နဲ့ လက်ခံတဲ့ task ကို ပြိုင်တူ run မယ်
            await asyncio.gather(
                send_audio(websocket, stream),
                receive_text(websocket)
            )

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nConnection Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(record_and_stream())