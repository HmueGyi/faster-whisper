import asyncio
import websockets
import pyaudio
import sys
from src.core import config
from src.core.logger import logger

class AudioClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = config.SAMPLE_RATE

    def start_stream(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        logger.info(f"🎙️ Microphones successfully opened at {self.rate}Hz.")

    async def send_audio(self, websocket):
        """Send audio from microphone to server (streaming)"""
        try:
            while True:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                await websocket.send(data)
                await asyncio.sleep(0.001)  # Small delay for streaming
        except Exception as e:
            logger.error(f"Sender Error: {e}")

    async def receive_text(self, websocket):
        """Receive transcribed text from server (real-time)"""
        try:
            while True:
                message = await websocket.recv()
                if message.strip():
                    sys.stdout.write(f"\r✅ {message} ")
                    sys.stdout.flush()
        except Exception as e:
            logger.error(f"Receiver Error: {e}")

    async def run(self):
        self.start_stream()
        logger.info("-" * 60)
        logger.info(f"Connecting to {self.uri}...")

        try:
            async with websockets.connect(self.uri) as websocket:
                logger.info("Connected! Start speaking...")
                await asyncio.gather(
                    self.send_audio(websocket),
                    self.receive_text(websocket)
                )
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopped by user.")
        except Exception as e:
            logger.error(f"❌ Connection Error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
