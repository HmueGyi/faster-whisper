import asyncio
import sys
from src.client.recorder import AudioClient
from src.core import config
from src.core.logger import logger

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Faster-Whisper STT Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=config.PORT, help="Server port")
    parser.add_argument("--language", default="en", help="Transcription language (e.g. 'en', 'my', 'ja')")
    return parser.parse_args()

async def main():
    args = get_args()
    uri = f"ws://{args.host}:{args.port}/ws/transcribe?language={args.language}"
    
    client = AudioClient(uri)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
