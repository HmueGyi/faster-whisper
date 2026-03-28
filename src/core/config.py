from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Server Configuration
HOST = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT = int(os.getenv("WHISPER_PORT", 8080))

# Model Configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "distil-large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
DOWNLOAD_ROOT = str(BASE_DIR / "models")

# Audio Settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 500  # Process every 500ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 0.003  # RMS energy threshold for speech detection

# WebSocket Settings
MAX_SILENCE_CHUNKS = 6  # ~3 seconds of silence before warning
