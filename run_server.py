import uvicorn
from src.core import config
from src.core.logger import logger

if __name__ == "__main__":
    logger.info("Starting Faster-Whisper STT Server...")
    uvicorn.run("src.server.main:app", host=config.HOST, port=config.PORT, reload=False)
