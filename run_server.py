import uvicorn
import os
import sys
import platform
from src.core import config
from src.core.logger import logger

def setup_cuda_windows():
    """Add NVIDIA libraries to the DLL search path on Windows."""
    if platform.system() == "Windows" and config.DEVICE == "cuda":
        try:
            import nvidia.cublas as cb
            import nvidia.cudnn as cn
            
            # Use os.add_dll_directory if available (Python 3.8+)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(os.path.join(cb.__path__[0], "bin"))
                os.add_dll_directory(os.path.join(cn.__path__[0], "bin"))
                logger.info("Added NVIDIA libraries to DLL search path.")
            else:
                os.environ["PATH"] = os.path.join(cb.__path__[0], "bin") + os.pathsep + \
                                     os.path.join(cn.__path__[0], "bin") + os.pathsep + \
                                     os.environ["PATH"]
                logger.info("Updated PATH with NVIDIA libraries.")
        except ImportError:
            logger.warning("NVIDIA libraries not found. Ensure CUDA is installed manually.")

if __name__ == "__main__":
    setup_cuda_windows()
    logger.info("Starting Faster-Whisper STT Server...")
    uvicorn.run("src.server.main:app", host=config.HOST, port=config.PORT, reload=False)
