import numpy as np
from faster_whisper import WhisperModel
from src.core import config
from src.core.logger import logger

class TranscriptionEngine:
    def __init__(self):
        logger.info(f"Loading model: {config.MODEL_SIZE} on {config.DEVICE}...")
        self.model = WhisperModel(
            config.MODEL_SIZE, 
            device=config.DEVICE, 
            compute_type=config.COMPUTE_TYPE, 
            download_root=config.DOWNLOAD_ROOT
        )
        logger.info("Model loaded successfully.")

    @staticmethod
    def calculate_energy(audio_bytes: bytes) -> float:
        """Calculate RMS energy of audio chunk"""
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return np.sqrt(np.mean(audio_float ** 2))

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """
        Streaming transcription with optimized accuracy settings.
        """
        try:
            audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            segments, _ = self.model.transcribe(
                audio_float,
                language=language,
                beam_size=5,  # Efficient balance
                vad_filter=True,
                condition_on_previous_text=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=100,
                    threshold=0.5
                ),
                without_timestamps=True,
            )
            
            text = " ".join([segment.text.strip() for segment in segments])
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
