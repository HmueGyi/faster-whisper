"""
Robotics Configuration for Faster Whisper
Moshi-like real-time streaming with optimized accuracy/latency trade-offs
"""

# ===== LATENCY OPTIMIZATION =====
# Lower latency suitable for robotics control

LATENCY_PROFILE = {
    "ultra_low": {
        "beam_size": 1,                    # Fastest - single hypothesis
        "chunk_size": 8000,                # 0.5 seconds
        "silence_duration_ms": 100,       # Detect silence quickly
        "speech_pad_ms": 30,              # Minimal padding
        "timeout": 2.0,                   # 2 second timeout
        "buffer_threshold": 16000,        # 1 second max buffer
    },
    "balanced": {
        "beam_size": 2,                    # Medium accuracy
        "chunk_size": 16000,               # 1 second
        "silence_duration_ms": 200,
        "speech_pad_ms": 50,
        "timeout": 3.0,
        "buffer_threshold": 32000,        # 2 seconds
    },
    "accurate": {
        "beam_size": 5,                    # Higher accuracy
        "chunk_size": 32000,               # 2 seconds
        "silence_duration_ms": 300,
        "speech_pad_ms": 100,
        "timeout": 5.0,
        "buffer_threshold": 64000,        # 4 seconds
    }
}

# ===== DEFAULT PROFILE FOR ROBOTICS =====
# "balanced" is best for most robotics applications
DEFAULT_PROFILE = "balanced"

# ===== AUDIO SETTINGS =====
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "format": "int16",
    "chunk_size": 512,                     # 32ms chunks for real-time
    "buffer_accumulation_time": 0.5,      # Accumulate for 500ms before sending
}

# ===== MODEL SETTINGS =====
MODEL_CONFIG = {
    "size": "distil-large-v3",            # Fast & accurate
    "device": "cuda",                     # GPU acceleration
    "compute_type": "int8",               # Quantization for speed
    "download_root": "./models",
}

# ===== ACCURACY ENHANCEMENTS =====
# These help improve accuracy while maintaining low latency

ACCURACY_FEATURES = {
    "vad_enabled": True,                  # Voice Activity Detection
    "vad_threshold": 0.5,                 # Sensitivity (0-1)
    "language": "en",                     # Set language for better accuracy
    "condition_on_previous": False,       # Prevent hallucinations
    "strip_silence": True,                # Remove leading/trailing silence
    "min_audio_length": 4000,             # 0.25 seconds minimum
}

# ===== REAL-TIME STREAMING PARAMETERS =====
STREAMING_CONFIG = {
    "enable_streaming": True,
    "max_concurrent_transcriptions": 1,   # One at a time
    "queue_size": 100,
    "timeout_between_chunks": 3.0,
    "speech_start_threshold": 0.01,       # RMS threshold
    "speech_end_threshold": 0.005,        # Silence RMS
}

# ===== ERROR HANDLING & RECOVERY =====
RESILIENCE = {
    "max_retries": 2,
    "retry_delay": 0.1,
    "fallback_to_higher_beam": True,     # If transcription fails, try beam_size+1
    "filter_empty_results": True,         # Skip silent chunks
}

def get_profile(profile_name=None):
    """Get transcription profile configuration"""
    if profile_name is None:
        profile_name = DEFAULT_PROFILE
    return LATENCY_PROFILE.get(profile_name, LATENCY_PROFILE["balanced"])

def merge_configs():
    """Merge all configurations into a single dict"""
    profile = get_profile(DEFAULT_PROFILE)
    return {
        **AUDIO_CONFIG,
        **MODEL_CONFIG,
        **ACCURACY_FEATURES,
        **STREAMING_CONFIG,
        **RESILIENCE,
        **profile,
    }
