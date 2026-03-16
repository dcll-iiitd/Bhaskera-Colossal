"""
bhaskera.audio.pipelines
-------------------------
Built-in ASR pipeline implementations.

Importing this package does NOT load any heavy HF model weights.
The registry loads pipelines lazily on first call to get_asr_pipeline().

Available pipelines
-------------------
  WhisperASRPipeline   — openai/whisper-*   (Seq2Seq, WER metric)
  Wav2Vec2ASRPipeline  — facebook/wav2vec2-* (CTC, WER metric)
  MMSASRPipeline       — facebook/mms-*      (CTC adapter, WER metric)

Custom pipeline
---------------
  from bhaskera.audio.pipelines.base import BaseASRPipeline
  from bhaskera.audio import register_asr_model

  class MyPipeline(BaseASRPipeline): ...
  register_asr_model("myfamily", MyPipeline)
"""
from bhaskera.audio.pipelines.base import BaseASRPipeline
from bhaskera.audio.pipelines.whisper_pipeline import WhisperASRPipeline
from bhaskera.audio.pipelines.wav2vec2_pipeline import Wav2Vec2ASRPipeline
from bhaskera.audio.pipelines.mms_pipeline import MMSASRPipeline

__all__ = [
    "BaseASRPipeline",
    "WhisperASRPipeline",
    "Wav2Vec2ASRPipeline",
    "MMSASRPipeline",
]
