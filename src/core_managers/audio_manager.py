import asyncio
from typing import List

import torch
import torchaudio
import whisperx

from src.utils.enums import AudioConfig

audio_config = AudioConfig()

from silero import silero_tts


class AudioManager:
    def __init__(self, language: str = None, whisper_model_name: str = None):
        self.device = audio_config.DEVICE
        self.compute_type = audio_config.COMPUTE_TYPE
        if whisper_model_name is None:
            self.whisper_model_name = audio_config.DEFAULT_WHISPER_MODEL
        else:
            self.whisper_model_name = whisper_model_name
        self.sample_rate = 48000
        if language is None:
            self.language = audio_config.DEFAULT_LANGUAGE
        else:
            self.language = language

        self.whisper_model = whisperx.load_model(
            self.whisper_model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.whisper_align_model, self.whisper_metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )

    def transcribe(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        result_aligned = whisperx.align(
            result["segments"],
            self.whisper_align_model,
            self.whisper_metadata,
            audio_path,
            device=self.device,
        )
        return " ".join(segment["text"] for segment in result_aligned["segments"])

    async def atranscribe(self, audio_path: str) -> str:
        return await asyncio.to_thread(self.transcribe, audio_path)

    def _split_text(self, text: str, max_length: int = 30) -> List:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(" ".join(current_chunk + [word])) <= max_length:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def text_to_speech(self, text: str, output_path: str) -> None:
        model, sample_rate = silero_tts(
            speaker="v3_en",
            sample_rate=self.sample_rate,
            model_name="silero_tts",
            device=self.device,
            language="en",
        )

        chunks = self._split_text(text)
        audio_list = []

        for chunk in chunks:
            try:
                audio_chunk = model.apply_tts(chunk, speaker="en_0")
                audio_list.append(audio_chunk)
            except Exception as e:
                print(f"Skipping chunk due to error: {e}")

        if audio_list:
            full_audio = torch.cat(audio_list)
            torchaudio.save(output_path, full_audio.unsqueeze(0), self.sample_rate)
        else:
            raise ValueError("No audio was generated from the input text.")

    # def text_to_speech(self, text: str, output_path: str) -> None:
    #     model, sample_rate = silero_tts(
    #         speaker="v3_en",
    #         sample_rate=self.sample_rate,
    #         model_name="silero_tts",
    #         device=self.device,
    #         language="en",
    #     )
    #
    #     audio = model.apply_tts(text, speaker="en_0")
    #     torchaudio.save(output_path, audio.unsqueeze(0), self.sample_rate)

    async def atext_to_speech(self, text: str, output_path: str) -> None:
        return await asyncio.to_thread(self.text_to_speech, text, output_path)
