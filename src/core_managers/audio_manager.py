import asyncio

import whisperx


# from silero import silero_stt, silero_tts, silero_te


class AudioManager:
    def __init__(self, language: str):
        self.device = "cpu"  # No support for mps yet
        self.compute_type = "float32"  # no support for float16 yet
        self.whisper_model = "base"
        self.sample_rate = 48000
        self.language = language

        # self.silero_model = silero_tts(
        #     language='en',
        #     device=self.device,
        #     model_name="silero_tts",
        #     sample_rate=self.sample_rate
        # )

        self.whisper_model = whisperx.load_model(
            self.whisper_model,
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

    #
    # def synthesize(self, text: str, output_path: str) -> None:
    #     audio = self.silero_model.apply_tts(text=text, speaker='en_0')
    #     torchaudio.save(output_path, audio.unsqueeze(0), self.sample_rate)
    #
    # async def synthesize_speech_silero(self, text: str, output_path: str) -> None:
    #     await asyncio.to_thread(self.synthesize, text, output_path)
