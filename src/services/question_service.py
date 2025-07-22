import torch
import asyncio

from concurrent.futures import ThreadPoolExecutor
from src.ml_models import flan_t5
from src.utils.enums import QuestionRecommendConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class QuestionService:
    def __init__(self):
        self.device = QuestionRecommendConfig.DEVICE
        self.model_name = QuestionRecommendConfig.MODEL_NAME
        self.model_path = QuestionRecommendConfig.MODEL_PATH
        self.topic_num: int = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.max_length = 128

    def load_model(self, topic: int=0):
        model_weigths = QuestionRecommendConfig.MODEL_SAVE_NAME[topic]
        model_weights_path = self.model_path / model_weigths
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        state_dict = torch.load(model_weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    async def async_load_model(self, topic: int = 0):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_model, topic)

    def predict(self, input_question: str, model) -> list[str]:
        inputs = self.tokenizer(
            input_question,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
            )

        model_output = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return model_output[0].split("|")





