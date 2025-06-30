import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List

class UnifiedAnalysisService:
    def __init__(self,
                 emotion_model_path: str = "models/emotion_classifier_corrected",
                 topic_model_path: str = "models/topic_classifier"):
        self.emotion_model_path = emotion_model_path
        self.topic_model_path = topic_model_path
        self.emotion_pipeline = None
        self.topic_pipeline = None
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.topic_labels = [
            'Cancer', 'Diabetes', 'Disease_Control_and_Prevention',
            'Genetic_and_Rare_Diseases', 'Growth_Hormone_Receptor',
            'Heart_Lung_and_Blood', 'Neurological_Disorders_and_Stroke',
            'Senior_Health', 'Others'
        ]
        self._load_models()

    def _load_models(self):
        # Emotion model
        self.emotion_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(self.emotion_model_path),
            tokenizer=AutoTokenizer.from_pretrained(self.emotion_model_path),
            top_k=None,
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1
        )
        # Topic model
        self.topic_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(self.topic_model_path),
            tokenizer=AutoTokenizer.from_pretrained(self.topic_model_path),
            top_k=None,
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1
        )

    def get_emotion(self, text: str) -> Dict[str, Any]:
        preds = self.emotion_pipeline(text, top_k=3)[0] if isinstance(self.emotion_pipeline(text, top_k=3), list) else self.emotion_pipeline(text, top_k=3)
        top_3 = [
            {
                'label': self.emotion_labels[int(p['label'].split('_')[-1])] if 'LABEL_' in p['label'] else self.emotion_labels[int(p['label'])],
                'confidence': float(p['score'])
            } for p in preds
        ]
        primary = top_3[0]['label']
        confidence = top_3[0]['confidence']
        return {
            'primary': primary,
            'confidence': confidence,
            'top_3': top_3
        }

    def get_topic(self, text: str) -> Dict[str, Any]:
        preds = self.topic_pipeline(text, top_k=3)[0] if isinstance(self.topic_pipeline(text, top_k=3), list) else self.topic_pipeline(text, top_k=3)
        top_3 = [
            {
                'label': self.topic_labels[int(p['label'].split('_')[-1])] if 'LABEL_' in p['label'] else self.topic_labels[int(p['label'])],
                'confidence': float(p['score'])
            } for p in preds
        ]
        primary = top_3[0]['label']
        confidence = top_3[0]['confidence']
        return {
            'primary': primary,
            'confidence': confidence,
            'top_3': top_3
        }

    def analyze_user_input(self, text: str) -> Dict[str, Any]:
        return {
            'emotion': self.get_emotion(text),
            'topic': self.get_topic(text)
        }

    def analyze_message(self, text: str) -> Dict[str, Any]:
        return self.analyze_user_input(text)

# Convenience functions
analyzer = UnifiedAnalysisService()
def get_emotion(text):
    return analyzer.get_emotion(text)
def get_topic(text):
    return analyzer.get_topic(text)
def analyze_message(text):
    return analyzer.analyze_user_input(text) 