# emotion_recognition_finetuning.py
# Fine-tune HuggingFace model for emotion recognition
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

class EmotionRecognitionTrainer:
    """Fine-tune HuggingFace model for emotion recognition"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.emotion_names = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        self.num_labels = 6
        
        # Create directories
        os.makedirs("models/emotion_classifier", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
    def load_prepared_data(self):
        """Load the prepared emotion dataset"""
        print("📊 Loading prepared emotion dataset...")
        
        # Run dataset prep if not already done
        from dataset_prep import DatasetPreparation
        data_prep = DatasetPreparation()
        emotion_dfs = data_prep.prepare_emotion_dataset()
        
        print(f"✅ Loaded emotion datasets")
        print(f"✅ Train: {len(emotion_dfs['train'])} samples")
        print(f"✅ Val: {len(emotion_dfs['val'])} samples") 
        print(f"✅ Test: {len(emotion_dfs['test'])} samples")
        
        return emotion_dfs
    
    def prepare_datasets(self, emotion_dfs):
        """Prepare datasets for HuggingFace"""
        print("🔧 Preparing emotion datasets for HuggingFace training...")
        
        # Load HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"✅ Loaded tokenizer: {self.model_name}")
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128  # Shorter for emotion texts
            )
        
        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(emotion_dfs['train'][['text', 'label']])
        val_dataset = Dataset.from_pandas(emotion_dfs['val'][['text', 'label']])
        test_dataset = Dataset.from_pandas(emotion_dfs['test'][['text', 'label']])
        
        # Apply tokenization
        print("🔄 Tokenizing emotion datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Rename for HuggingFace compatibility
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
        test_dataset = test_dataset.rename_column("label", "labels")
        
        print("✅ Emotion dataset preparation completed")
        return train_dataset, val_dataset, test_dataset
    
    def fine_tune_model(self, train_dataset, val_dataset):
        """Fine-tune HuggingFace model for emotion recognition"""
        print(f"🚀 Starting emotion recognition fine-tuning with {self.model_name}")
        print("-" * 60)
        
        # Load HuggingFace model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        print(f"✅ Loaded model: {self.model_name}")
        print(f"😊 Configured for {self.num_labels} emotion classes")
        
        # Define metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results/emotion_classifier",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs/emotion",
            logging_steps=25,
            save_steps=1000,
            eval_steps=1000,
            save_total_limit=2,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb
        )
        
        # Initialize HuggingFace Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Start training
        print("🔥 Training started...")
        train_result = self.trainer.train()
        
        # Save the fine-tuned model
        self.trainer.save_model("models/emotion_classifier")
        self.tokenizer.save_pretrained("models/emotion_classifier")
        
        print("✅ Emotion recognition fine-tuning completed!")
        print(f"😊 Final training loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate_model(self, test_dataset):
        """Evaluate the fine-tuned emotion recognition model"""
        print("\n📊 Evaluating emotion recognition model...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"✅ Emotion Recognition Accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.emotion_names))
        
        # Create confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        return accuracy, y_true, y_pred
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot normalized confusion matrix"""
        print("📊 Creating emotion confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = normalize(cm.astype('float'), axis=1, norm='l1')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Reds',
            xticklabels=self.emotion_names,
            yticklabels=self.emotion_names,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title('Emotion Recognition - Normalized Confusion Matrix', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('Actual Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/emotion_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved to results/emotion_confusion_matrix.png")
    
    def test_inference(self, text):
        """Test inference on a single text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet. Please run fine_tune_model first.")
            
        # Create inference pipeline
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Get prediction
        result = classifier(text)[0]
        emotion_id = int(result['label'].split('_')[-1])
        emotion = self.emotion_names[emotion_id]
        confidence = result['score']
        
        return {
            'emotion': emotion,
            'confidence': confidence
        }