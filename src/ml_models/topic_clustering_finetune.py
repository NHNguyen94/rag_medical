# topic_classification_finetuning.py
# Fine-tune HuggingFace model for medical topic classification

import pandas as pd
import numpy as np
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
import os
import warnings
warnings.filterwarnings("ignore")

class TopicClassificationTrainer:
    """Fine-tune HuggingFace model for medical topic classification"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.topic_names = []
        
        # Create directories
        os.makedirs("models/topic_classifier", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
    def load_prepared_data(self):
        """Load the prepared medical dataset"""
        print("📊 Loading prepared medical dataset...")
        
        # Run dataset prep if not already done
        from dataset_prep import DatasetPreparation
        data_prep = DatasetPreparation()
        medical_df = data_prep.prepare_medical_qa_dataset()
        
        # Get unique topics for naming
        topic_mapping = pd.read_csv("models/topic_mapping.csv")
        self.topic_names = topic_mapping.sort_values('id')['topic'].tolist()
        self.num_labels = len(self.topic_names)
        
        print(f"✅ Loaded {len(medical_df)} medical Q&A pairs")
        print(f"✅ Number of topics: {self.num_labels}")
        print(f"✅ Topics: {self.topic_names}")
        
        return medical_df
    
    def prepare_datasets(self, medical_df, test_size=0.2):
        """Prepare train/test datasets for HuggingFace"""
        print("🔧 Preparing datasets for HuggingFace training...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            medical_df, 
            test_size=test_size, 
            random_state=42,
            stratify=medical_df['topic_id']
        )
        
        print(f"📊 Train set: {len(train_df)} samples")
        print(f"📊 Test set: {len(test_df)} samples")
        
        # Load HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"✅ Loaded tokenizer: {self.model_name}")
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["combined_text"],
                truncation=True,
                padding="max_length",
                max_length=256  # Longer for medical Q&A
            )
        
        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df[['combined_text', 'topic_id']])
        test_dataset = Dataset.from_pandas(test_df[['combined_text', 'topic_id']])
        
        # Apply tokenization
        print("🔄 Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Rename for HuggingFace compatibility
        train_dataset = train_dataset.rename_column("topic_id", "labels")
        test_dataset = test_dataset.rename_column("topic_id", "labels")
        
        print("✅ Dataset preparation completed")
        return train_dataset, test_dataset
    
    def fine_tune_model(self, train_dataset, test_dataset):
        """Fine-tune HuggingFace model for topic classification"""
        print(f"🚀 Starting topic classification fine-tuning with {self.model_name}")
        print("-" * 60)
        
        # Load HuggingFace model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        print(f"✅ Loaded model: {self.model_name}")
        print(f"📊 Configured for {self.num_labels} topic classes")
        
        # Define metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results/topic_classifier",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir="./logs/topic",
            logging_steps=50,
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
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Start training
        print("🔥 Training started...")
        train_result = self.trainer.train()
        
        # Save the fine-tuned model
        self.trainer.save_model("models/topic_classifier")
        self.tokenizer.save_pretrained("models/topic_classifier")
        
        print("✅ Topic classification fine-tuning completed!")
        print(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate_model(self, test_dataset):
        """Evaluate the fine-tuned topic classification model"""
        print("\n📊 Evaluating topic classification model...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"✅ Topic Classification Accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.topic_names))
        
        # Create confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        return accuracy, y_true, y_pred
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot normalized confusion matrix"""
        print("📊 Creating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = normalize(cm.astype('float'), axis=1, norm='l1')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.topic_names,
            yticklabels=self.topic_names,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title('Topic Classification - Normalized Confusion Matrix', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Topic', fontsize=12)
        plt.ylabel('Actual Topic', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/topic_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved to results/topic_confusion_matrix.png")
    
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
        topic_id = int(result['label'].split('_')[-1])
        topic = self.topic_names[topic_id]
        confidence = result['score']
        
        return {
            'topic': topic,
            'confidence': confidence
        }