from typing import List, Dict
from pathlib import Path
import json
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch
import pandas as pd

from src.pipelines.question_recommendation.data_processor import QuestionDataProcessor
from src.pipelines.question_recommendation.question_generator import QuestionGenerator


class FineTuningPipeline:
    def __init__(
            self,
            model_name: str = "google/flan-t5-base",
            data_dir: str = "../../data/fine_tune_dataset",
            output_dir: str = "../../ml_modles/flant5",
            max_length: int = 256,
            batch_size: int = 2,
            learning_rate: float = 2e-5,
            num_epochs: int = 3
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize components
        self.data_processor = QuestionDataProcessor(data_dir=data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self) -> Dict[str, Dataset]:
        """Prepare the dataset for training."""
        # Process datasets
        processed_data = self.data_processor.process_datasets()

        # Pass both faiss_index and questions_mapping to QuestionGenerator
        question_generator = QuestionGenerator(
            faiss_index=processed_data['faiss_index'],
            questions_mapping=processed_data['questions_mapping']
        )

        # Generate training pairs
        training_data = []
        for idx, question in enumerate(processed_data['questions']):
            question_embedding = processed_data['embeddings'][idx]

            follow_up_questions = question_generator.generate_follow_up_questions(
                question_embedding,
                num_questions=4
            )

            training_data.append({
                'input': question,
                'output': follow_up_questions,
                'follow_up_combined': ' | '.join(follow_up_questions)
            })

        # Convert to DataFrame for inspection
        df = pd.DataFrame(training_data)

        # Save to CSV for manual inspection
        csv_path = self.output_dir / "training_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Training data saved to {csv_path}")

        # Convert to dataset
        dataset = Dataset.from_list(training_data)
        split_dataset = dataset.train_test_split(test_size=0.1)

        # Create a tokenization function that can access self
        def tokenize_batch(examples):
            return self.tokenize_function(examples)

        # Tokenize both splits
        tokenized_dataset = {
            'train': split_dataset['train'].map(
                tokenize_batch,
                batched=True,
                remove_columns=dataset.column_names
            ),
            'validation': split_dataset['test'].map(
                tokenize_batch,
                batched=True,
                remove_columns=dataset.column_names
            )
        }

        return tokenized_dataset

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        # Join the list of questions with a separator
        formatted_outputs = [" | ".join(questions) for questions in examples["output"]]

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                formatted_outputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):
        """Train the model."""
        # Prepare dataset
        datasets = self.prepare_training_data()


        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_dir=str(self.output_dir / "logs"),
            load_best_model_at_end=True,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding="max_length",
            max_length=self.max_length
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))