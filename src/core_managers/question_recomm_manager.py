from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import json
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import torch
import pandas as pd

from src.utils.enums import QuestionRecommendConfig
from src.services.question_generator_service import QuestionGenerator
from src.services.question_processor_service import QuestionDataProcessor


class FineTuningPipeline:
    def __init__(
        self,
        model_name: str = None,
        data_dir: str = None,
        output_dir: str = None,
        model_save_path: str = None,
        model_save_name: str = None,
        max_length: int = 256,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name

        # Initialize components
        self.data_processor = QuestionDataProcessor(
            data_dir=self.data_dir, output_dir=self.output_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create output directory
        # if not self.output_dir.exists():
        #     self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self) -> Dict[str, Dataset]:
        """Prepare the dataset for training."""
        # Process datasets
        processed_data = self.data_processor.process_datasets()

        # Initialize question generator
        question_generator = QuestionGenerator(
            faiss_index=processed_data["faiss_index"],
            questions_mapping=processed_data["questions_mapping"],
        )

        # Generate training pairs
        training_data = []
        for idx, question in tqdm(
            enumerate(processed_data["questions"]),
            total=len(processed_data["questions"]),
            desc="Generating training data",
        ):
            question_embedding = processed_data["embeddings"][idx]

            follow_up_questions = question_generator.generate_follow_up_questions(
                question_embedding, num_questions=4
            )
            if not follow_up_questions:
                print(f"Warning: No follow-ups for question at idx {idx}")

            training_data.append(
                {
                    "input": question,
                    "output": follow_up_questions,
                    "follow_up_combined": " | ".join(follow_up_questions),
                }
            )

        print(f"Sample training data:")
        for i in range(min(3, len(training_data))):
            print(f"Input: {training_data[i]['input']}")
            print(f"Output: {training_data[i]['output']}")
            print("---")

        # Convert to DataFrame for inspection
        df = pd.DataFrame(training_data)

        # Save to CSV for manual inspection
        csv_path = self.output_dir / "training_data.csv"
        df.to_csv(csv_path, index=False)

        # Convert to dataset
        dataset = Dataset.from_list(training_data)
        split_dataset = dataset.train_test_split(test_size=0.1)

        # Tokenize both splits
        tokenized_dataset = {
            "train": split_dataset["train"].map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            ),
            "validation": split_dataset["test"].map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            ),
        }

        print("Sample tokenized data:")
        sample = tokenized_dataset["train"][0]
        print(f"Input IDs shape: {len(sample['input_ids'])}")
        print(f"Labels shape: {len(sample['labels'])}")
        print(f"First few input IDs: {sample['input_ids'][:10]}")
        print(f"First few labels: {sample['labels'][:10]}")

        val_labels = tokenized_dataset["validation"]["labels"]
        all_nan = all(all(tok == -100 for tok in seq) for seq in val_labels)
        print(f"Validation all -100? {all_nan}")

        labels_flat = [
            label for labels in tokenized_dataset["train"]["labels"] for label in labels
        ]
        num_ignored = sum(1 for label in labels_flat if label == -100)
        print(f"Number of ignored tokens (-100): {num_ignored}/{len(labels_flat)}")

        return tokenized_dataset

    def tokenize_function(self, examples):
        """Tokenize the input and output sequences."""
        model_inputs = self.tokenizer(
            examples["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Join the list of questions with a separator
        formatted_outputs = [" | ".join(questions) for questions in examples["output"]]
        print(f"Formatted target sample: {formatted_outputs[:2]}")

        labels = self.tokenizer(
            text_target=formatted_outputs,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        labels_input_ids = labels["input_ids"].copy()
        labels_input_ids = [
            [
                (label if label != self.tokenizer.pad_token_id else -100)
                for label in label_seq
            ]
            for label_seq in labels_input_ids
        ]

        model_inputs["labels"] = labels_input_ids
        for i, seq in enumerate(labels_input_ids):
            if all(label == -100 for label in seq):
                print(f"Warning: All labels ignored at index {i}!")

        return model_inputs

    def train(self):
        """Train the model."""
        # Prepare dataset
        datasets = self.prepare_training_data()

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            # evaluation_strategy="epoch",
            eval_steps=100,
            do_eval=True,
            do_train=True,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            save_total_limit=3,
            predict_with_generate=False,
            fp16=False,
            logging_dir=str(self.output_dir / "logs"),
            load_best_model_at_end=True,
            log_level="info",
            report_to="none",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=100,
            dataloader_pin_memory=False,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, label_pad_token_id=-100
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        metrics = trainer.evaluate()
        print(f"Final validation loss: {metrics['eval_loss']:.4f}")

        # Save the model
        torch.save(self.model.state_dict(), self.model_save_path / self.model_save_name)
