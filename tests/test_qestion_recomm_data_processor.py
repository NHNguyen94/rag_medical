# import pytest
# import torch
# import pandas as pd
# from pathlib import Path
# import tempfile
# import shutil
#
# from src.pipelines.question_recommendation.data_processor import QuestionDataProcessor
#
#
# class TestQuestionDataProcessor:
#     @pytest.fixture(scope="class")
#     def temp_dirs(self):
#         # Create temporary directories for testing
#         with (
#             tempfile.TemporaryDirectory() as data_dir,
#             tempfile.TemporaryDirectory() as output_dir,
#         ):
#             # Create sample CSV files in data_dir
#             df1 = pd.DataFrame(
#                 {
#                     "question": ["What is diabetes?", "How to prevent heart disease?"],
#                     "answer": ["Diabetes is...", "To prevent heart disease..."],
#                 }
#             )
#             df2 = pd.DataFrame(
#                 {
#                     "question": ["What are symptoms of COVID?", "How to stay healthy?"],
#                     "answer": ["COVID symptoms include...", "To stay healthy..."],
#                 }
#             )
#
#             df1.to_csv(f"{data_dir}/dataset1.csv", index=False)
#             df2.to_csv(f"{data_dir}/dataset2.csv", index=False)
#
#             yield data_dir, output_dir
#
#     @pytest.fixture(scope="class")
#     def processor(self, temp_dirs):
#         data_dir, output_dir = temp_dirs
#         return QuestionDataProcessor(
#             data_dir=data_dir,
#             output_dir=output_dir,
#             embedding_dim=768,  # Using BERT base dimension for testing
#         )
#
#     def test_load_and_combine_datasets(self, processor):
#         combined_df = processor.load_and_combine_datasets()
#         assert isinstance(combined_df, pd.DataFrame)
#         assert len(combined_df) == 4  # Total number of questions from both files
#         assert "source" in combined_df.columns
#         assert set(combined_df["source"].unique()) == {"dataset1", "dataset2"}
#
#     def test_preprocess_data(self, processor):
#         combined_df = processor.load_and_combine_datasets()
#         processed_df = processor.preprocess_data(combined_df)
#
#         assert "cleaned_question" in processed_df.columns
#         assert processed_df["cleaned_question"].isna().sum() == 0  # No NaN values
#         assert len(processed_df) == len(
#             processed_df["cleaned_question"].unique()
#         )  # No duplicates
#
#     def test_create_embeddings(self, processor):
#         questions = ["What is diabetes?", "How to stay healthy?"]
#         embeddings = processor.create_embeddings(questions)
#
#         assert len(embeddings) == len(questions)
#         assert all(len(emb) == processor.embedding_dim for emb in embeddings)
#         assert all(isinstance(emb, list) for emb in embeddings)
#         assert all(isinstance(val, float) for emb in embeddings for val in emb)
#
#     def test_tokens_to_embedding(self, processor):
#         # Test single token conversion
#         tokens = processor.encoding_manager.tokenize_text("What is diabetes?")
#         embedding = processor._tokens_to_embedding(tokens)
#
#         assert isinstance(embedding, list)
#         assert len(embedding) == processor.embedding_dim
#         assert all(isinstance(val, float) for val in embedding)
#
#     def test_build_faiss_index(self, processor):
#         questions = ["What is diabetes?", "How to stay healthy?"]
#         embeddings = processor.create_embeddings(questions)
#         faiss_index = processor.build_faiss_index(questions, embeddings)
#
#         assert faiss_index is not None
#         # Test if we can search in the index
#         test_embedding = embeddings[0]
#         _, indices = faiss_index.search(torch.tensor([test_embedding]), k=1)
#         assert indices[0][0] == 0  # Should find the first question
#
#     def test_process_datasets(self, processor):
#         processed_data = processor.process_datasets()
#
#         assert isinstance(processed_data, dict)
#         assert all(
#             key in processed_data
#             for key in ["questions", "embeddings", "faiss_index", "metadata"]
#         )
#         assert len(processed_data["questions"]) == len(processed_data["embeddings"])
#         assert processed_data["metadata"]["embedding_dim"] == processor.embedding_dim
#
#         # Test if output file was created
#         output_file = processor.output_dir / "processed_data.json"
#         assert output_file.exists()
#
#     def test_dimension_handling(self, processor):
#         # Print actual dimensions for debugging
#         tokens = processor.encoding_manager.tokenize_text("Test question")
#         tokens_tensor = processor.encoding_manager.to_tensor([tokens], "long").to(
#             processor.encoding_manager.device
#         )
#
#         with torch.no_grad():
#             embeddings = processor.encoding_manager.model.embeddings(tokens_tensor)
#             print(f"Model embedding dimension: {embeddings.shape[-1]}")
#             print(f"Expected dimension: {processor.embedding_dim}")
