from src.core_managers.topic_clustering_manager import TopicClusteringManager, extract_texts_from_csv

def test_train_topic_model():
    # Use your actual data path and column name
    csv_path = "/Users/nasibamammadli/Downloads/archive/MedicalQuestionAnswering.csv"
    text_column = "Question"

    # Extract texts from CSV
    texts = extract_texts_from_csv(csv_path, text_column)
    print(f"Loaded {len(texts)} documents")

    # Initialize and train the model
    topic_manager = TopicClusteringManager()
    topics, probs = topic_manager.fit(texts)  # This will save "topic_model"

    # Print topic information
    topic_info = topic_manager.get_topic_info()
    print("\nTopic Information:")
    print(topic_info)

if __name__ == "__main__":
    test_train_topic_model()
