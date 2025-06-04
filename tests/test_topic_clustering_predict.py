from bertopic import BERTopic

def test_predict_topic():
    # Load the trained model
    topic_model = BERTopic.load("topic_model")

    # Test with some example questions
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes heart disease?",
        "What are the side effects of aspirin?"
    ]

    print("Testing topic prediction with example questions:")
    print("-" * 50)
    
    for question in test_questions:
        predicted_topic = topic_model.transform([question])[0]
        print(f"\nQuestion: {question}")
        print(f"Predicted topic: {predicted_topic}")

    # Print overall topic information
    print("\nTopic Information:")
    print("-" * 50)
    topic_info = topic_model.get_topic_info()
    print(topic_info)

if __name__ == "__main__":
    test_predict_topic()