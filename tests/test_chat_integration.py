from src.core_managers.chat_manager import ChatManager

def test_chat_integration():
    # Initialize the chat manager
    chat_manager = ChatManager()
    
    # Test messages
    test_messages = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes heart disease?",
        "What are the side effects of aspirin?"
    ]
    
    print("Testing Chat Integration with Topic Clustering")
    print("=" * 50)
    
    # Process each test message
    for message in test_messages:
        print(f"\nProcessing message: {message}")
        result = chat_manager.process_user_message(message)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Topic ID: {result['topic_id']}")
            print(f"Topic Name: {result['topic_name']}")
            print("\nSimilar Questions:")
            for i, question in enumerate(result['similar_questions'], 1):
                print(f"{i}. {question}")
    
    # Get topic statistics
    print("\nTopic Statistics")
    print("=" * 50)
    stats = chat_manager.get_topic_statistics()
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
    else:
        print(f"Total Topics: {stats['total_topics']}")
        print("\nTopic Distribution:")
        for topic_id, count in stats['topic_distribution'].items():
            topic_name = stats['topic_names'].get(topic_id, "Unknown")
            print(f"Topic {topic_id} ({topic_name}): {count} questions")

if __name__ == "__main__":
    test_chat_integration() 