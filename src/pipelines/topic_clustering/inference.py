from src.services.topic_clustering_service import TopicClusteringService
from src.utils.enums import ChatBotConfig

chatbot_config = ChatBotConfig()


def run_inference(text: str) -> None:
    topic_clustering_service = TopicClusteringService()
    model = topic_clustering_service.load_model()
    predicted_topic_no = topic_clustering_service.predict(text, model)
    predicted_topic = chatbot_config.DOMAIN_NUMBER_MAPPING[predicted_topic_no]
    print(f"Text: {text}, Predicted Topic: {predicted_topic}")


if __name__ == "__main__":
    texts = [
        "What are the symptoms of diabetes?",
        "How can I manage my hypertension?",
        "What is the best treatment for anxiety disorders?",
        "What are the side effects of chemotherapy?",
        "How can I improve my mental health?",
    ]

    for text in texts:
        run_inference(text)
