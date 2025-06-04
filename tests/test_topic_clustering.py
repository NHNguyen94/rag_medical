from src.core_managers.topic_clustering_manager import TopicClusteringManager
from src.core_managers.document_manager import DocumentManager

doc_manager = DocumentManager()
documents = doc_manager.load_csv_to_documents(
    "/Users/nasibamammadli/Downloads/archive/MedicalQuestionAnswering.csv",
    "Question"
)
texts = [doc.text for doc in documents]

topic_manager = TopicClusteringManager()
topics, probs = topic_manager.fit(texts)

print(topic_manager.get_topic_info())
