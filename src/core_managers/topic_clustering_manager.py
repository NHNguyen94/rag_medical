from bertopic import BERTopic
from hdbscan import HDBSCAN
import pandas as pd
from src.core_managers.document_manager import DocumentManager

class TopicClusteringManager:
    def __init__(self, **kwargs):
        """
        Production: Use HDBSCAN for clustering (best for real datasets)
        """
        #hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2)
        #self.model = BERTopic(hdbscan_model=hdbscan_model, **kwargs)
        self.model = BERTopic(**kwargs)
        self.topics = None
        self.probs = None

    def fit(self, documents):
        """
        Fit the BERTopic model to your list of documents (texts).
        Args:
            documents (List[str]): List of text documents to cluster.
        Returns:
            topics: List of topic assignments for each document.
            probs: List of probabilities for each topic assignment.
        """
        self.topics, self.probs = self.model.fit_transform(documents)
        self.model.save("topic_model")
        return self.topics, self.probs

    def predict_topic(self, new_document):
        """
        Predict the topic for a new document.
        Args:
            new_document (str): The text to assign a topic to.
        Returns:
            topic: The predicted topic number.
        """
        topic, _ = self.model.transform([new_document])
        return topic[0]

    def get_topic_info(self):
        """
        Get information about the topics discovered.
        Returns:
            DataFrame: Topic info (topic number, count, name, etc.)
        """
        return self.model.get_topic_info()

    def get_representative_docs(self, topic_id):
        """
        Get representative documents for a given topic.
        Args:
            topic_id (int): The topic number.
        Returns:
            List[str]: Representative documents for the topic.
        """
        return self.model.get_representative_docs(topic_id)

def extract_texts_from_csv(csv_path, text_column):
    df = pd.read_csv(csv_path)
    texts = df[text_column].dropna().astype(str).tolist()
    return texts

# Usage example:
#doc_manager = DocumentManager()
#documents = doc_manager.load_csv_to_documents(
 #   "/Users/nasibamammadli/Downloads/archive/MedicalQuestionAnswering.csv",
 #   "Question"
# )
# texts = [doc.text for doc in documents]

# topic_manager = TopicClusteringManager()
# topics, probs = topic_manager.fit(texts)  # This will save "topic_model"

# print(topic_manager.get_topic_info())

# Load the trained model
# topic_model = BERTopic.load("topic_model")

# user_message = "What are the symptoms of diabetes?"
# predicted_topic = topic_model.transform([user_message])[0]
# print(f"Predicted topic: {predicted_topic}")

# topic_info = topic_model.get_topic_info()
# print(topic_info)

