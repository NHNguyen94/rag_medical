from bertopic import BERTopic
from src.core_managers.topic_clustering_manager import TopicClusteringManager

class ChatManager:
    def __init__(self):
        """
        Initialize the ChatManager with a trained BERTopic model.
        """
        try:
            self.topic_model = BERTopic.load("topic_model")
            self.topic_manager = TopicClusteringManager()
            print("Successfully loaded topic model")
        except Exception as e:
            print(f"Error loading topic model: {e}")
            print("Please run test_topic_clustering_train.py first to train the model")
            self.topic_model = None
            self.topic_manager = None

    def process_user_message(self, user_message: str) -> dict:
        """
        Process a user message and return relevant information including topic.
        
        Args:
            user_message (str): The user's input message
            
        Returns:
            dict: A dictionary containing:
                - topic_id: The predicted topic ID
                - topic_info: Information about the topic
                - similar_questions: List of similar questions from the same topic
        """
        if not self.topic_model:
            return {
                "error": "Topic model not loaded. Please train the model first."
            }

        try:
            # Predict topic
            topic_id, _ = self.topic_model.transform([user_message])
            topic_id = topic_id[0]

            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].values[0]

            # Get representative documents (similar questions) for this topic
            try:
                similar_questions = self.topic_model.get_representative_docs(topic_id)
                if similar_questions is not None:
                    similar_questions = similar_questions[:5]  # Return top 5 similar questions
                else:
                    similar_questions = []
            except Exception as e:
                print(f"Warning: Could not get similar questions: {e}")
                similar_questions = []

            return {
                "topic_id": int(topic_id),
                "topic_name": topic_name,
                "similar_questions": similar_questions
            }

        except Exception as e:
            return {
                "error": f"Error processing message: {str(e)}"
            }

    def get_topic_statistics(self) -> dict:
        """
        Get statistics about all topics.
        
        Returns:
            dict: Statistics about topics including:
                - total_topics: Number of topics
                - topic_distribution: Distribution of topics
                - topic_names: List of topic names
        """
        if not self.topic_model:
            return {
                "error": "Topic model not loaded. Please train the model first."
            }

        try:
            topic_info = self.topic_model.get_topic_info()
            
            return {
                "total_topics": len(topic_info),
                "topic_distribution": topic_info['Count'].to_dict(),
                "topic_names": topic_info['Name'].to_dict()
            }
        except Exception as e:
            return {
                "error": f"Error getting topic statistics: {str(e)}"
            } 