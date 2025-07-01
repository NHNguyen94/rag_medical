# from src.services.topic_clustering_service import TopicClusteringService
#
#
# def run_train(batch_size: int, epochs: int) -> (float, float):
#     topic_clustering_service = TopicClusteringService()
#     train_loss, val_loss = topic_clustering_service.train(
#         batch_size=batch_size,
#         epochs=epochs
#     )
#     print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")
#     topic_clustering_service.save_model()
#
#
# def run_eval(model_path: str, validation_data_path: str) -> Dict:
#     topic_clustering_service = TopicClusteringService(
#         validation_data_path=validation_data_path,
#     )
#     model = topic_clustering_service.load_model(model_path=model_path)
#     eval_result = topic_clustering_service.evaluate(model=model)
#     return eval_result