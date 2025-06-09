# test_models.py - Test both fine-tuned models
# =============================================================================

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the correct class
from ml_models.topic_clustering import TopicClassificationTrainer

def test_topic_clustering():
    """Test the topic clustering model"""
    print("🧪 Testing Topic Clustering Model")
    print("=" * 50)
    
    # Initialize trainer
    trainer = TopicClassificationTrainer()
    
    # Load and prepare data
    medical_df = trainer.load_prepared_data()
    
    # Prepare datasets
    train_dataset, test_dataset = trainer.prepare_datasets(medical_df)
    
    # Fine-tune model
    train_result = trainer.fine_tune_model(train_dataset, test_dataset)
    
    # Evaluate model
    accuracy, y_true, y_pred = trainer.evaluate_model(test_dataset)
    
    # Test inference
    trainer.test_inference()
    
    print("\n✅ Topic Clustering Test Completed!")
    print(f"📊 Final Accuracy: {accuracy:.3f}")
    
    return accuracy > 0.7  # Return True if accuracy is above 70%

if __name__ == "__main__":
    success = test_topic_clustering()
    if success:
        print("\n✅ Topic Clustering Model is working correctly!")
    else:
        print("\n❌ Topic Clustering Model needs improvement!")