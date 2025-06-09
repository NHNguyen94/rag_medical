import pandas as pd
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the emotion recognition trainer
from ml_models.emotion_recognition import EmotionRecognitionTrainer

def test_emotion_recognition():
    """Test the emotion recognition model"""
    print("🧪 Testing Emotion Recognition Model")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EmotionRecognitionTrainer()
    
    # Load and prepare data
    emotion_dfs = trainer.load_prepared_data()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(emotion_dfs)
    
    # Fine-tune model
    train_result = trainer.fine_tune_model(train_dataset, val_dataset)
    
    # Evaluate model
    accuracy, y_true, y_pred = trainer.evaluate_model(test_dataset)
    
    # Test inference
    trainer.test_inference()
    
    print("\n✅ Emotion Recognition Test Completed!")
    print(f"📊 Final Accuracy: {accuracy:.3f}")
    
    return accuracy > 0.7  # Return True if accuracy is above 70%

if __name__ == "__main__":
    success = test_emotion_recognition()
    if success:
        print("\n✅ Emotion Recognition Model is working correctly!")
    else:
        print("\n❌ Emotion Recognition Model needs improvement!") 