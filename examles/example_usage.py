"""Example usage of Green Shield"""

from inference.predictor import Predictor

def main():
    # Initialize predictor
    predictor = Predictor(
        model_path='checkpoints/best_model.pt',
        config={'classes': ['chainsaw', 'gunshot', 'vehicle', 'fire', 'animal_distress', 'background']}
    )
    
    # Predict
    result = predictor.predict('data/raw/chainsaw/sample.wav')
    print(f"Detected: {result.class_name} ({result.confidence:.2%})")

if __name__ == "__main__":
    main()