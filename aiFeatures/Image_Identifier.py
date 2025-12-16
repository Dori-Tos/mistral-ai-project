import os
from typing import Optional, List, Dict, Any, Callable, Type, Union
from typing_extensions import Self
import tf_keras as keras
import numpy as np
import cv2 as cv

class Image_Identifier:
    """Singleton class for AI Image Identifier model."""
    
    _instance: Optional[Self] = None
    _initialized: bool = False
    
    def __new__(cls, file_name: str = 'Identifier_Model.h5') -> Self:
        if cls._instance is None:
            cls._instance = super(Image_Identifier, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, file_name: str = 'Identifier_Model.h5'):
        # Only initialize once (singleton pattern)
        if Image_Identifier._initialized:
            return
            
        # Suppress oneDNN warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Get the directory where this module is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, file_name)
        
        # Load the model
        try:
            self.model = keras.models.load_model(model_path)
            assert self.model is not None
            assert isinstance(self.model, keras.Model)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}:", e)
            import traceback
            traceback.print_exc()
            raise
        
        self.img_size = 48  # Model input size
        Image_Identifier._initialized = True
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict if an image is AI-generated or real.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
                - 'prediction': 'AI Generated' or 'Real'
                - 'score': float between 0 and 1 (higher = more likely AI generated)
                - 'confidence': percentage confidence
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess the image
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        image_resized = cv.resize(image, (self.img_size, self.img_size))
        image_resized = image_resized / 255.0
        
        # Prepare for prediction
        test = np.array([image_resized]).reshape(-1, self.img_size, self.img_size, 3)
        
        # Make prediction
        predictions = self.model.predict(test, verbose=0)
        score = float(predictions[0][0])
        
        # Determine classification
        if score <= 0.5:
            prediction = "Real"
            confidence = (1 - score) * 100
        else:
            prediction = "AI Generated"
            confidence = score * 100
        
        return {
            'prediction': prediction,
            'score': score,
            'confidence': confidence
        }
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict if an image is AI-generated or real from a numpy array.
        
        Args:
            image_array: Numpy array representing the image (BGR format)
            
        Returns:
            Dictionary containing:
                - 'prediction': 'AI Generated' or 'Real'
                - 'score': float between 0 and 1 (higher = more likely AI generated)
                - 'confidence': percentage confidence
        """
        if image_array is None or len(image_array.shape) != 3:
            raise ValueError("Invalid image array")
        
        # Preprocess the image
        image_resized = cv.resize(image_array, (self.img_size, self.img_size))
        image_resized = image_resized / 255.0
        
        # Prepare for prediction
        test = np.array([image_resized]).reshape(-1, self.img_size, self.img_size, 3)
        
        # Make prediction
        predictions = self.model.predict(test, verbose=0)
        score = float(predictions[0][0])
        
        # Determine classification
        if score <= 0.5:
            prediction = "Real"
            confidence = (1 - score) * 100
        else:
            prediction = "AI Generated"
            confidence = score * 100
        
        return {
            'prediction': prediction,
            'score': score,
            'confidence': confidence
        }
    
    def describe_model(self):
        """Print model summary and information."""
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        self.model.summary()
        
        print("\n" + "="*50)
        print("INPUT INFORMATION")
        print("="*50)
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")


def get_image_identifier() -> Image_Identifier:
    """Get the singleton instance of Image_Identifier."""
    return Image_Identifier()


# # Example usage
# if __name__ == "__main__":
#     # Create singleton instance
#     identifier = Image_Identifier()
    
#     # Describe the model
#     identifier.describe_model()
    
#     # Test with an image
#     test_image = "path/to/test/image.jpg"
#     if os.path.exists(test_image):
#         result = identifier.predict(test_image)
#         print(f"\nPrediction: {result['prediction']}")
#         print(f"Score: {result['score']:.4f}")
#         print(f"Confidence: {result['confidence']:.2f}%")
    
#     # Verify singleton behavior
#     identifier2 = Image_Identifier()
#     assert identifier is identifier2
#     print("\nSingleton pattern verified: Both instances are the same object")