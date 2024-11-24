import os 
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline: 
    def __init__(self, filename):
        self.filename = filename
        
    def predict(self): 
        model = load_model(os.path.join('artifacts', 'training', 'final_model.keras'))
        
        img_name = self.filename
        input_img = image.load_img(img_name, target_size=(224, 224))
        input_img = image.img_to_array(input_img) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        pred = np.argmax(model.predict(input_img), axis=1)
             
        class_indices = {0: 'Normal', 1: 'Stone', 2: 'Tumor'}
        
        class_name = class_indices.get(pred[0], "Unknown")
        
        return [
            {
                "image": class_name
            }
        ]