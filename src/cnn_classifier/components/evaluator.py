from pathlib import Path
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
from pathlib import Path 
from cnn_classifier.utils.common import save_json
from cnn_classifier.objects.config_object import EvaluationConfig


import dagshub
dagshub.init(repo_owner='viansebastian', repo_name='Kidney-Disease-MLFlow-CICD', mlflow=True)


class Evaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.2
        )
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = 'nearest'
        )
        
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.validation_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            shuffle=False,
            subset='validation',
            class_mode='categorical',
            **dataflow_kwargs
        )
    
    def evaluation(self): 
        self.model = self.load_model(self.config.model_path)
        self.valid_generator()
        self.score = self.model.evaluate(self.validation_generator)
        self.save_score()
        
    def save_score(self): 
        scores = {
            "loss" : self.score[0],
            "accuracy" : self.score[1]
        }
        save_json(
            path=Path('scores.json'),
            data=scores
        )
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    "loss" : self.score[0],
                    "accuracy" : self.score[1]
                }
            )
            
            # input_example = np.random.random(1, *self.config.params_image_size)
            # signature = infer_signature(input_example)
            
            if tracking_url_type_store != 'file':
                mlflow.keras.log_model(
                    self.model, 
                    'model', 
                    registered_model_name='MobileNetV2Kidney',
                    # input_example=input_example,
                    # signature=signature
                )
            else: 
                mlflow.keras.log_model(
                    self.model, 
                    'model',
                    # input_example=input_example,
                    # signature=signature
                    )
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path) 