import sys
sys.path.append('/home/gfspet/ml-projects/kidney-disease/src')
from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_trainer import Training
from cnn_classifier import logger


STAGE_NAME = 'Model Training'

class ModelTrainerPipeline: 
    def __init__(self):
        pass
    
    def main(self): 
        config = ConfigurationManager() 
        training_config = config.get_training_config()
        training = Training(config=training_config) 
        training.get_base_model()
        training.train_valid_generator()
        training.train()
        
        
        
if __name__ == '__main__':
    try: 
        logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
        model_trainer = ModelTrainerPipeline() 
        model_trainer.main() 
        logger.info(f'>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<')
        
    except Exception as e: 
        raise e