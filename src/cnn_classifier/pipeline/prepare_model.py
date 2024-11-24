import sys
sys.path.append('/home/gfspet/ml-projects/kidney-disease/src')
from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.pretrained_base_model import PretrainedBaseModel
from cnn_classifier import logger


# STAGE_NAME = 'Model Preparation Stage'

class PrepareModelPipeline: 
    def __init__(self):
        pass
    
    def main(self): 
        config = ConfigurationManager() 
        pretrained_base_model_config = config.get_pretrained_base_model_config()
        pretrained_base_model = PretrainedBaseModel(config=pretrained_base_model_config) 
        pretrained_base_model.get_base_model()
        pretrained_base_model.update_base_model()
        
# if __name__ == '__main__': 
#     try: 
#         logger.info(f'---------- {STAGE_NAME} started----------')
#         prerpare_base_model = PrepareModel() 
#         prerpare_base_model.main() 
#         logger.info(f'---------- {STAGE_NAME} completed----------')
        
#     except Exception as e: 
#         raise e