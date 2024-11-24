import sys
sys.path.append('/home/gfspet/ml-projects/kidney-disease/src')
from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.evaluator import Evaluator
from cnn_classifier import logger


# STAGE_NAME = 'Evaluation'

class EvaluatorPipeline: 
    def __init__(self):
        pass
    
    def main(self): 
        config = ConfigurationManager() 
        eval_config = config.get_eval_config()
        eval = Evaluator(config=eval_config) 
        eval.evaluation()
        eval.log_into_mlflow()