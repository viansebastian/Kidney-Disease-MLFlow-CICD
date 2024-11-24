from src.cnn_classifier import logger
from src.cnn_classifier.pipeline.data_ingestion import DataIngestionPipeline
from src.cnn_classifier.pipeline.prepare_model import PrepareModelPipeline
from src.cnn_classifier.pipeline.model_trainer import ModelTrainerPipeline
from src.cnn_classifier.pipeline.evaluator import EvaluatorPipeline



STAGE_NAME = 'Data Ingestion'
try: 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    data_ingest = DataIngestionPipeline() 
    data_ingest.main() 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<')
    
except Exception as e: 
    raise e


STAGE_NAME = 'Model Preparation'
try: 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    prepare_model = PrepareModelPipeline() 
    prepare_model.main() 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<')
    
except Exception as e: 
    raise e


STAGE_NAME = 'Model Training'
try: 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    model_trainer = ModelTrainerPipeline() 
    model_trainer.main() 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<')
    
except Exception as e: 
    raise e


STAGE_NAME = 'Evaluation'
try: 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    evaluator = EvaluatorPipeline() 
    evaluator.main() 
    logger.info(f'>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<')
    
except Exception as e: 
    raise e