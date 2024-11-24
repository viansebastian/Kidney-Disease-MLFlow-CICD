import os
import sys
sys.path.append('/home/gfspet/ml-projects/kidney-disease/src')
from cnn_classifier.constants import * 
from cnn_classifier.utils.common import read_yaml, create_directories
from cnn_classifier.objects.config_object import (
    DataIngestionConfig, 
    PretrainedModelConfig,
    TrainingConfig
)


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath=CONFIG_FILE_PATH, 
        params_filepath=PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig: 
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, 
            source_URL=config.source_URL, 
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_pretrained_base_model_config(self) -> PretrainedModelConfig:
        config = self.config.pretrained_base_model
        
        pretrained_base_model_config = PretrainedModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            used_model_path=Path(config.used_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP, 
            params_learning_rate=self.params.LEARNING_RATE,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        
        return pretrained_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        train_config = self.config.training
        pretrained_config = self.config.pretrained_base_model
        params = self.params
        train_data = os.path.join(self.config.data_ingestion.unzip_dir, 'data')
        
        create_directories([Path(train_config.root_dir)])
  
        train_config = TrainingConfig(
            root_dir=Path(train_config.root_dir),
            trained_model_path=Path(train_config.trained_model_path),
            used_model_path=Path(pretrained_config.used_model_path),
            training_data=Path(train_data),
            params_epoch=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )
        
        return train_config