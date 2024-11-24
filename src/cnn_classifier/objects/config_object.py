from dataclasses import dataclass
from pathlib import Path 


@dataclass(frozen=True) 
class DataIngestionConfig: 
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PretrainedModelConfig: 
    root_dir: Path
    base_model_path: Path
    used_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig: 
    root_dir: Path
    trained_model_path: Path
    used_model_path: Path
    training_data: Path
    params_epoch: int
    params_batch_size: int 
    params_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    
    
@dataclass(frozen=True)
class EvaluationConfig: 
    model_path: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int