import os 
import yaml
import base64
import json
import joblib
from cnn_classifier import logger
from typing import Any
from pathlib import Path
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
'ConfigBox allows pandas notation in dictionary (d2.keyname)'
'ensure_annotations will only allow defined type in a function'


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file: 
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file: {path_to_yaml} loaded successfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('yaml file is empty')
    except Exception as e: 
        raise e
    
@ensure_annotations
def create_directories(path_to_dirs: list, verbose=True):
    for path in path_to_dirs: 
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Created directory at: {path}')
            
@ensure_annotations
def save_json(path: Path, data: dict): 
    with open(path, 'w') as f: 
        json.dump(data, f, indent=4) 
    
    logger.info(f'JSON file saved at: {path}')
    
@ensure_annotations
def load_json(path: Path) -> ConfigBox: 
    with open(path) as f: 
        content = json.load(f) 
        
    logger.info(f'JSON file from ({path}) loaded successfully')
    
    return ConfigBox(content)

@ensure_annotations
def save_binary(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info(f'Binary file saved at: {path}')
    
@ensure_annotations
def load_binary(path: Path) -> Any: 
    data = joblib.load(path)
    logger.info(f'Binary file from ({path}) loaded successfully')
    
    return data
    
@ensure_annotations
def get_size(path: Path) -> str: 
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f'~ {size_in_kb} KB'

def decode_image(img_string, file_name):
    img_data = base64.b64decode(img_string)
    with open(file_name, 'wb') as f: 
        f.write(img_data)
        f.close() 
        
def encode_image(img_path): 
    with open(img_path, 'rb') as f: 
        return base64.b64encode(f.read())