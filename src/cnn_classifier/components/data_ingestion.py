import os 
import sys
sys.path.append('/home/gfspet/ml-projects/kidney-disease/src')
import zipfile
import gdown
from cnn_classifier import logger
from cnn_classifier.objects.config_object import DataIngestionConfig


class DataIngestion:
    def __init__(self, config=DataIngestionConfig):
        self.config = config
        
    def download_file(self) -> str: 
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs('artifacts/data_ingestion', exist_ok=True) 
            # os.makedirs(zip_download_dir, exist_ok=True)
            logger.info(f'Downloading data from ({dataset_url}) to ({zip_download_dir})')
            
            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)
            logger.info(f'Download completed')
            
        except Exception as e: 
            raise e
        
    def extract_zip(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True) 
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref: 
            zip_ref.extractall(unzip_path)