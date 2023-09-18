import os, sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_validation import DataValidation
from src.components.data_validation import DataValidationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the Data Ingestion component')

        try:
            df = pd.read_csv('notebook\data\stroke.csv')
            df.drop(columns=['id'], inplace=True)
            df['age'] = df['age'].astype('int32')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data Ingestion Completed!')

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

    
if __name__ == '__main__':
    module = DataIngestion()
    raw_data = module.initiate_data_ingestion()
    # print(raw_data)

    data_transformation = DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation()

    data_validation = DataValidation()
    train_arr, test_arr,_ = data_validation.initiate_data_transformation(train_data, test_data)
    print(train_arr.shape, test_arr.shape)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))