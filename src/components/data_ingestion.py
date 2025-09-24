import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass  
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        print("Starting data ingestion...")
        logging.info("Entered the data ingestion method or component")
        try:
            print("Reading CSV file: notebook/data/stud.csv")
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            print(f"Creating directory for artifacts: {os.path.dirname(self.ingestion_config.train_data_path)}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            print(f"Saving raw data to: {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            print(f"Saving train data to: {self.ingestion_config.train_data_path}")
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Train Test Split intiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            print(f"Saving train set to: {self.ingestion_config.train_data_path}")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            print(f"Saving test set to: {self.ingestion_config.test_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            print("Data ingestion completed successfully.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
data_transformation.initiate_data_transformation(train_data,test_data)