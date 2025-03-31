# To run this pipeline, open a terminal, navigate to this directory and run:
# python pipeline.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_tranier import ModelTrainer
from src.logger import logging

class Pipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Pipeline started")
            train_data_path, test_data_path = self.data_ingestion.initate_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            model = self.model_trainer.initiate_model_training(train_arr)
            logging.info("Pipeline completed")
            return model
        except Exception as e:
            logging.error("Pipeline failed")
            raise e

if __name__ == '__main__':
    pipeline = Pipeline()
    trained_model = pipeline.run_pipeline()