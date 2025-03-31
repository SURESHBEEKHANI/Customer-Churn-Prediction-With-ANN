import sys
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass  # Initialize the PredictPipeline class

    def predict(self, features):
        """
        Scale input features using a preprocessor and predict the target using a TensorFlow model.
        """
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.h5'
            preprocessor = load_object(file_path=preprocessor_path)
            # Load the TensorFlow model from model.h5
            model = tf.keras.models.load_model(model_path)
            
            # Convert CustomData to a 2D DataFrame before prediction
            df = features.get_data_as_data_frame()
            
            # Transform features before prediction
            data_scaled = preprocessor.transform(df)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 credit_score: int,
                 geography: str,
                 gender: str,
                 age: int,
                 tenure: int,
                 balance: float,
                 num_of_products: int,
                 has_cr_card: int,
                 is_active_member: int,
                 estimated_salary: float):
        """
        Initialize custom data for prediction.

        Parameters:
            credit_score (int): Credit score of the customer.
            geography (str): Customer's country.
            gender (str): Gender of the customer (Male/Female).
            age (int): Age of the customer.
            tenure (int): Number of years with the bank.
            balance (float): Account balance.
            num_of_products (int): Number of products used.
            has_cr_card (int): Whether the customer has a credit card (0/1).
            is_active_member (int): Whether the customer is an active member (0/1).
            estimated_salary (float): Estimated salary of the customer.
        """
        if credit_score is None or age is None or tenure is None or balance is None or num_of_products is None or estimated_salary is None:
            raise ValueError("Numeric fields cannot be None")

        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_of_products = num_of_products
        self.has_cr_card = has_cr_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary

    def get_data_as_data_frame(self):
        """
        Convert the input data into a pandas DataFrame.

        Returns:
            DataFrame: DataFrame containing the input features.
        """
        try:
            custom_data_input_dict = {
                'CreditScore': [self.credit_score],
                'Geography': [self.geography],
                'Gender': [self.gender],
                'Age': [self.age],
                'Tenure': [self.tenure],
                'Balance': [self.balance],
                'NumOfProducts': [self.num_of_products],
                'HasCrCard': [self.has_cr_card],
                'IsActiveMember': [self.is_active_member],
                'EstimatedSalary': [self.estimated_salary]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered successfully')
            logging.info(f"DataFrame contents: {df}")
            return df
        except Exception as e:
            logging.error('Exception occurred in getting dataframe')
            raise Exception(e)
