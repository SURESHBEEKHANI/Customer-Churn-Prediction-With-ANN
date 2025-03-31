from dataclasses import dataclass
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src.exception import CustomException
from src.logger import logging
from imblearn.over_sampling import SMOTE

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.h5')
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array: np.ndarray) -> tf.keras.Model:
        try:
            logging.info("Splitting training array into features and target")
            X = train_array[:, :-1]
            y = train_array[:, -1]

            logging.info("Applying SMOTE for handling imbalanced data")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            logging.info("SMOTE resampling completed")

            input_dim = X_res.shape[1]

            logging.info("Building ANN model")
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            optimizer = Adam(learning_rate=self.config.learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            logging.info("Starting model training")
            model.fit(X_res, y_res, epochs=self.config.epochs, batch_size=self.config.batch_size, verbose=1)

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            model.save(self.config.trained_model_file_path)
            logging.info("'model.h5' Saved")

            return model
        except Exception as e:
            logging.error("Exception occurred in Model Training")
            raise CustomException(e, None)
