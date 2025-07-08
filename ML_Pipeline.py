import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import tensorflow
import traceback
from data_cleaning import DataCleaningPipeline
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers
import pickle
from keras import ops
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation,Dropout
# from tensorflow.keras.constraints import max_norm
import logging
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.getenv('BASE')

class MLPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir

        self.saved_model = None

        try:
            self.saved_model = keras.models.load_model(f'{self.base_dir}/DATA/saved_keras_model.keras')
            print("Model loaded successfully!")
        except FileNotFoundError:
            logger.info("Error: Model file not found. Running model training...")
        except Exception as e:
            logger.info(f"Error loading model: {e}")
            traceback.print_exc()

        if self.saved_model == None:
            print('No data input. Running data pipeline and assigning default data')
            self.data_pipeline = DataCleaningPipeline()
            self.data_pipeline.run()
            self.data = self.data_pipeline.data
            del self.data_pipeline

        self.model = keras.Sequential()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def test_train_split(self):
        # Setting X and y variables to the .values of the features and label
        self.X = self.data.drop(columns='loan_repaid', axis=1).values
        self.y = self.data['loan_repaid'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=101)

        # Normalising the data
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def model_build(self):

        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

        # input layer
        self.model.add(layers.Dense(78, activation='relu'))
        self.model.add(layers.Dropout(0.2))

        # hidden layer
        self.model.add(layers.Dense(39, activation='relu'))
        self.model.add(layers.Dropout(0.2))

        # hidden layer
        self.model.add(layers.Dense(19, activation='relu'))
        self.model.add(layers.Dropout(0.2))

        # output layer
        self.model.add(layers.Dense(units=1, activation='sigmoid'))

        # configure model (compile basically means configure)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

    def model_fitting(self):
        logger.info('Picked data not found')
        self.model.fit(x=self.X_train,
                  y=self.y_train,
                  epochs=25,
                  batch_size=256,
                  validation_split=0.1,
                  )

        with open(f'{self.base_dir}/DATA/history.pkl', 'wb') as f:
            pickle.dump(self.model.history, f)

        a=1

        self.model.save(f'{self.base_dir}/DATA/saved_keras_model.keras')



    def run(self):
        if self.saved_model == None:
            self.test_train_split()
            self.model_build()
            self.model_fitting()
        else:
            print('Run Method not run. Data already loaded')


if __name__ == "__main__":
    ML_obj = MLPipeline()
    ML_obj.run()