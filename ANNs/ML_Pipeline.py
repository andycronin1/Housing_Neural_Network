import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
from data_cleaning import DataCleaningPipeline
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

load_dotenv()

BASE_DIR = os.getenv('BASE')

class MLPipeline:
    def __init__(self, data_input=None, base_dir=None):
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir

        if data_input == None:
            print('No data input. Running data pipeline and assigning default data')
            self.data_pipeline = DataCleaningPipeline()
            self.data_pipeline.run()
            self.data = self.data_pipeline.data
            del self.data_pipeline

        self.model = Sequential()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None




    def test_train_split(self):
        # Setting X and y variables to the .values of the features and label
        self.X = self.data.drop('loan_repaid', axis=1).values
        self.y = self.data['loan_repaid'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=101)

        # Normalising the data
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def model_build(self):

        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

        # input layer
        self.model.add(Dense(78, activation='relu'))
        self.model.add(Dropout(0.2))

        # hidden layer
        self.model.add(Dense(39, activation='relu'))
        self.model.add(Dropout(0.2))

        # hidden layer
        self.model.add(Dense(19, activation='relu'))
        self.model.add(Dropout(0.2))

        # output layer
        self.model.add(Dense(units=1, activation='sigmoid'))

        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

    def model_fitting(self):
        self.model.fit(x=self.X_train,
                  y=self.y_train,
                  epochs=25,
                  batch_size=256,
                  validation_data=(self.X_test, self.y_test),
                  )


if __name__ == "__main__":
    ML_obj = MLPipeline()
    ML_obj.test_train_split()
    ML_obj.model_build()
    ML_obj.model_fitting()