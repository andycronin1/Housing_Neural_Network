import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import tensorflow
from data_cleaning import DataCleaningPipeline
from ML_Pipeline import MLPipeline
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers
from keras import ops
load_dotenv()
BASE_DIR = os.getenv('BASE')


class PostProcessing:
    def __init__(self, data_input=None, base_dir=None):
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir






if __name__ == "__main__":
    data_cleaning = DataCleaningPipeline()
    data_cleaning.run()
    ML_obj = MLPipeline()
    ML_obj.run()