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
import pickle
import seaborn as sns
load_dotenv()
BASE_DIR = os.getenv('BASE')


class PostProcessing:
    def __init__(self, base_dir=None):

        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir

        # Load data if exists
        if f'{self.base_dir}/DATA/history.pkl' != None:
            print('Saved ML data found. Loading...')
            with open(f'{self.base_dir}/DATA/history.pkl', 'rb') as f:
                self.model_history = pickle.load(f)
        else:
            print('No saved ML data found')

    def val_loss_vs_train_loss(self):
        losses = pd.DataFrame(self.model_history.history)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=losses[['loss', 'val_loss']])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.show()



if __name__ == "__main__":
    postProcess = PostProcessing()
    postProcess.val_loss_vs_train_loss()