import os
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
from data_cleaning import DataCleaningPipeline
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.getenv('BASE')

class XgboostPipeline():
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir

        self.saved_xgmodel = None

        if self.saved_xgmodel == None:
            print('No data input. Running data pipeline and assigning default data')
            self.data_pipeline = DataCleaningPipeline()
            self.data_pipeline.run()
            self.data = self.data_pipeline.data
            del self.data_pipeline

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dtrain_clf = None
        self.dtest_clf = None
        self.results = None
        self.preds = None
        self.model = None

    def test_train_split(self):
        # Setting X and y variables to the .values of the features and label
        self.X = self.data.drop(columns='loan_repaid', axis=1).values
        self.y = self.data['loan_repaid'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=101)

    def model_build(self):

        # Create regression matrices (It is a highly optimized class for memory and speed.)
        #  converting datasets into this format is a requirement for the native XGBoost API
        self.dtrain_clf = xgb.DMatrix(self.X_train, self.y_train, enable_categorical=True)
        self.dtest_clf = xgb.DMatrix(self.X_test, self.y_test, enable_categorical=True)

        params = {"objective": "binary:logistic", "tree_method": "hist"}
        n = 1000

        # Using cross validation to train model
        # FYI: logloss function is the equivalent of binary cross entropy in neural network
        # test-logloss-mean is the equivalent of the validation loss .
        self.results = xgb.cv(
            params, self.dtrain_clf,
            num_boost_round=n,
            nfold=5,
            metrics=["logloss", "auc"],
            verbose_eval=100
        )

        # # Train the final model
        # self.model = xgb.train(
        #     params, self.dtrain_clf,
        #     num_boost_round=n
        # )

    def model_evaluation(self):
        self.preds = self.results.predict(self.dtest_clf)
        rmse = mean_squared_error(self.y_test, self.preds, squared=False)
        print(f"RMSE of the base model: {rmse:.3f}")


    def run(self):
        self.test_train_split()
        self.model_build()
        self.model_evaluation()

if __name__ == "__main__":
    xgboost_model = XgboostPipeline()
    xgboost_model.run()


