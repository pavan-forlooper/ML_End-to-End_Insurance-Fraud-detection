import numpy as np
import pandas as pd
import sklearn
import logging
import pickle
from data_preprocessing import Preprocess
from Clustering import Clustering
from training import Training
from prediction import Prediction
from testing_called_from_cloud import CloudTest

logging.basicConfig(filename='logs.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def main():
    # read data
    logging.info("starting")
    df = pd.read_csv(r'insuranceFraud.csv')
    #df_copy = df.copy()
    #df_copy2 = df.copy()

    # pre-processing
    pre_processing_instance = Preprocess(df)
    processed_df = pre_processing_instance.preprocessing_fn()
    # print(processed_df.info())
    # print(processed_df.columns, processed_df.head(5))

    #df_copy_dropped = df_copy.drop(['fraud_reported'], axis=1)
    #pre_processing_instance1 = Preprocess(df_copy_dropped)
    #processed_test_df = pre_processing_instance1.preprocessing_fn()
    # print(processed_test_df.info())
    # print(processed_test_df.columns, processed_test_df.head(5))

    # Cluster data
    clustering_instance = Clustering(processed_df)
    clustered_df = clustering_instance.clustering_fn()
    #print(clustered_df['clusters'].value_counts())

    #clustering_instance1 = Clustering(processed_test_df)
    #clustered_test_df = clustering_instance1.clustering_fn()
    #print(clustered_test_df['clusters'].value_counts())

    # print(clustered_df['clusters'].corr(clustered_test_df['clusters']))

    # TRAIN models
    training_instance = Training(clustered_df)
    training_instance.training_model()

    #test_cloud = CloudTest(pd.read_csv(r'insuranceFraud_test.csv'))
    #test_cloud_returned = test_cloud.main()
    #xgb_model = pickle.load(open('xgb_0.pkl', 'rb'))
    #print(xgb_model.predict(test_cloud_returned))

    # PREDICTION-test model
    #test_instance = Prediction(clustered_test_df)
    #df_test_returned = test_instance.testing_models()
    #print(df_test_returned)

    #test_cloud = CloudTest(df_copy)
    #test_cloud_returned = test_cloud.min()

    #print(test_cloud_returned)
    #print("final_test_cloud_correlation:", test_cloud_returned['predictions'].corr(processed_df['fraud_reported']))


if __name__ == "__main__":
    main()
    '''
    •	Data ingestion
    •	Data validation 
    •	Data cleaning
    •	Dividing Data into clusters.
    •	Train individual models for respective clusters
    •	Hyper parameter tuning of best performing model of each cluster.
    •	Save model. 
    •	Prediction – all data pre-processing
    •	Logging
    •	Deployment
    •	Processing client files for prediction'''
