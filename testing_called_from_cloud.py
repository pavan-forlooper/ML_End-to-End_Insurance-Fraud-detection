import logging
from data_preprocessing import Preprocess
from Clustering import Clustering
from prediction import Prediction

logging.basicConfig(filename='logs.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logging.captureWarnings(True)

class CloudTest:
    def __init__(self, df):
        self.df = df

    def main(self):
        # read data
        logging.info("starting")
        df_copy_dropped = self.df
        # pre-processing

        pre_processing_instance1 = Preprocess(df_copy_dropped)
        processed_test_df = pre_processing_instance1.preprocessing_fn()
        # print("process", processed_test_df.info())
        # print(processed_test_df.columns, processed_test_df.head(5))

        # Cluster data

        clustering_instance1 = Clustering(processed_test_df)
        clustered_test_df = clustering_instance1.clustering_fn()

        # print("cluster", clustered_test_df['clusters'].value_counts())

        # PREDICTION-test model
        test_instance = Prediction(clustered_test_df)
        df_test_returned = test_instance.testing_models()
        return df_test_returned
