from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import logging
import pickle
from imblearn.over_sampling import RandomOverSampler


class Training:
    def __init__(self, df):
        self.df = df

    def training_model(self):
        df = self.df

        try:
            df_copy = df.copy()


            for i in df['clusters'].unique():
                #print(i)
                cluster = df[df['clusters'] == i]
                x = df.drop(['fraud_reported'], axis=1)
                y = df['fraud_reported']

                # Handle imbalance
                random_sample = RandomOverSampler()
                x, y = random_sample.fit_resample(x, y)
                #print(x.shape, y.shape)

                xtrain, xtest, ytrain, ytest = train_test_split(x, y)

                xgb_model = XGBClassifier()
                xgb_model.fit(xtrain, ytrain)
                #print(xgb_model.score(xtest, ytest))
                logging.info("model_training.py: model trained successfully")

                with open("xgb_{}.pkl".format(i), "wb") as file:
                    pickle.dump(xgb_model, file)
                logging.info("model_training.py: models saved")

        except Exception as ex:
            logging.error("model_training.py: PROGRAMME FAILED, WITH :: %s" % ex)
