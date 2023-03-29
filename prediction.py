import pickle
import pandas as pd

class Prediction:
    def __init__(self, df):
        self.df = df

    def testing_models(self):
        df = self.df
        df_copy = df.copy()
        for i in df['clusters'].unique():
            cluster = df[df['clusters'] == i]
            with open("xgb_{}.pkl".format(i), "rb") as file:
                xgb_model = pickle.load(file)
            predicted = xgb_model.predict(cluster)
            cluster['predictions'] = pd.Series(predicted, index=cluster.index)
            df_copy.loc[df['clusters'] == i, 'predictions'] = cluster['predictions']
        return df_copy['predictions']

