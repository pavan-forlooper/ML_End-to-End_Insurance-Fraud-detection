from sklearn.cluster import KMeans
import logging
from kneed import KneeLocator
import pickle
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self, df):
        self.df = df

    def clustering_fn(self):
        try:
            df = self.df

            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
                kmeans.fit(df)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            logging.info("find_clusters.py: n_clusters found out")

            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            kmeans_model = KMeans(n_clusters=kn.knee, init='k-means++', random_state=42)

            logging.info("find_clusters.py: knee located")

            df['clusters'] = kmeans_model.fit_predict(df)
            # pca = PCA(n_components=2)
            # X_reduced = pca.fit_transform(x)

            # Plot the clusters
            #plt.plot(range(1, 11), wcss)
            #plt.show()

            with open("kmeans.pkl", "wb") as file:
                pickle.dump(kmeans_model, file)

            logging.info("find_clusters.py: model fit")

        except Exception as ex:
            logging.error("model.py: PROGRAMME FAILED, WITH :: %s" % ex)
        return df
