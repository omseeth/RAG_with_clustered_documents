"""
kmeans_cluster.py

This script helps with clustering documents with kmeans based on the distance between their vector representations (embeddings). To find the ideal number of clusters, silhouette coefficients are calculated for a range K of possible clusters.
"""

import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class KMeansCluster:
    """
    Vector based clustering with kmeans based on an optimal number of clusters according to the silhouette coefficient for a given set of vectors.
    """

    def __init__(self, vector_db: pd.DataFrame, embedding_model: str, num_k: int = 4, determine: bool = False, K: int = 30, verbose: bool = False):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.sil_scores = []
        self.inertias = []
        self.K = range(2, K)
        self.verbose = verbose
        if determine:
            self.k = self.determine_k_for_clusters(self.K)
            if self.verbose:
                self.plot_silhouettes()
        else:
            self.k = num_k
        self.df_split = self.kmeans(self.k)
    
    def determine_k_for_clusters(self, K: range) -> int:
        """
        Compute the silhouette coefficient and the elbow method for cluster in the range of K.

        Args:
            K: int to determine range of different numbers of clusters

        Returns:
            k: the optimal number of clusters according to the silhouette 
                coefficient
        """
        # sklearn KMeans requires a 2D input
        vectors = np.vstack(self.vector_db[self.embedding_model].to_numpy())
        sil_scores = {}
        # Ignoring runtime warning for zero division and overlfow
        np.seterr(all="ignore")

        for k in tqdm(K, desc="Calculating silhouette scores and inertias", total=len(K)):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(vectors)
            score = silhouette_score(vectors, kmeans.labels_)
            sil_scores[k] = score
            self.sil_scores.append(score)
            self.inertias.append(kmeans.inertia_)
        
        sorted_sil_scores = dict(sorted(sil_scores.items(), key=lambda item: item[1], reverse=True))
        k = next(iter(sorted_sil_scores))

        return k
    
    def kmeans(self, k: int) -> pd.DataFrame:
        """ 
        Fits the class' embedding vectors to k clusters with k-means++.

        Args:
            k, number of clusters

        Returns:
            df, with "cluster" column and "split" column, where each document 
                is assigned to a cluster k and a train/test split
        """
        print("Clustering documents with scikit-learn's k-means++.")

        # Ignoring runtime warning for zero division and overlfow
        np.seterr(all="ignore")

        # sklearn KMeans requires a 2D input
        vectors = np.vstack(self.vector_db[self.embedding_model].to_numpy())

        # Kmeans with k-means++
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(vectors)

        # Plotting the resulting clusters
        if self.verbose:
            self.plot_clusters_tsne(vectors, kmeans)

        self.vector_db["cluster"] = kmeans.fit_predict(vectors)

        # Splitting data for classifier training
        df_split = self.splitting_df(self.vector_db)

        # Save data
        df_split.to_json("data/vector_db.jsonl", orient="records", lines=True)

        return df_split
    
    def splitting_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Helper method to create a train & test split for the data based on the cluster classes.

        Args:
            df, with cluster classes
        
        Returns:
            df, with additional column "split" indicating train or test example
        """
        # Data is already shuffled
        split = []
        threshold = int(len(df) * 0.9) # 90/10 train/test split

        for i in range(len(df)):
            if i < threshold:
                split.append("train")
            else:
                split.append("test")

        df["split"] = split

        # Checking if the training split contains all cluster classes
        train_clusters = df[df["split"] == "train"]["cluster"].unique()
        all_clusters = set(df["cluster"].unique())
        missing_clusters = all_clusters - set(train_clusters)

        # Readjusting the dataframe
        for cluster in missing_clusters:
            # Find one example of the missing cluster from the test set
            idx = df[(df["split"] == "test") & (df["cluster"] == cluster)].index

            if len(idx) > 0:
                df.loc[idx[0], "split"] = "train"

        return df

    def plot_silhouettes(self):
        """
        Simple plots for silhouette coefficients and inertias.
        """
        plt.plot(self.K, self.sil_scores)
        plt.title("Silhouette score vs. number of clusters")
        plt.show()

        plt.plot(self.K, self.inertias)
        plt.title("Inertia (Elbow Method)")
        plt.show()
    
    def plot_clusters_tsne(self, vectors: np.ndarray, kmeans: KMeans):
        """
        Reduces dimensionality of embedding vectors with t-SNE and plots the resulting vectors according to their clusters in two dimensions.

        Args:
            vectors, original high-dimensional embeddings
            kmeans, fitted KMeans object
        """
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(vectors)
        # Each vector in vectors[i] belongs to cluster kmeans.labels_[i]
        labels = kmeans.labels_
        num_clusters = len(set(labels))

        colors = plt.cm.get_cmap("tab10", num_clusters)

        plt.figure(figsize=(10, 8))
        for cluster in range(num_clusters):
            idx = labels == cluster
            plt.scatter(reduced[idx, 0], reduced[idx, 1],
                        label=f"Cluster {cluster}", s=50,
                        color=colors(cluster))
        
        plt.title(f"KMeans Clusters Visualized with t-SNE")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """
    Example usage of determining clusters and clustering the embedded texts.
    """

    EMBEDDING_MODEL = "nomic-embed-text"
    df = pd.read_json("data/vector_db.jsonl", lines=True)
    K_range = 50
    self_determine_k = False
    num_k = 4
    verbose = False
    kmeans_cluster = KMeansCluster(df, EMBEDDING_MODEL, num_k, self_determine_k, K_range, verbose)
