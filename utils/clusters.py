import hdbscan
from sklearn.decomposition import PCA
import umap.umap_ as umap


def find_clusters(embeddings,min_cluster_size):
	"""
        Receives a pandas DataFrame containing embedding vectors of length 768 map them into an low-dimensional space 
        and finds the clusters in that space

        based on the idea in: https://umap-learn.readthedocs.io/en/latest/faq.html - section "From a more practical standpoint"
        
        Args:
            embeddings (:obj:`DataFrame[float]`):
                DataFrame of embedding vectors
            min_cluster_size (:obj:`int`):
            	Minimal cluster size

        Returns:
           :obj:`numpy array[int64]`: Cluster labels for each data point
    """

	n_components=min(len(embeddings),50)
	pca = PCA(n_components=n_components)
	embeddings=pca.fit_transform(embeddings)

	reducer = umap.UMAP(n_components=2)
	embeddings=reducer.fit_transform(embeddings)

	clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
	clusterer.fit(embeddings)

	return clusterer.labels_

