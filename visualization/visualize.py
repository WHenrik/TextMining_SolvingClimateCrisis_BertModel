import umap
import matplotlib.pyplot as plt
import umap.umap_ as umap           #to tackle some internal errors of umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class visualize_embeddings(object):
    r"""
        Object which is used for the low-dimensional visualization of the high-dimensional embeddings. 
    """

    def __init__(self):
        self.reducer=umap.UMAP()


    def fit(self,embeddings,scaled=False):
        """
        Receives a pandas DataFrame containing embedding vectors of length 768 and maps them in a low-dimensional space. 
        For a consistent representation of several datasets (e.g. training and test) this function should be called for every dataset once.

        Args:
            embeddings (:obj:`DataFrame[float]`):
                DataFrame of embedding vectors
            scaled (:obj:`bool`)
                In case incoming data should be standarized 
        """
        if scaled==True:
            embeddings = StandardScaler().fit_transform(embeddings)
        self.reducer.fit(embeddings)

    def show(self,embeddings,labels):
        """
        Visualizes the low-dimensional embedding in two dimensions.

        Args:
            embeddings (:obj:`DataFrame[float]`):
                Series of low-dimensional embeddings
        """

        embedding=self.reducer.transform(embeddings)
    
        plt.figure(figsize=[12,12])
    
        plt.scatter(embedding[:, 0],embedding[:, 1],c=[sns.color_palette()[x] for x in labels],s=2)
        plt.gca().set_aspect('equal', 'datalim')
        plt.show()
