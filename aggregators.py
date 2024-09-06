from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from components import TrainingModule, MappingModule
import numpy as np
import torch
import joblib


class VLAD(TrainingModule):
    """
    Vector of Locally Aggregated Descriptors (VLAD) encoding for image or document descriptors.

    Attributes:
        n_clusters (int): Number of clusters for KMeans.
        n_batchsize (int): Batch size for KMeans.
        gmp (bool): Whether to use generalized max pooling.
        gamma (float): Regularization parameter for Ridge regression used in GMP.
        powernorm (bool): Whether to apply power normalization to the output encoding.
        from_dir (str): Optional directory from which to load precomputed descriptors.
        n_workers (int): Number of parallel workers (default is 1).

    Methods:
        assignments(kdt, descriptors, clusters):
            Compute assignment of descriptors to nearest clusters.
        
        _train_impl(train_docset):
            Trains the KMeans model using the provided training dataset.
        
        process(document):
            Processes a document, encoding its descriptors based on trained cluster centers.
        
        process_helper(doc, cluster_centers, kdt):
            Helper function to process and encode descriptors for a single document.
    """

    def __init__(self, n_clusters=100, n_batchsize=1000000, gmp=True, gamma=1000, powernorm=True, from_dir=None):
        """
        Initializes the VLAD class with clustering and encoding parameters.
        
        Args:
            n_clusters (int): Number of clusters to use for KMeans.
            n_batchsize (int): Batch size for KMeans fitting.
            gmp (bool): Apply generalized max pooling if True.
            gamma (float): Regularization parameter for Ridge regression.
            powernorm (bool): Apply power normalization if True.
            from_dir (str, optional): Directory to load precomputed descriptors from.
        """
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=1, batch_size=n_batchsize, verbose=1)
        self.gmp = gmp
        self.gamma = gamma
        self.powernorm = powernorm
        self.n_clusters = n_clusters
        self.from_dir = from_dir
        self.n_workers = 1

    def assignments(self, kdt, descriptors, clusters):
        """
        Computes the assignments of descriptors to the nearest clusters using KDTree.
        
        Args:
            kdt (KDTree): KDTree built from cluster centers.
            descriptors (ndarray): Array of document descriptors.
            clusters (ndarray): Array of cluster centers.
        
        Returns:
            ndarray: Binary matrix of assignments of descriptors to clusters.
        """
        descriptors = np.nan_to_num(descriptors, nan=0)
        dists, indices = kdt.query(descriptors, k=1)
        assignment = np.zeros((len(descriptors), len(clusters)))
        assignment[np.arange(len(descriptors)), indices.flatten()] = 1
        return assignment

    def _train_impl(self, train_docset):
        """
        Trains the VLAD model by fitting KMeans on the training dataset.
        
        Args:
            train_docset: Dataset containing descriptors for training.
        """
        self.kmeans.fit(np.concatenate(train_docset.get_all("descriptors"), axis=0))
        self.kdt = [KDTree(self.kmeans.cluster_centers_, metric='euclidean')]
        self.cluster_centers = [self.kmeans.cluster_centers_]
        joblib.dump(self.cluster_centers, "cluster_centers.pkl")

    def process(self, document):
        """
        Processes a document by encoding its descriptors using trained cluster centers.
        
        Args:
            document: A document object containing descriptors to be processed.
        """
        for centroids, kdt in zip(self.cluster_centers, self.kdt):
            self.process_helper(document, centroids, kdt)

    def process_helper(self, doc, cluster_centers, kdt):
        """
        Helper method to process and encode descriptors for a document.
        
        Args:
            doc: Document object containing descriptors.
            cluster_centers (ndarray): Array of cluster centers.
            kdt (KDTree): KDTree built from cluster centers.
        """
        if self.from_dir is not None:
            try:
                desc = doc.load_features_from_disk(self.from_dir)
                assert desc.shape[-1] == 384
            except Exception as e:
                desc = torch.zeros((1, 384))
        else:
            desc = doc.descriptors
        T, D = desc.shape

        a = self.assignments(kdt, desc, cluster_centers)
        f_enc = np.zeros((D * len(cluster_centers)), dtype=np.float32)
        num_clusters = cluster_centers.shape[0]
        all_residuals = desc[:, np.newaxis, :] - cluster_centers

        for k in range(num_clusters):
            mask = a[:, k] > 0
            nn = np.sum(mask)
            if nn == 0:
                continue
            res = all_residuals[mask, k, :]
            if self.gmp:
                clf = Ridge(alpha=self.gamma, fit_intercept=False, solver='sparse_cg', max_iter=500)
                clf.fit(res, np.ones((nn,)))
                f_enc[k * D:(k + 1) * D] = clf.coef_
            else:
                f_enc[k * D:(k + 1) * D] = torch.sum(res, axis=0)

        if self.powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        f_enc = normalize(f_enc.reshape(1, -1), norm='l2')

        if doc.encoding is None:
            doc.encoding = f_enc
        else:
            doc.encoding = np.concatenate([doc.encoding, f_enc], axis=1)



class bVLAD(VLAD):
    """
    Bagged VLAD (bVLAD) encoding.

    Extends VLAD to use bagging, where multiple subsets of clusters are sampled to form bags.

    Attributes:
        n_bags (int): Number of bags.
        ratio_per_bag (float): Ratio of clusters per bag.
    """

    def __init__(self, n_bags=20, ratio_per_bag=0.1, n_clusters=1000, n_batchsize=1000000, gmp=True, gamma=1000, powernorm=True, from_dir=None):
        """
        Initializes the bVLAD class with bagging parameters and VLAD parameters.
        
        Args:
            n_bags (int): Number of bags to create.
            ratio_per_bag (float): Proportion of clusters in each bag.
            n_clusters (int): Number of clusters for KMeans.
            n_batchsize (int): Batch size for KMeans fitting.
            gmp (bool): Apply generalized max pooling if True.
            gamma (float): Regularization parameter for Ridge regression.
            powernorm (bool): Apply power normalization if True.
            from_dir (str, optional): Directory to load precomputed descriptors from.
        """
        super().__init__(n_clusters, n_batchsize, gmp, gamma, powernorm)
        self.n_bags = n_bags
        self.ratio_per_bag = ratio_per_bag
        self.n_workers = 1
        self.from_dir = from_dir

    def _train_impl(self, train_docset):
        """
        Trains the bVLAD model using bagging, where multiple bags of clusters are sampled.
        
        Args:
            train_docset: Dataset containing descriptors for training.
        """
        cluster_centers, kdts = [], []
        descs = train_docset.get_descriptors()
        self.kmeans.fit(descs)
        all_centers = self.kmeans.cluster_centers_
        bag_size = int(self.n_clusters * self.ratio_per_bag)
        for i in range(self.n_bags):
            random_indices = np.random.choice(self.n_clusters, bag_size, replace=False)
            bag_centers = all_centers[random_indices]
            cluster_centers.append(bag_centers)
            kdts.append(KDTree(bag_centers, metric="euclidean"))

        self.kdt = kdts
        self.cluster_centers = cluster_centers

    
class SumPooling(MappingModule):
    """
    A class representing a sum pooling aggregator.
    Args:
        from_dir (str, optional): The directory path to load features from disk. Defaults to None.
    Methods:
        process(document): Process the given document by summing the descriptors.
    Attributes:
        from_dir (str): The directory path to load features from disk.
    """
    def __init__(self, from_dir=None):
        self.from_dir = from_dir
        
    def process(self, document):
        if self.from_dir is not None:
            desc = document.load_features_from_disk(self.from_dir)
        else:
            desc = document.descriptors
        if desc.shape[-1] != 384:
            print("break")
        if len(desc.shape) == 1:
            print("break")
        document.encoding = desc.sum(axis=0)