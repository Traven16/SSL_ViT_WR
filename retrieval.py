from multiprocessing import Lock, Pool, Value
from sklearn.calibration import LinearSVC
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import torch
import numpy as np
import logging
from parmap import map as parmap
from reranking.sgr import sgr_reranking
from reranking.krnn import kRNN

import warnings

from sklearn.exceptions import ConvergenceWarning

from components import MappingModule
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from PIL import Image

from tqdm import tqdm

class ESVM(MappingModule):
    
    def __init__(self, enc_train, C):
        self.train_enc = enc_train
        self.C = C
        self.labels = np.zeros(len(self.train_enc) + 1)
        self.labels[0] = 1
        self.same = False
        self.num_workers = 6
    
    def __call__(self, docset, override_encs=None):
        if override_encs is not None:
            for i, (_,d) in enumerate(docset):
                d.encoding = override_encs[i] 
        return super().__call__(docset)
            
    def process(self, doc):
        to_classify = np.zeros((len(self.labels),self.train_enc.shape[1]),
                           dtype=self.train_enc.dtype)
        to_classify[1:] = self.train_enc
        esvm = LinearSVC(C=self.C, class_weight='balanced',dual='auto')
        to_classify[0] = doc.encoding
        
        esvm.fit(to_classify, self.labels)
        
        x = normalize(esvm.coef_, norm='l2')
        doc.encoding = x


class Retrieval(): 
    def __init__(self, rerank, dims=[384],label_split="-", metric="cosine"):
        """
        Class to handle Retrieval.

        Parameters:
        - rerank (bool): Indicates whether reranking is enabled or not.
        - batchsize (bool or int): The batch size for incremental PCA. If False, incremental PCA is disabled.
        - dims (list): The dimensions for the retrieval object.
        - label_split (str): The character used to split labels.
        - metric (str): The metric used for retrieval.

        Returns:
        None
        """
        self.rerank = rerank
        self.dims=dims
        self.c=label_split
        self.metric = metric
       

    # def compute_indices(self, encs, encs2=None):
    #     """ This is not used anymore. Check this out if you have memory issues with cosine distance """
    #     sims = pairwise_distances(X=encs,Y=encs2, metric=self.metric, n_jobs=15)
    #     sims = 1 - sims

    #     np.fill_diagonal(sims, -np.finfo(sims.dtype).max)
    #     # sort each row of the similarity matrix in descending order
    #     indices = np.argsort(sims)[:, ::-1]
        
    #     #joblib.dump(indices, "indices.pkl")
    #     return indices, sims
    
    @staticmethod
    def compute_top1_and_map(tens, labels):
        """
        Compute the top-1 accuracy and mean average precision (mAP) for a given set of tensors and labels.
        Args:
            tens (torch.Tensor): The encodings to evaluate.
            labels (List[int]): The list of writer identifiers. Ensure same ordering as in tens.
        Returns:
            Tuple[float, float]: A tuple containing the top-1 accuracy and mAP.
        Raises:
            RuntimeWarning: If no matching labels are found, a RuntimeWarning is raised.
        """
        if not isinstance(tens, torch.Tensor): # let's try to be gracious if tens is a numpy array
            tens = torch.tensor(tens)
        labels = torch.tensor([int(l) for l in labels])
        tens = torch.nn.functional.normalize(tens, p=2, dim=1) # better safe than sorry
        cosine_sim = tens @ tens.T # if you get memory issues, check out self.compute_inidices above (beware that it computes distance, not similarity)
        
        # Ranking
        cosine_sim.diagonal().fill_(-float('inf')) # ensure self-sim is lowest
        ranking = torch.argsort(cosine_sim, descending=True)
        ranking = labels[ranking]
        ranking = ranking[:, :-1] # remove self
        
        aps = []
        correct = 0
        for i in range(len(ranking)):
            relevant = (ranking[i] == labels[i]) # get binary list whether doc is relevant
            if relevant.sum() == 0:
                continue #skip distractors
            
            correct += relevant[0] # is highest ranked doc relevant?
            precision_at_k = torch.cumsum(relevant.float(), dim=0) / torch.arange(1, len(relevant) + 1, device=relevant.device)
            aps.append(torch.mean(precision_at_k[relevant])) # store average precision
        
        if len(aps) == 0:
            raise RuntimeWarning("No matching labels found. Returning 0 for mAP and Top1")
            
        return correct / len(aps), sum(aps) / len(aps)

    
    def log(self, name, top1, map):
        """
        Logs the name, top1 accuracy, and mAP (mean Average Precision) to the console and logging file.

        Parameters:
        - name (str): The name of the log.
        - top1 (float): The top1 accuracy value.
        - map (float): The mAP value.

        Returns:
        None
        """
        print(f"{name} \t Top1: {top1:.2%} \t mAP: {map:.2%}")
        logging.info(f"{name} \t Top1: {top1:.2%} \t mAP: {map:.2%}")
        
        
    def evaluate_(self, describe_str, train_encodings, train_labels, test_encodings, test_labels):
        """
        Evaluate the performance of the model on train and test data. Automatically logs the results with matching describtor string.
        
        Parameters:
        - describe_str: str, a description string for logging purposes
        - train_encodings: numpy array, encodings of the train data
        - train_labels: numpy array, labels of the train data
        - test_encodings: numpy array, encodings of the test data
        - test_labels: numpy array, labels of the test data
        """
        top1_train, map_train = self.compute_top1_and_map(torch.tensor(train_encodings), train_labels)
        self.log(f"train_{describe_str}", top1_train, map_train)   
        
        top1_test, map_test = self.compute_top1_and_map(torch.tensor(test_encodings), test_labels)
        self.log(f"test_{describe_str}", top1_test, map_test)     
    
    
    def evaluate(self, train_docset, test_docset):
        """
        Evaluate the performance of the model using train and test document sets.
        Parameters:
        - train_docset (DocumentSet): The document set used for training.
        - test_docset (DocumentSet): The document set used for testing.
        Returns:
        None
        """
        train_encs = train_docset.get_all_stacked("encoding")
        test_encs = test_docset.get_all_stacked("encoding")
        
        train_labels = train_docset.get_all("writer_id")        
        test_labels = test_docset.get_all("writer_id")
        
        self.evaluate_("PrePCA", train_encs, train_labels, test_encs, test_labels)
        
        # Fit PCA
        for d in self.dims:
            pca = PCA(n_components=d, whiten=True)
            pca.fit(train_encs)
            train_encs_pca = pca.transform(train_encs)
            test_encs_pca = pca.transform(test_encs)
            
            self.evaluate_(f"PCA^t_{d}", train_encs_pca, train_labels, test_encs_pca, test_labels)


class VisualizeRetrieval(Retrieval):
    """Might not work"""
    def visualize(self, docset, dump):
        encs = docset.get_all("encoding")
        if len(encs[0].shape)==1:
            encs = np.vstack(encs)
        else:
            encs = np.concatenate(encs, axis=0)
        
                   
        indices, dists = self.compute_indices(encs)
        docs = list(docset.values())
        
        for i, idx_group in tqdm(enumerate(indices)):
            # Assuming each img is a PIL Image
            img_i, _ = docs[i][2]
            img_width, img_height = img_i.size

            # Create a new image to hold the composite
            # The width is img_width * 6 because we have img_i + 5 img_j
            composite_image = Image.new('RGB', (img_width * 6, img_height))

            # Place img_i on the far left
            composite_image.paste(img_i, (0, 0))

            # Place the next 5 images (img_j) next to img_i
            for j in range(5):
                img_j_index = idx_group[j]  # Assuming compute_indices returns indices of the closest images
                img_j, _ = docs[img_j_index][2]
                composite_image.paste(img_j, ((j+1) * img_width, 0))

            # Save or display the composite image
            composite_image_path = f'{dump}/{docs[i].filename}.jpg'  # Change the path accordingly
            composite_image.save(composite_image_path)
                
