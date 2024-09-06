import bisect
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
import numpy as np
from dataclasses import dataclass
import os
import cv2
from skimage.filters.thresholding import threshold_sauvola
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
#from misc.utils import PatchCropper
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
from torch.utils.data import DataLoader, random_split






    #from documentSet import Document
class Document(Dataset):
    def __init__(self, filename, directory, page_id, d_patches=224):
        self.directory = directory
        self.filename = filename
        self.page_id = page_id
        self.keypoints = None
        self.descriptors = None
        self.encoding = None
        self.writer_id = None
        self.pseudo_targets = None
        self.pseudo_target = 0
        self.writer_id = None
        self.patch_paths = None
        self.to_tensor = ToTensor()
        self.img = None
        self.d_patches = d_patches
        self.transforms = None
        self.mode = "normal"
        

    def load_img_pil(self):        
        if self.img is None:
            self.img = Image.open(os.path.join(self.directory, self.filename)).convert("1")

    def __getitem__(self, idx):  
        if self.img is None:
            self.load_img_pil()  
        kp = self.keypoints[idx]
        if self.mode == "word":
            patch=self.word_image_to_patch(56,224)
        else:
            patch = self._patch(kp).convert("RGB")
        
        pseudo_target = self.pseudo_target #self.pseudo_targets[idx]
        if self.transforms is not None:
            patch = self.transforms(patch)
        return patch, self.page_id, self.writer_id
    
    def load_features_from_disk(self, dir, numpy=False, load_from_filename=False):
        path = self.filename.split(".")[0] if load_from_filename else str(self.page_id)
        try:
            d = torch.load(os.path.join(dir, path), map_location="cpu")
        except Exception as e:
            d = torch.zeros((1,384))
        if numpy:
            return d.numpy()
        else:
            return d
    
    
    def get_samples(self, num):
        if num == -1:
            num = len(self.keypoints)
        num = min(num, len(self.keypoints))
        idx = np.random.choice(np.arange(len(self.keypoints)), num, replace=False)
        samples = [self[i][0] for i in idx]
        return samples
    
    def word_image_to_patch(self, target_height=56, max_width=224):
        # assume that this "document" is a word image
        self.load_img_pil()
        old_width, old_height = self.img.size
        aspect_ratio = old_width / old_height
        
        new_width = old_width / old_height * target_height
        new_width = int(min(new_width, max_width))
        
        #new_width = max(max_width, int(target_height * aspect_ratio))
        
        resized_image = self.img.resize((new_width, target_height), Image.ANTIALIAS)
        
        # resized_image = ImageOps.invert(resized_image)
        
        new_image = Image.new('RGB', (224, 224), (0, 0, 0))
        new_image.paste(resized_image, (0, 0))
        new_image.paste(resized_image, (0, 56))
        new_image.paste(resized_image, (0, 112))
        new_image.paste(resized_image, (0, 168))
        # new_image.save("test.png")
                
        return new_image
        
       
    
    def _patch(self, keypoint):
        x1, x0 = keypoint
        h0 = h1 = self.d_patches // 2

        # Calculate the bounding box for cropping
        left = x0 - h0
        upper = x1 - h1
        right = x0 + h0
        lower = x1 + h1

        # Crop the image
        return self.img.crop((left, upper, right, lower))

    def __len__(self):
        return len(self.keypoints)


    def load(self, gray=True, binary=False):
        file_path = os.path.join(self.directory, self.filename)

        if gray or binary:
            img = cv2.imread(file_path, 0) #0 = grayscale
            if gray:
                return img
            if binary:
                return img.astype(np.bool)
        return cv2.imread(file_path)
            
    def patches(self, patchsize, gray=True, mode="cv2"):
        if isinstance(self.keypoints[0], cv2.KeyPoint):
            keypoints = np.array([p.pt for p in self.keypoints], dtype=np.int16)
        else:
            keypoints = self.keypoints

        patches = PatchCropper(patchsize).extract_patches(self.load(gray), keypoints)
        
        if mode == "pil":
            patches = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in patches]
            
        return patches
    
    def load_patches(self, num=None):
        
        if num is None:
            patch_paths = self.patch_paths
        
        idxs = np.random.choice(len(self.patch_paths), num, replace=False)
        patch_paths = self.patch_paths[idxs]
        patches = [self.to_tensor(Image.open(f)) for f in patch_paths]
        patches = torch.stack(patches)
        return patches
    

class DocumentSet():
    def __init__(self, directory, transforms=None, d_patches=224):
        self.directory = directory
        self.documents = {}
        # create a document instance for every document
        # for i, f in enumerate(os.listdir(directory)):
            
        #     self.documents[f] = Document(f, directory, i)
        #     self.documents[f].transforms=transforms
        
        file_count = sum(len(files) for _, _, files in os.walk(directory))  # Get the number of files
        with tqdm(total=file_count) as pbar:  # Do tqdm this way
            for root, dirs, files in os.walk(directory):  # Walk the directory
                
                for i, f in enumerate(files):
                    full_path = os.path.join(root, f)
                    doc_name = os.path.relpath(full_path, start=directory)
                    self.documents[doc_name] = Document(f, root, i, d_patches)
                    self.documents[doc_name].transforms = transforms
                    pbar.update(1)
        
        writer_ids = set([ d.filename.split("-")[0] for d in self.documents.values()])
        writer_ids = {k:i for i,k in enumerate(writer_ids)}
        for d in self.documents.values():
            d.writer_id = writer_ids[d.filename.split("-")[0]]
            
        self.docs = list(self.documents.values())

    def get_all(self, attribute: str):
        all = [getattr(doc, attribute) for fname, doc in self if getattr(doc, attribute) is not None]
        #all = np.concatenate(all, axis=0)
        return all
    
    def get_all_stacked(self, attribute):
        encs = self.get_all(attribute)
        if len(encs[0].shape)==1:
                encs = np.vstack(encs)
        else:
            encs = np.concatenate(encs, axis=0)
        return encs


    
    def get_descriptors(self, size=-1):
        all = [doc.descriptors for f_name, doc in self.documents.items() if doc.descriptors is not None]
        all = np.concatenate(all, axis=0)

        if size == -1 or size > len(all):
            return all
        
        idx = np.random.choice(np.arange(len(all)), size, replace=False)
        return all[idx]
  
    def as_dataset(self, d_patches, mode="page"):
        
        return ConcatDocumentDataset(self, d_patches, mode)

    def __getitem__(self, idx):
        return self.docs[idx]
    
    
    def get_writer_subset_descriptors(self, size=1):
        if size>=1:
            return self.get_descriptors()
        
        writers = [fname.split("-")[0] for fname, doc in self]
        idx = np.random.choice(np.arange(len(writers)), size=int(len(writers)*size), replace=False)
        sampled_writers = {writers[i] for i in idx}
        sampled_document_descriptors = [doc.descriptors for fname, doc in self if fname.split("-")[0] in sampled_writers]
        return np.concatenate(sampled_document_descriptors, axis=0)
    
    def values(self):
        return self.documents.values()

    def __iter__(self):
        return iter(self.documents.items())
    
    def __len__(self):
        return len(self.documents)
    
    def get_sample_windows(self, doc_index, num_samples):
        samples = self.docs[doc_index].get_samples(num_samples)
        samples = torch.stack(samples)
        return samples
        

class StyleSampleDataset(Dataset):
    def __init__(self, dir_documents, keypoint_filters, num_samples, transforms):
        self.document_set = DocumentSet(dir_documents, ToTensor())
        for op in keypoint_filters:
            op(self.document_set)
        self.writer_docs = {}
        for fname, doc in self.document_set:
            writer = fname.split("-")[0]
            if writer in self.writer_docs:
                self.writer_docs[writer].append(doc)
            else:
                self.writer_docs[writer] = [doc]
        
        for writer, doc_list in self.writer_docs.items():
            self.writer_docs[writer] = ConcatDataset(doc_list)
            
        self.num = num_samples
        self.transforms = transforms
    
    def __getitem__(self, writer_id):
        if isinstance(writer_id, list):
            samples = [self[i] for i in writer_id]
            samples = torch.stack(samples, dim=0)
        else:    
            samples = self.get_sample_windows(writer_id, self.num)
            
        return self.transforms(samples)
    
    def __len__(self):
        return len(self.document_set)
    
    def get_sample_windows(self, writer_id, num_samples):
        # get correct dataset
        dset = self.writer_docs[writer_id]
        
        # sample
        idx = np.random.choice(np.arange(len(dset)), num_samples, replace=False)
        samples = [dset[i][0] for i in idx]
        samples = torch.stack(samples)
        
        return samples
        
        

class ConcatDocumentDataset(ConcatDataset):
    """ We extend ConcatDataset to also return indices"""
    def __init__(self, document_set, d_patches=256, mode="page", transforms=None):
        for doc in document_set.values():
            doc.d_patches = d_patches
            if mode=="page":
                doc.pseudo_target = doc.page_id
            elif mode=="writer":
                doc.pseudo_target = doc.writer_id
        self.documentset = document_set
        self.transforms = transforms
        super().__init__(self.documentset.values())

    def __getitem__(self, idx):
        patch, pageid, writerid = super().__getitem__(idx)
        patch = self.transforms(patch)
        
        return patch, pageid



