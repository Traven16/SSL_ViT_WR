from components import *
import numpy as np
from skimage.filters.thresholding import threshold_sauvola
import cv2
from dataclasses import dataclass
from features import SIFT_Descriptor, PseudoTargetsKMeans

from constants import *


@dataclass
class PatchSettings:
    n_per_page: int = 1000
    keypoint_mode: str = KEYPOINT_MODE_SIFT
    save_mode: str = PATCH_SAVEMODE_WRITER
    d_patches: int = 256
    d_stride: int = 128
    d_min_distance: int = 15
    n_classes: int = 1000
    n_batchsize: int = 500000


class SIFTKeypoints(MappingModule):
    def __init__(self, d_patches = 32, t_contrast = 0):
        self.sift = cv2.SIFT_create(contrastThreshold=t_contrast)
        self.d_patches = d_patches

    def process(self, document):
        img = document.load()
        document.keypoints = self.sift.detect(img)

class PerPageKeypointSampler(MappingModule):
    def __init__(self, n_per_page, attributes=["keypoints", "descriptors"]):
        self.n_per_page = n_per_page
        self.attributes = attributes

    @staticmethod
    def apply_to_attribute(document, attr, idxs):
        if getattr(document, attr) is not None:
            a = getattr(document, attr)
            setattr(document, attr, [a[i] for i in idxs])

    def process(self, document):
        idxs = np.random.choice(len(document.keypoints), 
                                min(len(document.keypoints), self.n_per_page),
                                replace=False)
        
        for attr in self.attributes:
            self.apply_to_attribute(document, attr, idxs)



class GridKeypoints(MappingModule):
    
    def __init__(self, d_patches, d_stride):
        self.d_patches = d_patches
        self.d_stride = d_stride
        
    def process(self, document):
        img = document.load()
        height, width = img.shape
        s = self.d_stride
        h_ = self.d_patches//2
        
        h_range = list(range(h_, height - h_, s))
        w_range = list(range(h_, width - h_, s))
 
        keypoints = [[h,w] for h in h_range for w in w_range]
        if len(keypoints) == 0:
            keypoints = [[0,0]]
        keypoints = np.array(keypoints)
        document.keypoints = keypoints



class ForegroundKeypoints(MappingModule):
    def process(self, document):
        img = document.load()
        white_pixel_indices = np.where(img == 255)
        keypoints = np.column_stack(white_pixel_indices)
        document.keypoints = keypoints
        
class WhitenessFilter(MappingModule):
    def __init__(self, d_patches, r_white):
        self.d_patches = d_patches
        self.n_white = (d_patches**2) * r_white
        self.n_workers = 1
        #print(self.n_white)
        
    def process(self, document):
        document.load_img_pil()
        new_kps = []
        for kp in document.keypoints:
            if sum(document._patch(kp).getdata()) > self.n_white:
                new_kps.append(kp)
                
        document.keypoints = new_kps
        document.img = None

class KeypointFilter(MappingModule):
    def __init__(self, d_patches = 32, n_min_distance = 0, b_foreground = False, mode="cv2"):
        self.d_patches = d_patches
        self.n_min_distance = n_min_distance
        self.b_foreground = b_foreground
        self.mode = mode

    def process(self, document):
        image = document.load()

        # We define an array to indicate valid positions.
        # 1 = valid, 0 = invalid
        valid = np.ones_like(image)  

        if self.b_foreground:
            
            # Remove keypoints that lie on background pixels by projecting
            # a binarized version of the image ontop of the detected keypoints.
            thresh = threshold_sauvola(image, window_size=41)
            valid[image < thresh] = 0
        
        # Remove keypoints that lie to close to the border, such that the window
        # sampled around it would be out of bounds. 
        b = self.d_patches // 2
        valid[:b], valid[-b:], valid[:,:b], valid[:, -b:] = 0, 0, 0, 0 

        new_kps = []
        
        min_dist = self.n_min_distance

        for kp in document.keypoints:
            if self.mode == "cv2":
                x0, x1 = (int(kp.pt[1]), int(kp.pt[0])) # Reminder: OpenCV keypoints are ordered "wrong"
            else:
                x0, x1 = kp[1], kp[0]
                
            if valid[x0, x1] == 1:
                valid[x0-min_dist:x0+min_dist, x1-min_dist:x1+min_dist] = 0 # avoid too similar patches in the future
                new_kps.append(kp)

        document.keypoints = new_kps


       
class PatchExtractionSIFTKeypoints(ModuleWrapper):
    def __init__(self, 
                 dir_patches="",
                 mode="writer",
                 n_patches_per_page=2500,
                 d_patches=256, 
                 n_min_distance=15,
                 n_workers=16,):
                
        modules = [
            SIFTKeypoints(d_patches=d_patches),
            KeypointFilter(d_patches=d_patches, n_min_distance=n_min_distance),
            PerPageKeypointSampler(n_per_page=n_patches_per_page),
        ]
        super().__init__(modules)
        
class PatchExtractionGrid(ModuleWrapper):
    def __init__(self, d_patches, d_stride, dir_patches, mode):
        modules = [
            GridKeypoints(d_patches=d_patches, d_stride=d_stride),
            #LoadImages(), 
            #WhitenessFilter(d_patches=d_patches, r_white=0.025),
        ]
        super().__init__(modules)
        
        
class PatchExtractionSingleShot(ModuleWrapper):
    def __init__(self, d_patches, d_stride, dir_patches, mode):
        modules = [
            GridKeypoints(d_patches=d_patches, d_stride=d_stride),
            WhitenessFilter(d_patches=d_patches, r_white=0.015),
            PerPageKeypointSampler(10, ["keypoints"])
        ]
        super().__init__(modules)


class PatchExtractionSIFTCluster(PatchExtractionSIFTKeypoints):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        
        # insert new modules before the patch_writer
        patch_writer = self.modules[-1]
        pseudo_sift = [
            SIFT_Descriptor(b_hellinger_norm=True),
            PseudoTargetsKMeans(n_clusters=kwargs["n_classes"], n_batchsize=kwargs["n_batchsize"]),   
        ]
        self.modules = self.modules[:-1] + pseudo_sift + [patch_writer]
    