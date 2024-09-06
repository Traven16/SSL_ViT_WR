import numpy as np
import os 
import pickle
import sklearn

from sklearn.preprocessing import normalize

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

class DebugPrinter:
    def __init__(self, var, prefix, suffix):
        self.v = var
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, x):
        if self.v.args.debug_print:
            print(self.prefix, x, self.suffix)


class PatchCropper():
    def __init__(self, patch_size):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.h0, self.h1 = patch_size[0]//2, patch_size[1]//2 

    def _extract_single_patch(self, img, keypoint):
        x1, x0 = keypoint
        
        x0 = clamp(x0, self.h0, img.shape[0] - self.h0)
        x0 = slice(x0 - self.h0, x0 + self.h0)

        x1 = clamp(x1, self.h1, img.shape[1] - self.h1)
        x1 = slice(x1 - self.h1, x1 + self.h1)

        return img[x0, x1]
    
    def _patch(self, image, keypoint):
        x1, x0 = keypoint
        x0 = slice(x0 - self.h0, x0 + self.h0)
        x1 = slice(x1 - self.h1, x1 + self.h1)
        return image[x0, x1]


    def extract_patches(self, image, keypoints):

        if isinstance(keypoints, np.ndarray):
            assert keypoints.shape[1] == 2, "Unclear definition of keypoint given. 2-D dimensional array of keypoints expected."

            #return [self._extract_single_patch(image, keypoint) for keypoint in keypoints]
            return [self._patch(image, keypoint) for keypoint in keypoints]
        #ToDO: Add more cases here.

    def __call__(self, *args):
        return self.extract_patches(*args)


class LocalFeatureLoader():
    def __init__(self, dir: str):
        self.dir = dir
        self.files = os.listdir(dir)

    def load(self):
        descs = []
        for filename in self.files:
            filepath = os.path.join(self.dir, filename) 
            with open(filepath, "rb") as f:  
                d = pickle.load(f)
                descs.append(d)
        
        return descs, self.files
    
    def all(self):
        d, f = self.load()
        return np.concatenate(d, axis=0)
    

def hellinger_kernel(data):
    return normalize(np.sqrt(data),norm='l1', axis=1)

def powernorm(data, alpha):
    data = np.sign(data) * np.abs(data) ** alpha
    data = sklearn.preprocessing.normalize(data, norm="l2", axis=1)
    return data

