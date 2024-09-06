
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from components import *
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from constants import *
from misc.documentSet import Document



class SIFT_Descriptor(MappingModule):
    def __init__(self, b_hellinger_norm):
        self.sift = cv2.SIFT_create()
        self.b_hellinger_norm = b_hellinger_norm

    def process(self, document):
        def update_kp(kp: cv2.KeyPoint):
            kp.angle = 0
            return kp
        
        img = document.load()
        # Update keypoints' angles to zero
        kps = [update_kp(kp) for kp in document.keypoints]
        
        # Compute descriptors based on keypoints
        _, desc = self.sift.compute(img, kps)

        if self.b_hellinger_norm:
            desc = sklearn.preprocessing.normalize(desc, norm='l1')
            desc = np.sign(desc) * np.sqrt(np.abs(desc))
            desc = sklearn.preprocessing.normalize(desc, norm='l2')
        
        # Update document's descriptors
        document.descriptors = desc
        document.keypoints = np.array([p.pt for p in document.keypoints], dtype=np.int16)
        
class HellingerKernel(MappingModule):
    def process(self, document):
        desc = document.descriptors
        desc = sklearn.preprocessing.normalize(desc, norm='l2')
        desc = np.sign(desc) * np.sqrt(np.abs(desc))
        desc = sklearn.preprocessing.normalize(desc, norm='l2')
        document.descriptors = desc
        

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


class PatchWriter(MappingModule):
    def __init__(self, d_patches, dir_patches, mode="page", ):
        self.d_patches = d_patches
        self.dir_patches = dir_patches
        self.mode = mode
        
    
    def process(self, document: Document):
        if self.mode == PATCH_SAVEMODE_NONE:
            return 
        
        def make_and_save(patch_path, patch_name, document):
            os.makedirs(patch_path, exist_ok=True)
            cv2.imwrite(patch_name, patch)
            #document.patch_paths.append(patch_name)

        document.patch_paths = []
        patches = document.patches(self.d_patches)
        if self.mode == PATCH_SAVEMODE_PAGE :
            for index, patch in enumerate(patches):
                patch_path = join(self.dir_patches, document.filename.split(".")[0])
                patch_name = join(patch_path, f"{index}.png")
                make_and_save(patch_path, patch_name, document)
                
            
        elif self.mode == PATCH_SAVEMODE_CLUSTER:
            for index, (patch, target) in enumerate(zip(patches, document.pseudo_targets)):
                patch_path = join(self.dir_patches, str(target))
                patch_name = join(patch_path, f"{document.filename.split('.')[0]}_{index}.png")
                make_and_save(patch_path, patch_name, document)

        elif self.mode == PATCH_SAVEMODE_WRITER :
            for index, patch in enumerate(patches):
                patch_path = join(self.dir_patches, document.filename.split("-")[0])
                patch_name = join(patch_path, f"{document.filename.split('.')[0]}_{index}.png")
                make_and_save(patch_path, patch_name, document)
                

class PseudoTargetsKMeans(TrainingModule):
    def __init__(self, n_clusters=1000, n_init=1, n_batchsize=100000):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                      n_init=n_init, 
                                      batch_size=n_batchsize)
          

    def _train_impl(self, train_docset):
        descriptors = train_docset.get_descriptors()
        self.kmeans.fit(descriptors)


    # This is used to inference
    def process(self, document):
        document.pseudo_targets = self.kmeans.predict(document.descriptors)

class  FeatureExtraction():
    def __init__(self, model, d_patches, transf, dump, args):
        self.model = model.cuda()
        self.transforms = transf
        self.dir_cls = os.path.join(dump, "cls")
        self.dir_fg = os.path.join(dump, "fg")
        self.dir_all = os.path.join(dump, "all")
        self.d_patches = d_patches
        self.n_workers = 10
        self.args=args
        self.avg2d = torch.nn.AvgPool2d(args.patch_size, args.patch_size)
        self.n_fg = args.extract_threshold_foreground / (args.patch_size**2)
        
        #print("Saving to", self.dir_fg)
        
        for dir_path in [self.dir_cls, self.dir_fg, self.dir_all]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_features(self, F_fg, F_all, F_cls, fname):
        
        F_all = torch.vstack(F_all)
        F_fg = torch.vstack(F_fg)
        F_fg = F_all[F_fg]
        
        F_all = F_all.reshape((-1, F_all[0].shape[-1]))
        if len(F_fg) == 0:
            F_fg = torch.zeros((1, F_all.shape[-1]), dtype=torch.float32)
        else:
            F_fg = F_fg.reshape((-1, F_fg[0].shape[-1]))
        F_cls = torch.vstack(F_cls)

        torch.save(F_fg, os.path.join(self.dir_fg, fname))
        assert F_fg.shape[-1] == 384 # TODO: don't hardcode
        
        torch.save(F_cls, os.path.join(self.dir_cls, fname))
        assert F_cls.shape[-1] == 384  # TODO: don't hardcode
       

        
    def __call__(self, docset):
          
        dataset = docset.as_dataset(self.d_patches, mode="page")
        dataset.transforms = transforms.ToTensor()
        dataloader = DataLoader(dataset, 500, num_workers=10, shuffle=False, drop_last=False)
        F_fg, F_all, F_cls = [], [], []

        curr_idx = 0
        print(len(dataset))
                   
        
        for i, (patches, page_ids) in tqdm(enumerate(dataloader), total=len(dataloader), desc=self.__class__.__name__):
                        
                patches = patches.cuda()
                with torch.no_grad():
                    x, f = self.model.forward_features_all(patches, return_foreground = True)
                    
                    f_cls = x[:, 0]
                    f_all = x[:, 1:]#.mean(dim=1)
                
                max_id = page_ids.max()
                
                curr_doc = page_ids == curr_idx
                F_fg.append(f[curr_doc].cpu().detach())
                F_all.append(f_all[curr_doc].cpu().detach())
                F_cls.append(f_cls[curr_doc].cpu().detach())
                
                while curr_idx < max_id:
                    
                    # there exists a higher id                    
                    if sum(page_ids == curr_idx) == 0:
                        curr_idx += 1
                        continue

                    # save current one - all items have been appended.
                    self.save_features(F_fg, F_all, F_cls, str(curr_idx))
                    list(docset.values())[curr_idx].img = None

                    F_fg, F_all, F_cls = [], [], []
                    
                    # increment to new index
                    curr_idx += 1
                    
                    curr_doc = page_ids == curr_idx
                    F_fg.append(f[curr_doc].cpu().detach())
                    F_all.append(f_all[curr_doc].cpu().detach())
                    F_cls.append(f_cls[curr_doc].cpu().detach())
                    
        # save last index
        self.save_features(F_fg, F_all, F_cls, str(curr_idx))
        return docset
        
                  
class SampleSubsetOfExtractedLocalFeatures(MappingModule):
    def __init__(self, dump, n_samples, n_workers=5):
        self.dir_fg = os.path.join(dump, "fg")
        self.n_samples = n_samples
        self.n_workers = n_workers
        
    def process(self, document):
        f = torch.load(os.path.join(self.dir_fg, document.filename.split(".")[0]))
        if self.n_samples == -1:
            document.descriptors = f
        else:
            indices = torch.randperm(len(f))[:min(len(f), self.n_samples)]
            document.descriptors = f[indices]   

                

class FeatureLoader(MappingModule):
    def __init__(self, file, ratio=1):
        self.encodings = joblib.load(file)
        self.ratio = ratio
    
    def process(self, document):
        
        descriptors = self.encodings[document.filename.split('.')[0]]
        if 0 < self.ratio < 1:
            idx = np.random.choice(len(descriptors), int(len(descriptors)*self.ratio), replace=False)
            descriptors = descriptors[idx]
        document.descriptors = descriptors
        
class FeatureLoaderFromDir(MappingModule):
    def __init__(self, dir, ratio=1, max_features=None):
        self.dir = dir
        self.max_features = max_features
        self.files = os.listdir(dir)
        self.ratio = ratio
        
    def process(self, document):
        try:
            desc = document.load_features_from_disk(self.dir)
        except Exception as e:
            #print(f"No features for: {document.filename}")
            desc = torch.zeros((1, 384))
        indices = torch.randperm(len(desc))[:int(self.ratio*len(desc))]
        if self.max_features is not None:
            indices = indices[:self.max_features]
        document.descriptors = desc[indices]
        
 

