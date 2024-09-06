from concurrent.futures import ThreadPoolExecutor
import os
from typing import List

import joblib
from tqdm import tqdm
from misc.documentSet import DocumentSet
from os.path import join


class MappingModule():
    def __init__(self):
        pass

    def process(self, document):
        raise NotImplementedError


    def __call__(self, docsets):
        # class_name = self.__class__.__name__
        # if SHOW_TQDM:
        #     print(class_name)   
        n_workers = self.n_workers if hasattr(self, 'n_workers') and self.n_workers is not None else 16
        self.map(self.process, docsets, n_workers)
        return docsets
    
    def map_single_docset(self, func, docset: DocumentSet, n_workers):
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for _ in tqdm(executor.map(func, docset.values()), total=len(docset), desc=self.__class__.__name__):
                pass

    def map_multiple_docsets(self, func, docsets_list: List[DocumentSet], n_workers):
        for docset in docsets_list:
            self.map_single_docset(func, docset, n_workers)
    
    def map(self, func, docset, n_workers):
        if isinstance(docset, list):
            return self.map_multiple_docsets(func, docset, n_workers)
        if isinstance(docset, tuple):
            return self.map_multiple_docsets(func, docset, n_workers)
        if isinstance(docset, DocumentSet):
            return self.map_single_docset(func, docset, n_workers)
        

    
class TrainingModule(MappingModule):
    def __init__(self):
        pass

    def train(self, docset):
        self._train_impl(docset)
        self.trained = True
        return self

    def _train_impl(self, train_docset):
        raise NotImplementedError
    
    def __call__(self, docsets):
        if not self.trained:
            raise RuntimeError("Ensure to train the module!")
        else:
            super().__call__(docsets)

class PipelineArgs():
    def __init__(self, dir_train_imgs, dir_test_imgs, dump_train, dump_test, dump) -> None:
        self.dir_train_imgs = dir_train_imgs
        self.dir_test_imgs = dir_test_imgs
        self.dump_train = dump_train
        self.dump_test = dump_test
        self.dump = dump

        os.makedirs(self.dump_train, exist_ok=True)
        os.makedirs(self.dump_test, exist_ok=True)
        os.makedirs(self.dump, exist_ok=True)

class Pipeline():
    def __init__(self, pipeline_args):
        self.pargs = pipeline_args

    def build_docsets(self, distinct=True):
        train_docset = DocumentSet(self.pargs.dir_train_imgs)
        if distinct:
            test_docset = DocumentSet(self.pargs.dir_test_imgs)
        else:
            test_docset = train_docset    
            
        print(len(train_docset), len(test_docset))
        return train_docset, test_docset
        
    
    def save_docsets(self, name, docsets):
        os.makedirs(os.path.dirname(join(self.pargs.dump, name)), exist_ok=True)
        joblib.dump(docsets, join(self.pargs.dump, name))
        
    
    def load_docsets(self, name):
        return joblib.load(join(self.pargs.dump, name))
    
    
class ModuleWrapper():
    def __init__(self, modules):
        self.modules = modules

    def __call__(self, train_docset, test_docset):
        for m in self.modules:
            if isinstance(m, TrainingModule):
                m.train(train_docset)
            
            if test_docset is None:
                m(train_docset)
            else:
                m((train_docset, test_docset))
