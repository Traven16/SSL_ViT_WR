""" This module encapsulate a renewed version of the Pipeline """


from dataclasses import dataclass
import logging
import os
from typing import Any, List, Optional, Tuple

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.preprocessing import normalize

import torch
from torchvision import transforms
from components import *; from keypoints import *; from aggregators import *
from features import *; from retrieval import *; from helpers import *
from constants import *

from misc.documentSet import DocumentSet
from os.path import join
import warnings

import models_vit

class DC_Pipeline(Pipeline):
    def __init__(
            self, 
            pargs,               
            extract_features = True,
            aggregate_features = True,
            run_prediction = True,
            patches_train_args = None,
            patches_test_args = None, 
            model_args = None,
            extract_features_args = None,
            aggregate_features_args = None,
            prediction_args = None,
                 ):
        super().__init__(pargs)
        
        # ==== which stages to run? =====
        
        self.extract_features = extract_features
        self.aggregate_features = aggregate_features
        self.run_prediction = run_prediction
        
        
        # ==== settings for each stage? =====
        self.patches_train_args = patches_train_args
        self.patches_test_args = patches_test_args
        self.model_args = model_args
        self.extract_features_args = extract_features_args
        self.aggregate_features_args = aggregate_features_args
        self.predicition_args = prediction_args


    def run_patchextraction(self, train_docset, test_docset):
        if self.extract_train_patches:
            self.generate_keypoints(train_docset, self.patches_train_args, train=True)
        
        if self.extract_test_patches:
            self.generate_keypoints(test_docset, self.patches_test_args, train=False)        
        
        
    
    def generate_keypoints(self, docset: DocumentSet, args: PatchSettings, train: bool) -> DocumentSet:
        """
        Generates keypoints for the given document set based on the specified patch settings.
        Args:
            docset (DocumentSet): The document set for which keypoints need to be generated.
            args (PatchSettings): The patch settings specifying the keypoint mode and other parameters.
            train (bool): A flag indicating whether the keypoints are being generated for training or not.
        Returns:
            DocumentSet: The document set with keypoints generated.
        Raises:
            None
        Examples:
            # Create a document set
            docset = DocumentSet()
            # Create patch settings
            patch_settings = PatchSettings()
            # Generate keypoints for the document set
            keypoints = generate_keypoints(docset, patch_settings, train=True)
        """
                
        store_dir = None

        if args.keypoint_mode == KEYPOINT_MODE_SIFT:

            extractor = PatchExtractionSIFTKeypoints(
                store_dir, 
                args.save_mode,
                d_patches=args.d_patches,
                n_patches_per_page=args.n_per_page,
                n_min_distance=args.d_min_distance,
                )
            
        elif args.keypoint_mode == KEYPOINT_MODE_SIFT_CLUSTER:
            
            extractor = PatchExtractionSIFTCluster(
                store_dir, 
                args.save_mode, 
                d_patches=args.d_patches,
                n_patches_per_page=args.n_per_page,
                n_min_distance=args.d_min_distance,
                n_classes = args.n_classes,
                n_batchsize = args.n_batchsize,
                )
                        
        elif args.keypoint_mode == KEYPOINT_MODE_GRID:
            
            extractor = PatchExtractionGrid(
                dir_patches=store_dir,
                d_patches=args.d_patches,
                d_stride=args.d_stride, 
                mode=args.save_mode
                )
            
        elif args.keypoint_mode == KEYPOINT_MODE_SINGLESHOT:
            
            extractor = PatchExtractionSingleShot(
                dir_patches=store_dir,
                d_patches=args.d_patches,
                d_stride=args.d_stride, 
                mode=args.save_mode
                )
        
        extractor(docset, None)  
        return docset
    
    
    def _run_aggregation_helper(self, encoding, savename, train_docset, test_docset, dump_train=None, dump_test=None, folder="cls"):
        """
        Helper method for running aggregation.
        Args:
            encoding: The encoding module used for aggregation.
            savename: The name of the aggregation.
            train_docset: The training document set.
            test_docset: The test document set.
            dump_train: The directory path for dumping training data.
            dump_test: The directory path for dumping test data.
            folder: The folder name for storing encoded data.
        Returns:
            None
        """
        print("Aggregating", savename)
        if isinstance(encoding, TrainingModule):
            # We load a subset of the descriptors of each training document. In the end, we remove them again
            # to keep file size small
            train_docset = FeatureLoaderFromDir(os.path.join(dump_train, folder), ratio=0.2)(train_docset)
            encoding.train(train_docset)
            train_docset = clean_descriptors(train_docset) 
        
        
        # apply encoding to train set + save
        encoding.from_dir = join(dump_train, folder)
        train_docset = encoding(train_docset)
        self.save_docsets(f"train/{savename}", train_docset)
        
        # apply to test set + save
        encoding.from_dir = join(dump_test, folder)
        encoding(test_docset)
        self.save_docsets(f"test/{savename}", test_docset)
       
    
    def load_model(self):
        model = models_vit.__dict__[self.model_args.arch](
            patch_size=self.model_args.patch_size,
            global_pool=False,
            num_classes=0,
            n_fg=self.model_args.extract_threshold_foreground,
        )  
        
        # Load model
        checkpoint = self.model_args.checkpoint
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if self.model_args.model_checkpoint_loadmode in ["dino", "swinv2"]:
        
            checkpoint_model = checkpoint["student"]
            # remove `module.` prefix
            checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}

        else:
            checkpoint_model = checkpoint["model"]
             
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        model.eval()
        return model
            
    
    
    def run_pipeline(self):
        docsets = self.build_docsets()
        train_docset = docsets[0]
        test_docset = docsets[1]
        
        # ================================================
        # ============= I : Feature Extraction ===========
        # ================================================
        print("i")
        if self.extract_features:
            # LOAD MODEL
            if self.model_args.extract_train or self.model_args.extract_test:
                model = self.load_model()
            
            # TRAINING SET
            if self.model_args.extract_train:
                train_docset = self.generate_keypoints(train_docset, self.patches_train_args, train=True)

                feature_extractor = FeatureExtraction(model, 
                    self.patches_test_args.d_patches, transforms.ToTensor(), 
                    self.pargs.dump_train, args=self.model_args)
                
                train_docset = feature_extractor(train_docset)
            
            # TEST SET
            if self.model_args.extract_test:
                test_docset = self.generate_keypoints(test_docset, self.patches_test_args, train=False)
                
                feature_extractor = FeatureExtraction(model, 
                    self.patches_test_args.d_patches, transforms.ToTensor(), 
                    self.pargs.dump_test, args=self.model_args)
                
                test_docset = feature_extractor(test_docset)  
        
        # ================================================
        # =========== II : Feature Aggregation ===========
        # ================================================
        print("ii")
        if self.aggregate_features:
            # Perform aggregation for all combinations of encoding and mode.
            for encoding in [encoding for encoding, cond in [
                                (SumPooling(), self.model_args.aggregate_sum),
                                (VLAD(n_clusters=self.model_args.aggregate_vlad_centroids, gmp=False), self.model_args.aggregate_vlad),
                                (bVLAD(5, 0.4, 250), self.model_args.aggregate_bvlad)] # FIXME: hardcoded
                            if cond]:
                
                for m in    [mode for mode, cond in [
                                ('cls', self.model_args.aggregate_cls), 
                                ('fg', self.model_args.aggregate_fg), 
                                ('all', self.model_args.aggregate_all)] 
                            if cond]:
                                       
                    self._run_aggregation_helper(encoding, f"{m}_{encoding.__class__.__name__}", train_docset, test_docset, self.pargs.dump_train, self.pargs.dump_test, m)
                    

        # ================================================
        # =========== III : Retrieval ====================
        # ================================================
        print("iii")
        if self.run_prediction:
            logging.info(self.model_args)
            for test_docset_path in os.listdir(join(self.pargs.dump, "test")):
                # logging
                logging.info(f"Evaluating: {test_docset_path}")
                print(test_docset_path)
                
                # load docsets
                test_docset = self.load_docsets(f"test/{test_docset_path}")
                train_docset = self.load_docsets(f"train/{test_docset_path}")
                
                # evaluate
                Retrieval(rerank=True).evaluate(train_docset, test_docset)
               
        
        # # Step 5: Visualize # probably doesn't work atm
        # if False: # TODO: don't hardcode
        #     print("Visualize")
        #     test_docset = self.load_docsets("eval/fg_SumPooling")
        #     os.makedirs(join(self.model_args.dump_run, "visual"), exist_ok=True)
        #     VisualizeRetrieval(rerank=False).visualize(test_docset, join(self.model_args.dump_run, "visual") )
         
        
def run_experiment(args):
    
    stride = int(args.model_image_size*args.stride_factor_eval)
    patches_train = PatchSettings(
        keypoint_mode=KEYPOINT_MODE_GRID if not args.force_single_shot else KEYPOINT_MODE_SINGLESHOT, 
        save_mode=PATCH_SAVEMODE_NONE, 
        d_patches=args.dataset_window_size, 
        d_stride=stride)
    patches_eval = PatchSettings(
        keypoint_mode=KEYPOINT_MODE_GRID if not args.force_single_shot else KEYPOINT_MODE_SINGLESHOT, 
        save_mode=PATCH_SAVEMODE_NONE,
        d_patches=args.model_image_size, 
        d_stride=stride)
    
    pargs = PipelineArgs(
        args.train_images,
        args.test_images,
        join(args.dump_run, "features_train"),
        join(args.dump_run, "features_test"),
        join(args.dump_run, "docsets"),)
    
    pipeline = DC_Pipeline(pargs, 
                           args.extract, 
                           args.aggregate,
                           args.retrieval, 
                           patches_train, 
                           patches_eval, 
                           args, )
    
    logging.basicConfig(filename=join(args.dump_run, "results.log"), level=logging.INFO, format='%(message)s')
    
    pipeline.run_pipeline()
    
    
    