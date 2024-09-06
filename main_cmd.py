import argparse
from pathlib import Path
from pipeline import run_experiment
import datetime
import petname

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_args_parser():
    parser = argparse.ArgumentParser('AttMask', add_help=False)
    
    parser.add_argument('--ssl_method', default="dino", type=str, 
                        choices=['dino', 'attmask'],
                        help="Which SSL method to run")
    
    parser.add_argument('--incremental_pca_batchsize', default=-1, type=int, help="Used during Retrieval")
    parser.add_argument('--stride_factor_eval', default=1, type=float, help="Used during Retrieval")
    parser.add_argument('--force_keypoints', default=False, type=bool_flag, help="Force new keypoints to be created. Set this to true if something changed about keypoint creation and run directory already exists.")
    parser.add_argument('--aggregate_cls', type=bool_flag, default=True, help="what to aggregate?")
    parser.add_argument('--aggregate_fg', type=bool_flag, default=True, help="what to aggregate?")
    parser.add_argument('--aggregate_vlad', type=bool_flag, default=True, help="what to aggregate?")
    parser.add_argument('--aggregate_bvlad', type=bool_flag, default=False, help="what to aggregate?")
    parser.add_argument('--aggregate_sum', type=bool_flag, default=True, help="what to aggregate?")

    parser.add_argument('--aggregate_vlad_centroids', default=100, type=int, help="How many centroids to use for VLAD?")
    parser.add_argument('--extract_threshold_foreground', default=1, type=int, help="How many foreground pixels to consider a patch as FG patch?")
    parser.add_argument('--extract_train', type=bool_flag, default=True, help="what to extract?")
    parser.add_argument('--extract_test', type=bool_flag, default=True, help="what to extract?")
    
    # params for window extraction
    parser.add_argument('--dataset_window_size', default=256, type=int, help="Window size for dataset patches.")
    parser.add_argument('--model_image_size', default=224, type=int, help="Image size for the model.")
    
    # what folder structure
    parser.add_argument('--train_images', type=str, default="/data/traven/iam/binary", help="Path to training images.")
    parser.add_argument('--test_images', type=str, default="/data/traven/iam/binary", help="Path to test images.")
    parser.add_argument('--dump_run', type=str, default="/data/traven/runs/iam/004", help="Path to dump training data.")
    parser.add_argument('--checkpoint', type=str, default="/data/amatei/experiments/004.pth", help="checkpoint to use for eval")

    # eval params
    parser.add_argument('--eval_pca_dims', nargs='+', type=int, default=[128, 256, 384, 512, 786, 1024], help="List of dimensionalities")
    parser.add_argument('--label_authorid_stopchar', type=str, default="-", help="checkpoint to use for eval")
    parser.add_argument('--model_checkpoint_loadmode', type=str, default="dinov2", help="Path to training images.")
    parser.add_argument('--force_single_shot', type=bool_flag, default=False, help="only use 1 patch per document?")

    # what parts to run
    parser.add_argument('--train', type=bool_flag, default=False, help="Whether to train the model.")
    parser.add_argument('--extract', type=bool_flag, default=True, help="Whether to run inference.")
    parser.add_argument('--aggregate', type=bool_flag, default=True, help="Whether to aggregate results.")
    parser.add_argument('--retrieval', type=bool_flag, default=True, help="Whether to run prediction.")

    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('WR', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print(args)
    run_experiment(args)