import torch
import numpy
from module.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

def segment_slide(patch_dir, save_dir, segmentor_weight):
    """Segment slide using nnUNet

    Args:
        patch_dir (str): Path to slide
        save_dir (str): Path to save segmented slide
    """
    print("Segmenting slide...")
    predictor = nnUNetPredictor(verbose=True)

    predictor.initialize_from_trained_model_folder(
        segmentor_weight,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )
    predictor.predict_from_files(
        patch_dir,
        save_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    print("Done!")