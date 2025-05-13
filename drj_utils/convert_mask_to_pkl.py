import cv2
import numpy as np
from utils.file_utils import save_pkl
import openslide

def create_pickle_from_mask(mask_path, wsi_path, seg_level, output_pkl_path):
    # Load WSI to get downsample factor
    wsi = openslide.OpenSlide(wsi_path)
    downsample = wsi.level_downsamples[seg_level]
    
    # Load and process mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Binary mask (255 for tissue, 0 for background)
    mask = (mask > 0).astype(np.uint8)  # Ensure binary
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    
    # Filter contours (similar to segmentTissue)
    foreground_contours = []
    hole_contours = []
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)  # Foreground contours
    for cont_idx in hierarchy_1:
        cont = contours[cont_idx]
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        area = cv2.contourArea(cont) - sum(cv2.contourArea(contours[h]) for h in holes)
        if area > 100:  # Adjust threshold as needed
            foreground_contours.append(cont)
            hole_contours.append([contours[h] for h in holes if cv2.contourArea(contours[h]) > 16])
    
    # Scale contours to level 0
    scale = downsample[0]  # Assume isotropic downsample
    foreground_contours = [cont * scale for cont in foreground_contours]
    hole_contours = [[hole * scale for hole in holes] for holes in hole_contours]
    
    # Convert to expected format ([N, 1, 2] arrays)
    foreground_contours = [cont.astype(np.int32).reshape(-1, 1, 2) for cont in foreground_contours]
    hole_contours = [[hole.astype(np.int32).reshape(-1, 1, 2) for hole in holes] for holes in hole_contours]
    
    # Save to pickle
    asset_dict = {'tissue': foreground_contours, 'holes': hole_contours}
    save_pkl(output_pkl_path, asset_dict)

# Example usage
create_pickle_from_mask(
    mask_path='slide_001_mask.png',
    wsi_path='slide_001.svs',
    seg_level=6,
    output_pkl_path='slide_001_mask.pkl'
)