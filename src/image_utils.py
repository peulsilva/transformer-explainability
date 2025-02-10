import numpy as np 
import cv2

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_patch_coordinates(index, patch_size=16, image_size=224):
    """
    Given an index from the CLS token attention vector (size 196), return
    the equivalent patch coordinates in the original image (224x224).
    
    Args:
        index (int): Index in the attention vector (0 to 195)
        patch_size (int): Size of each patch (default: 16)
        image_size (int): Size of the image (default: 224)
    
    Returns:
        (x_start, y_start, x_end, y_end): Coordinates of the corresponding patch in the image
    """
    # Ensure index is within bounds
    if not (0 <= index < (image_size // patch_size) ** 2):
        raise ValueError("Index out of bounds. Must be between 0 and 195.")
    
    # Compute the number of patches per row/column
    num_patches_per_row = image_size // patch_size  # 224 // 16 = 14
    
    # Compute row and column in the patch grid
    row = index // num_patches_per_row
    col = index % num_patches_per_row
    
    # Compute pixel coordinates in the original image
    x_start = col * patch_size
    y_start = row * patch_size
    x_end = x_start + patch_size
    y_end = y_start + patch_size
    
    return (x_start, y_start, x_end, y_end)
