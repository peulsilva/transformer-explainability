import numpy as np
from src.image_utils import get_patch_coordinates

def perturb_text(inputs, importance_matrix, k, tokenizer, most_important : bool = True):
    """
    Removes the top k most important tokens based on importance w.r.t. CLS.
    """
    input_ids = inputs["input_ids"].squeeze().tolist()
    cls_importance = importance_matrix[0]  # CLS token importance
    token_importances = cls_importance[1:-1]  # Exclude CLS and SEP

    num_tokens_to_remove = int(len(token_importances) * k)
    # print(num_tokens_to_remove, token_importances.shape)
    if num_tokens_to_remove == 0:
        return inputs  # No perturbation if too few tokens

    # Get indices of most important tokens (excluding CLS)
    if most_important:
        important_indices = np.argsort(token_importances)[-num_tokens_to_remove:]
    else:
        important_indices = np.argsort(token_importances)[0:num_tokens_to_remove]

    # Create a new input ID list with removed tokens
    perturbed_input_ids = [
        token for i, token in enumerate(input_ids) if i-1 not in important_indices
    ]

    # Tokenize perturbed text again
    perturbed_inputs = tokenizer.decode(perturbed_input_ids)
    return tokenizer(perturbed_inputs, return_tensors="pt", truncation=True, padding=True)

def perturb_image(inputs, importance_matrix, k, most_important: bool = True):
    """
    Removes the top k most important image patches based on importance w.r.t. CLS.
    """

    patch_importances = importance_matrix[0, 1:]  # Importance of each patch w.r.t. CLS
    num_patches = patch_importances.shape[0]
    
    num_patches_to_remove = int(num_patches * k)
    if num_patches_to_remove == 0:
        return inputs  # No perturbation if too few patches

    # Get indices of most important (or least important) patches
    if most_important:
        important_indices = np.argsort(patch_importances)[-num_patches_to_remove:]
    else:
        important_indices = np.argsort(patch_importances)[:num_patches_to_remove]

    # Mask out the selected patches (set them to zero)
    perturbed_inputs = inputs.clone()
    patch_size = int(np.sqrt(inputs.shape[1]))  # Assuming square patches
    

    for idx in important_indices:
        # row, col = divmod(idx, patch_size)
        x_start, y_start, x_end, y_end = get_patch_coordinates(idx)
        perturbed_inputs[:, :, x_start:x_end, y_start:y_end] = 0  # Zero out the patch embedding

    return perturbed_inputs  # Return the modified image patches