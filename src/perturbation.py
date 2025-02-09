import numpy as np

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
