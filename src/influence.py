import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

class Influence:
    def __init__(self, model, discard_ratio = 0.9, is_vit : bool = False, return_mask : bool = True, head_fusion = 'mean'):
        self.model = model
        self.model.eval()
        self.device = self.model.device
        self.discard_ratio = discard_ratio
        self.is_vit = is_vit
        self.return_mask = return_mask
        self.head_fusion = head_fusion

    def convert_to_cpu(self, list):
        new_list = []
        for el in list:
            new_list.append(el.to('cpu'))

        return new_list
    
    def apply_discard_ratio(self, attention_matrix):
        """Apply discard ratio to drop low attention values."""
        
        if self.discard_ratio > 0:
            flat = attention_matrix.view(-1)
            num_to_discard = int(flat.size(0) * self.discard_ratio)
            
            # Get indices of the lowest values
            _, indices = flat.topk(num_to_discard, largest=False)
            flat[indices] = 1e-7  # Set them to zero
            
            attention_matrix = flat.view(attention_matrix.shape)  # Reshape back
        return attention_matrix / attention_matrix.sum(dim=-1, keepdims=True)

    def __call__(self, **kwargs):
        """Compute Influence matrix showing each token's impact on every other token."""
        
        # Tokenize input text
        with torch.no_grad():
            outputs = self.model(**kwargs, output_attentions=True, output_hidden_states=True)

        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            input_ids = kwargs.get('pixel_values')
        
        attentions = self.convert_to_cpu(outputs.attentions) # List of attention tensors from each layer
        hidden_states = self.convert_to_cpu(outputs.hidden_states)  # List of hidden state tensors from each layer

        
        num_layers = len(attentions)
        seq_len = input_ids.shape[1]
        if self.is_vit:
            seq_len = 197

        # Compute norms of hidden states at each layer
        norm_hidden_states = torch.stack([torch.norm(hs, dim=-1) for hs in hidden_states])  # Shape: (num_layers, batch, seq_len)

        # Initialize Influence matrix (size: seq_len × seq_len)
        influence_matrix = torch.zeros(seq_len, seq_len)

        # Influence propagation through layers
        influence_scores = torch.eye(seq_len)  # Identity matrix: each token initially influences itself
        
        for layer in range(num_layers):  
            if self.head_fusion == 'mean':
                attention_matrix = attentions[layer].max(dim=1)[0]
            if self.head_fusion == 'max':
                attention_matrix = attentions[layer].max(dim=1)[0]  # Average over heads, shape: (seq_len, seq_len)
            attention_matrix = self.apply_discard_ratio(attention_matrix,)
            r_k = norm_hidden_states[layer][0] / ((attention_matrix @ hidden_states[layer]).norm(dim=-1) + 1e-6)

            # Apply Influence propagation formula
            influence_attention = torch.matmul(attention_matrix, influence_scores)
            
            influence_scores = (
                influence_scores * r_k[:, None] +
                influence_attention
            ) / (1 + r_k[:, None])  # Equation from the image

        # Normalize final Influence matrix
        influence_matrix = influence_scores.cpu()
        influence_matrix /= (influence_matrix.sum(axis=-1, keepdims=True) + 1e-6)

        if self.is_vit and self.return_mask:
            mask = influence_matrix[0, 0, 1:].numpy()
            width = 14
            mask = mask.reshape(width, width)
            mask = mask / np.max(mask)
            return mask

        return influence_matrix