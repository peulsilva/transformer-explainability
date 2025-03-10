import torch
import numpy
import sys
import numpy as np
from collections import OrderedDict

# def rollout(attentions, discard_ratio,is_vit : bool = False):
#     result = torch.eye(attentions[0].shape[-1]).unsqueeze(dim = 0)
#     with torch.no_grad():
#         for attention_heads_fused in attentions:
            
#             # Drop the lowest attentions, but
#             # don't drop the class token
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

            
#             I = torch.eye(attention_heads_fused.size(-1)).unsqueeze(dim = 0)
#             # print(I.shape, attention_heads_fused.shape, I.device, attention_heads_fused.device)
#             a = (attention_heads_fused + 1.0*I)/2

#             a = a / a.sum(dim=-1, keepdims = True)
#             result = torch.matmul(a, result)


    
#     # Look at the total attention between the class token,
#     # and the image patches
#     if is_vit:
#         mask = result[0, 0 , 1 :]
#         # # In case of 224x224 image, this brings us from 196 to 14
#         width = int(mask.size(-1)**0.5)
#         mask = mask.reshape(width, width).numpy()
#         mask = mask / np.max(mask)

#         return mask
#     return result

class RawAttention:
    def __init__(
        self, 
        model, 
        is_vit : bool = False,
        attention_layer_name='attn_drop', 
        head_fusion="mean",
        discard_ratio=0.9
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        

        self.attentions = []
        self.is_vit = is_vit

    def get_attention(self, module, input, output):
        if self.head_fusion == "mean":
            attention_heads_fused = output[1].mean(dim=1)
        elif self.head_fusion == "max":
            attention_heads_fused = output[1].max(dim=1)[0]
        elif self.head_fusion == "min":
            attention_heads_fused = output[1].min(dim=1)[0]
        else:
            raise "Attention head fusion type Not supported"
        
        self.attentions.append(attention_heads_fused.cpu())

    def __call__(self, layer_idx = 0 ,input_tensor = None, **kwargs):

        for name, module in self.model.named_modules():
            if name.endswith(self.attention_layer_name):
                module.register_forward_hook(self.get_attention)
        self.attentions = []

        with torch.no_grad():
            output = self.model(**kwargs, output_attentions = True)

        attn_matrix = self.attentions[layer_idx]

        self.remove_hooks()

        return output, attn_matrix
    
    def remove_hooks(self):
        """
        Removes any forward hooks attached to the self-attention modules of the base model.
        """
        for name, module in self.model.named_modules():
            if name.endswith(self.attention_layer_name):
                module._forward_hooks = OrderedDict()
