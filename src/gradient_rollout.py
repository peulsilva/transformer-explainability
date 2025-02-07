import torch
import numpy as np

from collections import OrderedDict

def grad_rollout(attentions, gradients, discard_ratio, is_vit : bool = False):
    result = torch.eye(attentions[0].size(-1))\
        .unsqueeze(dim = 0)\
        .to(attentions[0].device)
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):   
            weights = grad.to(attention.device)

            attention_heads_fused = (attention*weights).mean(axis=1)
            # attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))\
                .unsqueeze(dim = 0)\
                .to(attention.device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1, keepdims = True)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    if is_vit:
        mask = result[0, 0 , 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask    
    
    return result

class AttentionGradRollout:
    def __init__(
        self, 
        model, 
        attention_layer_name='attn_drop',
        discard_ratio=0.9,
        is_vit : bool = False
    ):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        self.remove_hooks()
        for name, module in self.model.named_modules():
            if name.endswith(attention_layer_name):
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

        self.is_vit = is_vit



    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, **kwargs):

        self.attentions = []
        self.attention_gradients = []

        category_index = kwargs['labels']
        self.model.zero_grad()
        output = self.model(**kwargs)
        # loss = output.loss
        # loss.backward()
        category_mask = torch.zeros(output.logits.size(), device = output.logits.device)
        category_mask[:, category_index] = 1
        loss = (output.logits*category_mask).sum()
        loss.backward()

        for i in range(len(output.attentions)):
            self.attentions.append(output.attentions[i].to('cpu'))

        return output, grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio, self.is_vit)
    
    def remove_hooks(self):
        """
        Removes any forward hooks attached to the self-attention modules of the base model.
        """
        for name, module in self.model.named_modules():
            if name.endswith(self.attention_layer_name):
                module._forward_hooks = OrderedDict()
