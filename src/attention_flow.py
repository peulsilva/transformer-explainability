import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from torchvision import transforms


class AttentionFlow:
    def __init__(self, model, is_vit : bool = False ,head_fusion="mean", discard_ratio=0.9, output_flow=2.0):
        """
        Initialize the Attention Flow Analyzer.
        
        :param head_fusion: Type of head fusion ('mean', 'max', 'min')
        :param discard_ratio: Percentage of lowest attention weights to discard
        :param output_flow: Output flow for max-flow computation
        """
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.output_flow = output_flow
        self.model = model
        self.is_vit = is_vit

    def compute_attention_matrices(self, attention_layers):
        """
        Generate attention matrices (A matrices) from attention layers.

        :param attention_layers: List of attention weight tensors from the model.
        :return: List of adjusted attention matrices.
        """
        a_matrices = []
        
        for attention in attention_layers:
            # Fusion across attention heads
            if self.head_fusion == "mean":
                attention_fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attention_fused = attention.max(dim=1)[0]
            elif self.head_fusion == "min":
                attention_fused = attention.min(dim=1)[0]
            else:
                raise ValueError("Unsupported attention head fusion type")

            # Flatten and discard lowest attention values
            flat_attention = attention_fused.view(attention_fused.size(0), -1)
            _, indices = flat_attention.topk(int(flat_attention.size(-1) * self.discard_ratio), largest=False)
            indices = indices[indices != 0]  # Keep CLS token attention
            flat_attention[0, indices] = 0

            # Normalize with identity matrix (residual connection)
            identity = torch.eye(attention_fused.size(-1))
            a_matrix = (attention_fused + identity) / 2
            a_matrix = a_matrix / a_matrix.sum(dim=-1, keepdim=True)

            a_matrices.append(a_matrix)
        
        return a_matrices

    def build_attention_graph(self, a_matrices, source_token):
        """
        Build a directed graph from attention matrices.

        :param a_matrices: Processed attention matrices.
        :param source_token: Token index as the source node.
        :return: NetworkX graph for flow computation.
        """
        num_layers = len(a_matrices)
        num_tokens = a_matrices[0].size(-1)
        if self.is_vit:
            num_tokens = 197
        graph = nx.DiGraph()
        source = "source"
        sink = "sink"

        # Add edges from source to first layer nodes
        for i in range(num_tokens):
            graph.add_edge(source, f"0-{i}", capacity=self.output_flow)

        # Add attention-weighted edges between layers
        for layer in range(num_layers):
            A = a_matrices[layer].squeeze().numpy()
            for i in range(num_tokens):
                for j in range(num_tokens):
                    graph.add_edge(f"{layer}-{i}", f"{layer+1}-{j}", capacity=A[i, j])

        # Add edges from last layer to sink
        for i in range(num_tokens):
            graph.add_edge(f"{num_layers}-{i}", sink, capacity=1.0)

        return graph

    def compute_max_flow(self, graph):
        """
        Compute the maximum attention flow using the NetworkX max-flow algorithm.

        :param graph: Directed NetworkX graph representing the attention flow.
        :return: Maximum flow values.
        """
        source, sink = "source", "sink"
        flow_value, flow_dict = nx.maximum_flow(graph, source, sink)

        # Extract flow values from last layer to input tokens
        num_tokens = len([node for node in graph.nodes if node.startswith("0-")])
        attention_flow = np.array([flow_dict[f"0-{i}"][f"1-{j}"] for i in range(num_tokens) for j in range(num_tokens)])
        attention_flow = attention_flow.reshape(num_tokens, num_tokens).sum(axis=0)  # Sum over paths

        return attention_flow

    def compute_attention_flow(self, attention_layers):
        """
        Compute attention flow for all tokens.

        :param attention_layers: Attention weights from a Transformer model.
        :return: Normalized attention flow mask.
        """
        a_matrices = self.compute_attention_matrices(attention_layers)
        num_tokens = a_matrices[0].size(-1)

        mask = torch.zeros(num_tokens)
        for token_idx in tqdm(range(num_tokens), desc="Computing Attention Flow"):
            graph = self.build_attention_graph(a_matrices, token_idx)
            mask[token_idx] = self.compute_max_flow(graph)[token_idx]

        # Remove CLS token and reshape into a 2D mask
        if self.is_vit:
            mask = mask[1:]  # Exclude CLS token
            mask = mask.reshape(int(mask.size(-1) ** 0.5), -1).numpy()
            mask /= np.max(mask)  # Normalize

        return mask
    
    def __call__(self, *args, **kwds):
        with torch.no_grad():
            out = self.model(**kwds, output_attentions = True)

        return self.compute_attention_flow(out.attentions)


    def visualize_attention_flow(self, attention_flow, tokens):
        """
        Visualize attention flow values.

        :param attention_flow: Attention flow values.
        :param tokens: Tokenized input text.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(tokens)), attention_flow, tick_label=tokens)
        plt.xlabel("Input Tokens")
        plt.ylabel("Attention Flow Importance")
        plt.title("Attention Flow Importance per Token")
        plt.xticks(rotation=90)
        plt.show()