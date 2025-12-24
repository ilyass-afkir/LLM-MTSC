"""
Semantic Space Informed Prompting - Simple Version Following the Paper

Paper describes 3 simple steps:
1. Reduce vocabulary embeddings E to semantic anchors E' using f(E)
2. Match input time series P with anchors using cosine similarity γ(P, e')
3. Select top-K anchors and prepend them to input: Z = [e'_1; ...; e'_K; P]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSpaceInformedPrompting(nn.Module):
    def __init__(
        self,           
        prompt_length: int,         
        top_k: int,                
        embedding_key: str,    
        source_embedding: nn.Module                
    ):
        super().__init__()
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.source_embedding = source_embedding
        
    def forward(self, x):
        B, _, _ = x.shape
        
        # Pool input sequence to single vector
        if self.embedding_key == "mean":
            query = x.mean(dim=1)  # [B, embed_dim]
        elif self.embedding_key == "max":
            query = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown embedding_key: {self.embedding_key}")
        
        query_norm = F.normalize(query, p=2, dim=1)  # [B, embed_dim]

        # --- Use actual embeddings tensor ---
        anchors = self.source_embedding() # [num_anchors, embed_dim]
        anchors_norm = F.normalize(anchors, p=2, dim=1)  # [num_anchors, embed_dim]

        # Cosine similarity
        similarity = torch.matmul(query_norm, anchors_norm.T)  # [B, num_anchors]

        # Select top-K
        _, top_k_indices = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

        # Gather selected anchors
        selected_anchors = anchors[top_k_indices]  # [B, top_k, embed_dim]

        # Concatenate with input
        prompted_embedding = torch.cat([selected_anchors, x], dim=1)  # [B, top_k + seq_len, embed_dim]

        return prompted_embedding