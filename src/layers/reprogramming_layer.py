"""
Reprogramming layer.
"""

import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt

class ReprogrammingLayer(nn.Module):
    def __init__(self, llm_hidden_size, num_attention_heads, attention_dropout=0.1):
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = llm_hidden_size // num_attention_heads

        # Linear projections for multi-head attention
        self.query_projection = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.key_projection   = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.value_projection = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.out_projection   = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        target_embedding: [B, num_patches, llm_hidden_size]  (patch embeddings)
        source_embedding: [num_tokens, llm_hidden_size]      (mapped LLM embeddings)
        value_embedding:  [num_tokens, llm_hidden_size]      (usually same as source_embedding)
        """

        B, N, H = target_embedding.shape  # B=batch, N=num_patches, H=hidden_size
        S, _ = source_embedding.shape     # S=num_tokens

        # Multi-head projections
        Q = rearrange(self.query_projection(target_embedding), "B N (h d) -> B h N d", h=self.num_attention_heads)
        K = rearrange(self.key_projection(source_embedding), "S (h d) -> h S d", h=self.num_attention_heads)
        V = rearrange(self.value_projection(value_embedding), "S (h d) -> h S d", h=self.num_attention_heads)

        # Scaled dot-product attention
        scale = 1.0 / sqrt(self.head_dim)
        attn_scores = torch.einsum("B h N d, h S d -> B h N S", Q, K) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum over values
        out = torch.einsum("B h N S, h S d -> B h N d", attn_probs, V)

        # Combine heads
        out = rearrange(out, "B h N d -> B N (h d)")

        # Output projection
        out = self.out_projection(out)

        return out  # [B, num_patches, llm_hidden_size]

