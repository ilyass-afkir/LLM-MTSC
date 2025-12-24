"""
Heads
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ClassificationHeadOFA(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, num_classes)
       
    def forward(self, x):

        if self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = rearrange(x, "B N H -> B (N H)")

        x = self.layer_norm(x)
        #x = self.dropout(x)       
        logits = self.linear_projection(x)
        return logits
        
        
class ClassificationHeadLLMFew(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Sequential(
            nn.Linear(self.llm_hidden_size * self.num_patches, self.num_classes),
            nn.Dropout(self.dropout)
        )
       
    def forward(self, x, llm_output):

        if self.activation == 'gelu':
            x = F.gelu(llm_output + x)
        elif self.activation == 'relu':
            x = F.relu(llm_output + x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(llm_output + x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = rearrange(x,"B N H -> B (N H)")
        x = self.layer_norm(x)
        logits = self.linear_projection(x)

        return logits


class ClassificationHeadTimeLLM(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, self.num_classes)
    
    def forward(self, x):

        x = rearrange(x,"B N H -> B (N H)")

        x = self.layer_norm(x)
        self.dropout(x)
        
        if self.activation == 'gelu':
            x = nn.GELU(x)
        elif self.activation == 'relu':
            x = nn.ReLU(x)
        elif self.activation == 'leaky_relu':
            x = nn.LeakyReLU(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        logits = self.linear_projection(x)
        
        return logits


class ClassificationHeadSI2Tempo(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
        top_k: int
    ):
        super().__init__()
        
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        self.activation = act

        self.layer_norm = nn.LayerNorm(llm_hidden_size * (3 * num_patches + top_k))
        self.linear = nn.Linear(llm_hidden_size * (3 * num_patches + top_k), num_classes)
        
    def forward(self, x):
        x = rearrange(x,"B N H -> B (N H)")
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits

class ClassificationHeadDeepRange(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str
    ):
        super().__init__()

        hidden_in = llm_hidden_size * num_patches
        self.layer_norm = nn.LayerNorm(hidden_in)

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        self.activation = act

        self.linear = nn.Linear(hidden_in, num_classes)

    def forward(self, x):
        x = rearrange(x, "B N H -> B (N H)")
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


class ClassificationHeadLetsC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_cnn_blocks: int,
        cnn_channels: int,
        kernel_size: int,
        mlp_hidden: int,
        dropout: float,
        pooling_type: str,
        use_batch_norm: bool,
    ):
        super().__init__()
        
        self.pooling_type = pooling_type
        cnn_layers = []
        in_channels = input_dim
        
        for _ in range(num_cnn_blocks):
            # Conv block
            cnn_layers.append(nn.Conv1d(
                in_channels, 
                cnn_channels, 
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=not use_batch_norm
            ))
            
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(cnn_channels))
            
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            
            in_channels = cnn_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Global Pooling (instead of flatten!)
        if pooling_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling_type}")
        
        # MLP classifier - much smaller than with flatten!
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    
    def forward(self, x):

        x = rearrange(x, "B N H -> B H N")
        
        # CNN feature extraction
        x = self.cnn(x)  # (B, cnn_channels, L)
        
        # Global pooling (KEY: no flatten!)
        x = self.global_pool(x)  # (B, cnn_channels, 1)
        x = rearrange(x, "B C 1 -> B C" )

        # Classification
        logits = self.classifier(x)
        
        return logits
