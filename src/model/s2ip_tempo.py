"""
S2IP + Tempo
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch

from torch import nn
from omegaconf import DictConfig
from hydra import initialize, compose
import torch
from torch import nn    
from omegaconf import DictConfig

from src.layers.adaptive_decomposition import LearnableClassicalDecomposition
from src.layers.data_embedding import PatchBasedDataEmbedding
from src.layers.semantic_prompting import SemanticSpaceInformedPrompting
from src.layers.source_embedding import SourceEmbedding
from src.layers.classification_head import ClassificationHeadSI2Tempo
from src.utils.load_llm import LLMLoader


class S2IPTempo(nn.Module):
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2

        self.decomposition = LearnableClassicalDecomposition(
            sequence_length=self.cfg.training.sequence_length,
            hidden_dim=self.cfg.model.ff_hidden_dim,
            dropout=self.cfg.model.ff_dropout,
            moving_avg_kernel_size=self.cfg.model.moving_avg_kernel_size,
            moving_avg_stride=self.cfg.model.moving_avg_stride
        )
        
        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)
    
        self.patch_embedding = PatchBasedDataEmbedding(
            use_linear=True,
            embed_type=None,
            freq=None,
            use_token=False,
            use_positional=False,
            use_temporal=False,
            llm_hidden_size= self.cfg.llm.hidden_size,
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
            dropout=self.cfg.model.patch_embedding_dropout,
            num_channels=self.cfg.training.num_channels
        )

        self.word_embedding = self.llm.get_input_embeddings().weight
        self.source_embedding = SourceEmbedding(
            vocab_size=self.cfg.llm.vocab_size,
            small_vocab_size=self.cfg.model.small_vocab_size,
            word_embedding=self.word_embedding
        )

        self.semantic_prompting = SemanticSpaceInformedPrompting(
            prompt_length=self.cfg.model.prompt_length,        
            top_k=self.cfg.model.top_k,               
            embedding_key=self.cfg.model.embedding_key,
            source_embedding=self.source_embedding                
        )

        #self.classification_head = ClassificationHeadLetsC(
            #input_dim=self.cfg.llm.hidden_size ,
           # num_classes=self.cfg.training.num_classes,
            #num_cnn_blocks=self.cfg.model.num_cnn_blocks,
           # cnn_channels=self.cfg.model.cnn_channels,
            #kernel_size=self.cfg.model.kernel_size,
           # mlp_hidden=self.cfg.model.mlp_hidden,
            #dropout=self.cfg.model.dropout,
            #pooling_type=self.cfg.model.pooling_type,
            #use_batch_norm=self.cfg.model.use_batch_norm,
        #)

        self.classification_head = ClassificationHeadSI2Tempo(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
            top_k=self.cfg.model.top_k 
        )
        
    def forward(self, x):
        if self.cfg.model.task_name == "classification":
            logits = self.classify(x) 
            return logits
     
    def classify(self, x):

        trend, season, residual = self.decomposition(x)
        trend_patched = self.patch_embedding(trend) 
        season_patched = self.patch_embedding(season) 
        residual_patched = self.patch_embedding(residual) 
        x_decomp_patched = torch.cat((trend_patched, season_patched, residual_patched), dim=1)
        
        semantic_prompt = self.semantic_prompting(x_decomp_patched)
        llm_output = self.llm(inputs_embeds=semantic_prompt).hidden_states[-1]
        logits = self.classification_head(llm_output)
                                          
        return logits                                  


