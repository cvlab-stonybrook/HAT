
import torch
import torch.nn as nn
from torchvision import models
import math
from os.path import join, dirname
from typing import Optional
import numpy as np
from typing import Tuple


class PositionalEncoding2D(nn.Module):
    """
    Helper Module that adds positional encoding to the token
    embedding to introduce a notion of word order.
    """
    def __init__(self,
                 pa,
                 d_model: int,
                 dropout: float,
                 height: int = 20,
                 width: int = 32,
                 patch_num: list = None):
        super(PositionalEncoding2D, self).__init__()
        d_model = d_model // 2
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) /
                        d_model)
        self.pa = pa
        self.n_special_symbols = len(pa.special_symbols)
        self.d_model = d_model

        pos_h = torch.arange(0, height).reshape(height, 1)
        pos_h_embedding = torch.zeros((height, d_model))
        pos_h_embedding[:, 0::2] = torch.sin(pos_h * den)
        pos_h_embedding[:, 1::2] = torch.cos(pos_h * den)
        pos_h_embedding = pos_h_embedding

        pos_w = torch.arange(0, width).reshape(width, 1)
        pos_w_embedding = torch.zeros((width, d_model))
        pos_w_embedding[:, 0::2] = torch.sin(pos_w * den)
        pos_w_embedding[:, 1::2] = torch.cos(pos_w * den)
        pos_w_embedding = pos_w_embedding
        self.height = height
        self.width = width

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('pos_w_embedding', pos_w_embedding)
        self.register_buffer('pos_h_embedding', pos_h_embedding)

    def forward(self, tgt_seq, scale=1):
        # Remove special_symbols
        gaze_symbol_idx = torch.logical_and(tgt_seq != self.pa.pad_idx,
                                            tgt_seq != self.pa.eos_idx)
        pe = torch.zeros(*tgt_seq.shape, self.d_model * 2).to(tgt_seq.device)
        if gaze_symbol_idx.sum() == 0:
            return pe
        
        actions = tgt_seq[gaze_symbol_idx] - self.n_special_symbols
        y = actions // (self.width / scale) + scale // 2
        x = actions % (self.width / scale) + scale // 2
        pe_valid = self.forward_pos(x, y)
        pe[gaze_symbol_idx] = pe_valid
        return pe

    def forward_pos(self, x, y):
        assert x.max() < self.width and y.max() < self.height, "out of range"
        pe_x = self.pos_w_embedding[x.long()]
        pe_y = self.pos_h_embedding[y.long()]
        pe = torch.cat([pe_x, pe_y], dim=1)
        return pe

    def forward_featmaps(self, size, scale=1):
        h, w = size
        assert h == math.ceil(self.height / scale) and w == math.ceil(
            self.width / scale), "wrong input"
        smp_ind_x = torch.arange(scale // 2, self.width, scale)
        smp_ind_y = torch.arange(scale // 2, self.height, scale)
        pe_x = self.pos_w_embedding[smp_ind_x].transpose(0, 1)
        pe_y = self.pos_h_embedding[smp_ind_y].transpose(0, 1)
        pe_x = pe_x.unsqueeze(1).repeat(1, h, 1)
        pe_y = pe_y.unsqueeze(2).repeat(1, 1, w)
        pe = torch.cat([pe_x, pe_y], dim=0)
        return pe.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """helper Module that adds positional encoding to the token embedding to introduce a notion of word order."""
    def __init__(self, emb_size: int, maxlen: int = 100):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.pos_embedding[:token_embedding.size(0), :]

    def forward_pos(self, pos: torch.Tensor):
        return self.pos_embedding[pos].squeeze(1)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    See details in https://arxiv.org/abs/2006.10739.
    """

    def __init__(self, pa, 
                 d_model: int, 
                 dropout: float,
                 height: int = 20,
                 width: int = 32, 
                 scale: Optional[float] = None,) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.pa = pa
        self.n_special_symbols = len(pa.special_symbols)
        self.dropout = nn.Dropout(dropout)
        self.height = height
        self.width = width
        self.d_model = d_model // 2
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, self.d_model)),
        )

    def forward_pos(self, x: torch.Tensor, y: torch.Tensor, normalize=True) -> torch.Tensor:
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        if normalize:
            # Normalize to [0, 1]
            x, y = x / self.width, y / self.height
        coords = torch.stack([x, y], dim=-1)
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return pe

    def forward_featmaps(self, size: Tuple[int, int], scale: int = 1) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self.forward_pos(x_embed, y_embed, normalize=False)
        return pe.permute(2, 0, 1)  # C x H x W

    def forward(self, tgt_seq, scale=1)-> torch.Tensor:
        # Remove special_symbols
        gaze_symbol_idx = torch.logical_and(tgt_seq != self.pa.pad_idx,
                                            tgt_seq != self.pa.eos_idx)
        pe = torch.zeros(*tgt_seq.shape, self.d_model * 2).to(tgt_seq.device)
        if gaze_symbol_idx.sum() == 0:
            return pe
        
        actions = tgt_seq[gaze_symbol_idx] - self.n_special_symbols
        y = actions // (self.width / scale) + scale // 2
        x = actions % (self.width / scale) + scale // 2
        pe_valid = self.forward_pos(x, y)
        pe[gaze_symbol_idx] = pe_valid
        return pe
