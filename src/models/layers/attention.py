# Removed unused typing import

import torch
from torch import nn


class Attention(nn.Module):
    """Multi-head attention mechanism.

    Attributes:
        heads: Number of attention heads.
        embed_dim: Total embedding dimension (head_dim * num_heads).
        w_q: Query projection layer.
        w_k: Key projection layer.
        w_v: Value projection layer.
        scele: Scaling factor for attention scores.
        dropout: Dropout layer for attention weights.
        w_out: Output projection layer with dropout.
    """

    def __init__(
        self,
        head_dim: int,
        q_dim: int,
        k_dim: int | None = None,
        v_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize attention module.

        Args:
            head_dim: Dimension of each attention head.
            q_dim: Dimension of query input.
            k_dim: Dimension of key input. If None, uses q_dim.
            v_dim: Dimension of value input. If None, uses q_dim.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = q_dim
        self.heads = num_heads
        self.embed_dim = head_dim * num_heads
        self.w_q = nn.Linear(q_dim, self.embed_dim, bias=False)
        self.w_k = nn.Linear(k_dim, self.embed_dim, bias=False)
        self.w_v = nn.Linear(v_dim, self.embed_dim, bias=False)
        self.scele = head_dim**0.5
        self.dropout = nn.Dropout(dropout)
        self.w_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(dropout),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)

        if self.w_out[0].bias is not None:
            nn.init.constant_(self.w_out[0].bias, 0.0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attn_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of attention mechanism.

        Args:
            q: Query tensor of shape (batch, seq_len, q_dim).
            k: Key tensor of shape (batch, seq_len, k_dim).
            v: Value tensor of shape (batch, seq_len, v_dim). If None, uses k.
            attn_mask: Attention mask tensor. If provided, masked positions will be
                filled with -inf.
            attn_weights: Additional attention weights to multiply with computed scores.

        Returns:
            Tuple of (output, attention_weights) where:
                - output: Attended output tensor of shape (batch, seq_len, embed_dim)
                - attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)
        """
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v) if v is not None else k

        # (batch, len, (head, dim)) -> (batch, head, len, dim)
        q = q.view(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scele
        if attn_weights is not None:
            scores = scores * attn_weights
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.embed_dim)
        return self.w_out(out), attn


if __name__ == "__main__":
    # Test the Attention module
    batch_size = 2
    seq_len = 128
    q_dim = 256

    q = torch.randn(batch_size, seq_len, q_dim)
    k = torch.randn(batch_size, seq_len, q_dim)
    v = torch.randn(batch_size, seq_len, q_dim)

    head_dim = 32
    num_heads = 8
    attn = Attention(head_dim, q_dim, num_heads=num_heads)
    out, attn_weights = attn(q, k, v)
    print(out.shape, attn_weights.shape)

    multihead_attn = nn.MultiheadAttention(head_dim * num_heads, num_heads, batch_first=True)
    out, attn_weights = multihead_attn(q, k, v, average_attn_weights=False)
    print(out.shape, attn_weights.shape)
