import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, head_dim, q_dim, k_dim=None, v_dim=None, num_heads=8, dropout=0.0) -> None:
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

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)

        if self.w_out[0].bias is not None:
            nn.init.constant_(self.w_out[0].bias, 0.0)

    def forward(self, q, k, v=None, attn_mask=None, attn_weights=None):
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
