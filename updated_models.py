import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_heads,
        num_layers,
        output_dim=7,
        max_cells=255,
        num_cls_latents=32,
    ):
        super().__init__()

        self.max_cells = max_cells
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cls_latents = num_cls_latents

        self.adjusted_output_dim = (hidden_dim // n_heads) * n_heads

        # ------------------------
        # Embedding
        # ------------------------
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                batch_first=False
            ),
            num_layers=num_layers
        )

        # ------------------------
        # CLS → multi-latent KV
        # ------------------------
        self.cls_to_kv = nn.Linear(
            hidden_dim,
            num_cls_latents * self.adjusted_output_dim
        )

        # Cell → Query
        self.cell_to_q = nn.Linear(
            hidden_dim,
            self.adjusted_output_dim
        )

        # Cell-dependent attention bias
        self.cell_attn_bias = nn.Linear(
            hidden_dim,
            num_cls_latents
        )

        # Cross-Attention
        self.reverse_cross_attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_output_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=False
        )

        # Norms
        self.pre_norm = nn.LayerNorm(self.adjusted_output_dim)
        self.post_norm = nn.LayerNorm(self.adjusted_output_dim)

        # Output heads
        self.final_output = nn.Linear(self.adjusted_output_dim, output_dim)
        self.cls_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        """
        x:    [B, max_cells, input_dim]
        mask: [B, max_cells] (1 = valid, 0 = padding)
        """

        B, N, _ = x.shape

        # ------------------------
        # Embed + CLS
        # ------------------------
        x = self.embedding(x)              # [B, N, H]
        x = x.permute(1, 0, 2)             # [N, B, H]

        cls = self.cls_token.repeat(1, B, 1)  # [1, B, H]
        x = torch.cat([cls, x], dim=0)        # [N+1, B, H]

        cls_mask = torch.ones(B, 1, device=mask.device)
        key_padding_mask = torch.cat([cls_mask, mask], dim=1) == 0

        # ------------------------
        # Transformer Encoder
        # ------------------------
        x = self.transformer(
            x,
            src_key_padding_mask=key_padding_mask
        )

        cls_feat = x[0]        # [B, H]
        cell_feat = x[1:]      # [N, B, H]

        # ------------------------
        # Build CLS latent KV
        # ------------------------
        cls_kv = self.cls_to_kv(cls_feat)  # [B, K*D]
        cls_kv = F.relu(self.cls_to_kv(cls_feat))
        cls_kv = cls_kv.view(
            B, self.num_cls_latents, self.adjusted_output_dim
        ).permute(1, 0, 2)  # [K, B, D]

        # ------------------------
        # Cell query
        # ------------------------
        cell_q = self.cell_to_q(cell_feat)  # [N, B, D]
        cell_q = self.pre_norm(cell_q)

        # ------------------------
        # Cell-dependent attention bias
        # ------------------------
        cell_bias = self.cell_attn_bias(cell_feat)  # [N, B, K]
        cell_bias = cell_bias.permute(1, 0, 2)      # [B, N, K]

        # Cross-Attention
        attn_out, attn_weights = self.reverse_cross_attention(
            query=cell_q,
            key=cls_kv,
            value=cls_kv,
            need_weights=True
        )
        # attn_out: [N, B, D]

        # ------------------------
        # Residual + norm
        # ------------------------
        attn_out = attn_out + cell_q
        attn_out = self.post_norm(attn_out)

        # ------------------------
        # Output
        # ------------------------
        attn_out = attn_out.permute(1, 0, 2)  # [B, N, D]
        x_out = self.final_output(attn_out)   # [B, N, output_dim]

        cls_out = self.cls_output(cls_feat)   # [B, output_dim]

        logits = torch.sigmoid(x_out)

        return cls_out, x_out, logits
