"""The Transformer: a small Vision Transformer, trained from scratch on CIFAR-100.

    python vit.py      trains the model and writes results/vit.json
"""

import torch
import torch.nn as nn

import train


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class Block(nn.Module):
    """Encoder block: attention, then an MLP, each with a layer norm in front and a skip connection."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, num_classes=100, image_size=32, patch_size=4, dim=384, depth=6, heads=6, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        # A convolution with kernel size = stride = patch size cuts the image
        # into patches and applies the same linear layer to each of them.
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Attention does not know where a token came from, so every position
        # gets its own learned vector.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)       # (B, 64, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed          # (B, 65, dim)
        x = self.norm(self.blocks(x))
        return x[:, 0]                                           # the class token

    def forward(self, x):
        return self.fc(self.forward_features(x))


if __name__ == "__main__":
    args = train.setup()
    model = ViT(train.NUM_CLASSES).to(train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    train.run("vit", "ViT", model, optimizer, args, warmup_epochs=10, grad_clip=1.0)
