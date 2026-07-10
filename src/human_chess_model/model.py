from __future__ import annotations

import torch
from torch import nn

from human_chess_model.constants import BOARD_SHAPE, MOVE_VOCAB_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ChessPolicyNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 6) -> None:
        super().__init__()
        in_channels = BOARD_SHAPE[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.policy(self.trunk(self.stem(x)))


def build_model(channels: int = 128, blocks: int = 6) -> ChessPolicyNet:
    return ChessPolicyNet(channels=channels, blocks=blocks)


class SquareTransformerPolicyNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        in_channels = BOARD_SHAPE[0]
        self.square_proj = nn.Linear(in_channels, embed_dim)
        self.position = nn.Parameter(torch.zeros(1, 64, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * embed_dim, MOVE_VOCAB_SIZE),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 3, 1).reshape(x.shape[0], 64, x.shape[1])
        x = self.square_proj(x) + self.position
        x = self.encoder(x)
        x = self.norm(x)
        return self.policy(x)


def build_policy_model(
    arch: str = "cnn",
    channels: int = 128,
    blocks: int = 6,
    embed_dim: int = 192,
    layers: int = 6,
    heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.05,
) -> nn.Module:
    if arch == "cnn":
        return ChessPolicyNet(channels=channels, blocks=blocks)
    if arch in {"transformer", "chessformer"}:
        return SquareTransformerPolicyNet(
            embed_dim=embed_dim,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    raise ValueError(f"unknown architecture: {arch}")
