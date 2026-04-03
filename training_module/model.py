from __future__ import annotations

import torch
import torch.nn as nn


class DeepfakeVideoModel(nn.Module):
    """
    Video-level binary classifier.

    Output logits order is:
    - index 0: FAKE
    - index 1: REAL
    """

    def __init__(
        self,
        backbone_name: str = "xception",
        pretrained: bool = True,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required. Install it with: pip install timm") from exc

        self.backbone_name = backbone_name
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = getattr(self.feature_extractor, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Unable to infer num_features for backbone: {backbone_name}")

        self.feature_dim = int(feature_dim)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.feature_extractor(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features.view(batch_size, seq_len, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_features = self.encode_frames(x)
        lstm_out, _ = self.lstm(frame_features)

        # Mean pooling is more stable than taking only the last timestep.
        temporal_features = torch.mean(lstm_out, dim=1)
        temporal_features = self.dropout(temporal_features)
        return self.classifier(temporal_features)


def build_model(
    backbone_name: str,
    pretrained: bool,
    hidden_dim: int,
    lstm_layers: int,
    dropout: float,
) -> DeepfakeVideoModel:
    return DeepfakeVideoModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
