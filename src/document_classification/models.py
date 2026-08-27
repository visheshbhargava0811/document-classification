"""PyTorch model definitions.

These are copied verbatim (architecture-wise) from the training notebooks so that
the original trained weight files load cleanly with ``load_state_dict``:

- ``DocumentCornerModel`` -> ``dl_project_doc_extract_weights.pt``
- ``Unet``                -> ``model.pt``

Both are optional: the pipeline runs with classic OpenCV fallbacks when the
weight files are not present.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DocumentCornerModel(nn.Module):
    """CNN that regresses the 8 corner coordinates of a document.

    Output layout (matching the training dataset label order) is::

        [x_bl, x_br, x_tl, x_tr, y_bl, y_br, y_tl, y_tr]
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),  # assumes 128x128 input
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class Unet(nn.Module):
    """U-Net denoiser: grayscale (1-channel) in, grayscale out, sigmoid range."""

    def __init__(self) -> None:
        super().__init__()

        def convblock(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = convblock(1, 64)
        self.enc2 = convblock(64, 128)
        self.enc3 = convblock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.middle = convblock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = convblock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = convblock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = convblock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(m), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return torch.sigmoid(self.out(d1))
