# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
import numpy as np
from pathlib import Path
from fairseq2.typing import Device
from seamless_communication.assets import download_manager


class KmeansModel(nn.Module):
    @staticmethod
    def load_model(km_path: Path, device: Device) -> "KmeansModel":
        km_model = np.load(km_path)
        centroids_numpy = km_model.transpose()
        return KmeansModel(torch.from_numpy(centroids_numpy), device)

    def __init__(self, centroids: Tensor, device: Device):
        super().__init__()

        self.centroids = nn.Parameter(centroids, requires_grad=False).to(device)
        self.centroid_norm = nn.Parameter(
            (centroids**2).sum(0, keepdims=True), requires_grad=False
        ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        dist: Tensor = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.centroids)
            + self.centroid_norm
        )
        return dist.argmin(dim=-1)


if __name__ == "__main__":
    kmeans_uri = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    km_path = download_manager.download_checkpoint(kmeans_uri, "kmeans_10k")
    device = torch.device("cuda:1")
    model = KmeansModel.load_model(km_path, device)
    t = torch.randn((1000, 1280), device=device, dtype=torch.float32)
    units = model(t)
    print(units)
