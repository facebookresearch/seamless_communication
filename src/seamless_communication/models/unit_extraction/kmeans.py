# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
import numpy as np
from fairseq2.typing import Device
from seamless_communication.assets import download_manager


class KmeansModel(nn.Module):
    def __init__(self, kmeans_uri: str, device: Device):
        super().__init__()
        km_path = download_manager.download_checkpoint(kmeans_uri, kmeans_uri)
        km_model = np.load(km_path)
        centroids_numpy = km_model.transpose()
        centroids = torch.from_numpy(centroids_numpy)

        self.centroids = centroids.to(device)
        self.centroid_norm = (self.centroids**2).sum(0, keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        dist: Tensor = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.centroids)
            + self.centroid_norm
        )
        return dist.argmin(dim=-1)
