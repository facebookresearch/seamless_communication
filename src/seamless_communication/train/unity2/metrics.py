# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, Optional, Sequence, Tuple, final
import torch
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean, Sum, Throughput
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.typing import override

from seamless_communication.models.unity.model import UnitYBatch

@final
class UnitYMetricBag(MetricBag):
    """Holds the common metrics of a UnitY model."""

    unit_nll_loss: Mean
    text_nll_loss: Mean
    duration_loss: Mean
    forward_sum_loss: Mean
    aux_loss: Mean
    batch_size: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_examples: Sum
    num_source_elements: Sum
    num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("unit_nll_loss", Mean(device=d), persistent=False)
        self.register_metric("text_nll_loss", Mean(device=d), persistent=False)
        self.register_metric("duration_loss", Mean(device=d), persistent=False)
        self.register_metric("forward_sum_loss", Mean(device=d), persistent=False)
        self.register_metric("aux_loss", Mean(device=d), persistent=False)

        self.register_metric("batch_size", Mean(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("elements_per_second", Throughput(device=d), persistent=False)  # fmt: skip

        self.num_examples = Sum(device=d)

        self.num_source_elements = Sum(device=d)
        self.num_target_elements = Sum(device=d)

    def update_metrics(
        self,
        batches: Sequence[UnitYBatch],
        losses: Sequence[Tensor],
        elapsed_time: float,
    ) -> None:
        """Update the metrics.

        :param batches:
            The batches processed by the model in the last training step.
        :param output:
            The losses generated by the model for each batch in ``batches``.
        :param elapsed_time:
            The total elapsed time to read and process ``batches``.
        """
        unit_nll_loss = torch.zeros((), dtype=torch.float64)
        text_nll_loss = torch.zeros((), dtype=torch.float64)
        duration_loss = torch.zeros((), dtype=torch.float64)
        forward_sum_loss = torch.zeros((), dtype=torch.float64)
        aux_loss = torch.zeros((), dtype=torch.float64)

        batch_size = torch.zeros((), dtype=torch.float64)

        num_source_elements = torch.zeros((), dtype=torch.float64)
        num_target_elements = torch.zeros((), dtype=torch.float64)
        num_target_text_elements = torch.zeros((), dtype=torch.float64)

        for batch, batch_loss in zip(batches, losses):
            unit_nll_loss += float(batch_loss["unit_nll_loss"])
            text_nll_loss += float(batch_loss["text_nll_loss"])
            duration_loss += float(batch_loss["duration_loss"])
            forward_sum_loss += float(batch_loss["forward_sum_loss"])
            aux_loss += float(batch_loss["aux_loss"])

            batch_size += batch.batch_size

            num_source_elements += batch.num_source_elements()
            num_target_elements += batch.num_target_elements() - batch.batch_size
            num_target_text_elements += batch.num_target_text_elements() - batch.batch_size

        unit_nll_loss /= num_target_elements * math.log(2)
        text_nll_loss /= num_target_text_elements * math.log(2)
        duration_loss /= num_target_elements
        forward_sum_loss /= num_target_elements * math.log(2)
        aux_loss /= num_target_elements * math.log(2)

        self.unit_nll_loss.update(unit_nll_loss, weight=num_target_elements)
        self.text_nll_loss.update(text_nll_loss, weight=num_target_text_elements)
        self.duration_loss.update(duration_loss, weight=num_target_elements)
        self.forward_sum_loss.update(forward_sum_loss, weight=num_target_elements)
        self.aux_loss.update(aux_loss, weight=num_target_elements)

        self.batch_size.update(batch_size * self.gang.size)

        self.elements_per_batch.update(num_target_elements * self.gang.size)

        self.elements_per_second.update(int(num_target_elements), elapsed_time)

        self.num_examples.update(batch_size)

        self.num_source_elements.update(num_source_elements)
        self.num_target_elements.update(num_target_elements)

    def reset_batch_metrics(self) -> None:
        """Reset the batch metrics to their initial state."""
        self.unit_nll_loss.reset()
        self.text_nll_loss.reset()
        self.duration_loss.reset()
        self.forward_sum_loss.reset()
        self.aux_loss.reset()
        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        values["elapsed_time"] = self.elements_per_second.elapsed_time_sec
