# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mixins for fairseq2 simuleval agents
"""


class EarlyStoppingMixin:
    def reset_early(self) -> None:
        """
        Implement to override for different behavior on a reset that
        happens before EOS
        """
        raise NotImplementedError()
