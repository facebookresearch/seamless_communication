# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Mixins + common for fairseq2 simuleval agents
"""

from simuleval.data.segments import Segment
from simuleval.agents.states import AgentStates as AgentStatesOrig


class EarlyStoppingMixin:
    def reset_early(self) -> None:
        """
        Implement to override for different behavior on a reset that
        happens before EOS
        """
        raise NotImplementedError()


class AgentStates(AgentStatesOrig):  # type: ignore
    def update_target(self, segment: Segment) -> None:
        """An AgentStates impl which doesn't update states.target"""
        self.target_finished = segment.finished


class NoUpdateTargetMixin:
    """A shortcut to make agents default to the AgentStates impl above"""

    def build_states(self) -> AgentStates:
        return AgentStates()
