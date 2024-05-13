# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch, MagicMock
from seamless_communication.denoise.demucs import Demucs, DenoisingConfig
import torch
from fairseq2.memory import MemoryBlock

class TestDemucs(unittest.TestCase):
    def test_init_works(self):
        config = DenoisingConfig(model="htdemucs", sample_rate=16000)
        demucs = Demucs(denoise_config=config)
        self.assertEqual(demucs.denoise_config.model, "htdemucs")
        self.assertEqual(demucs.denoise_config.sample_rate, 16000)

    @patch("seamless_communication.denoise.demucs.torchaudio.load")
    @patch("seamless_communication.denoise.demucs.Path")
    @patch("seamless_communication.denoise.demucs.sp.run")
    def test_denoise(self, mock_run, mock_path, mock_load):

        mock_run.return_value = MagicMock(returncode=0)
        mock_load.return_value = (torch.randn(1, 16000), 16000)
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.glob.return_value = [MagicMock()]
        mock_path.return_value.open.return_value.__enter__.return_value.read.return_value = b""
        config = DenoisingConfig(model="htdemucs", sample_rate=16000)
        demucs = Demucs(denoise_config=config)
        result = demucs.denoise(audio=None)

        mock_run.assert_called_once()
        mock_load.assert_called_once()
        self.assertIsInstance(result, MemoryBlock)
        