# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import unittest
from argparse import Namespace
from unittest.mock import Mock
from seamless_communication.segment.silero_vad import SileroVADSegmenter, Segment
import numpy as np


class TestSileroVADSegmenter(unittest.TestCase):
    def test_init_works(self):
        segmenter = SileroVADSegmenter(
          sample_rate=16000, 
          chunk_size_sec=10, 
          pause_length=0.5)
        self.assertEqual(segmenter.sample_rate, 16000)
        self.assertEqual(segmenter.chunk_size_sec, 10)
        self.assertEqual(segmenter.pause_length, 0.5)


    def test_segment_long_input(self):
        self.segmenter = SileroVADSegmenter(
          sample_rate=16000, 
          chunk_size_sec=10, 
          pause_length=0.5)
        self.segmenter.get_speech_timestamps = Mock(
          return_value=[{0: 0, 1: 10000}, 
          {0: 20000, 1: 30000}])
        segments = self.segmenter.segment_long_input(audio=None)
        expected_segments = [[0, 10000], [20000, 30000]]
        self.assertEqual(segments, expected_segments)


    def test_recursive_split(self):
        segmenter = SileroVADSegmenter(
          sample_rate=16000, 
          chunk_size_sec=10,
          pause_length=0.5)
        sgm = Segment(0, 10000, np.random.rand(10000))
        segments = []
        max_segment_length = 5000
        min_segment_length = 1000
        window_size_samples = 100
        threshold = .5

        segmenter.recursive_split(
          sgm, 
          segments, 
          max_segment_length, 
          min_segment_length, 
          window_size_samples, 
          threshold)

        assert all([seg.duration < max_segment_length for seg in segments])
        assert all([seg.duration > min_segment_length for seg in segments])
