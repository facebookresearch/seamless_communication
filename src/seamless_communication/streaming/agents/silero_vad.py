# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
import queue
import time
from argparse import ArgumentParser, Namespace
from os import SEEK_END
from typing import Any, List, Optional, Union

import numpy as np
import torch
from seamless_communication.streaming.agents.mixins import EarlyStoppingMixin
from simuleval.agents import AgentStates, SpeechToSpeechAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import EmptySegment, Segment, SpeechSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SPEECH_PROB_THRESHOLD = 0.6


class SileroVADStates(EarlyStoppingMixin, AgentStates):
    def __init__(self, args: Namespace) -> None:
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.silence_limit_ms = args.silence_limit_ms
        self.speech_soft_limit_ms = args.speech_soft_limit_ms
        self.window_size_samples = args.window_size_samples
        self.chunk_size_samples = args.chunk_size_samples
        self.sample_rate = args.sample_rate
        self.debug = args.debug
        self.test_input_segments_wav = None
        self.debug_log(args)
        self.input_queue: queue.Queue[Segment] = queue.Queue()
        self.next_input_queue: queue.Queue[Segment] = queue.Queue()
        super().__init__()

    def clear_queues(self) -> None:
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
            self.input_queue.task_done()
        # move everything from next_input_queue to input_queue
        while not self.next_input_queue.empty():
            chunk = self.next_input_queue.get_nowait()
            self.next_input_queue.task_done()
            self.input_queue.put_nowait(chunk)

    def reset(self) -> None:
        super().reset()
        # TODO: in seamless_server, report latency for each new segment
        self.first_input_ts: Optional[float] = None
        self.silence_acc_ms = 0
        self.speech_acc_ms = 0
        self.input_chunk: np.ndarray[Any, np.dtype[np.int16]] = np.empty(
            0, dtype=np.int16
        )
        self.is_fresh_state = True
        self.clear_queues()
        self.model.reset_states()

    def reset_early(self) -> None:
        """
        Don't reset state before EOS
        """
        pass

    def get_speech_prob_from_np_float32(
        self, segment: np.ndarray[Any, np.dtype[np.float32]]
    ) -> List[Any]:
        t = torch.from_numpy(segment)
        speech_probs = []
        for i in range(0, len(t), self.window_size_samples):
            chunk = t[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                break
            speech_prob = self.model(chunk, self.sample_rate).item()
            speech_probs.append(speech_prob)
        return speech_probs

    def debug_log(self, m: Any) -> None:
        if self.debug:
            logger.info(m)

    def process_speech(
        self,
        segment: Union[np.ndarray[Any, np.dtype[np.float32]], Segment],
        tgt_lang: Optional[str] = None,
    ) -> None:
        """
        Process a full or partial speech chunk
        """
        queue = self.input_queue
        if self.source_finished:
            # current source is finished, but next speech starts to come in already
            self.debug_log("use next_input_queue")
            queue = self.next_input_queue

        # NOTE: we don't reset silence_acc_ms here so that once an utterance
        # becomes longer (accumulating more silence), it has a higher chance
        # of being segmented.
        self.silence_acc_ms = self.silence_acc_ms // 2

        if self.first_input_ts is None:
            self.first_input_ts = time.time() * 1000

        while len(segment) > 0:
            # add chunks to states.buffer
            i = self.chunk_size_samples - len(self.input_chunk)
            self.input_chunk = np.concatenate((self.input_chunk, segment[:i]))
            segment = segment[i:]
            self.is_fresh_state = False
            if len(self.input_chunk) == self.chunk_size_samples:
                queue.put_nowait(
                    SpeechSegment(
                        content=self.input_chunk, finished=False, tgt_lang=tgt_lang
                    )
                )
                self.input_chunk = np.empty(0, dtype=np.int16)

    def check_silence_acc(self, tgt_lang: Optional[str] = None) -> None:
        silence_limit_ms = self.silence_limit_ms
        if self.speech_acc_ms >= self.speech_soft_limit_ms:
            self.debug_log("increase speech threshold")
            silence_limit_ms = self.silence_limit_ms // 2
        self.debug_log(f"silence_acc_ms: {self.silence_acc_ms}")
        if self.silence_acc_ms >= silence_limit_ms:
            self.debug_log("=== end of segment")
            # source utterance finished
            self.silence_acc_ms = 0
            self.speech_acc_ms = 0
            if self.input_chunk.size > 0:
                # flush partial input_chunk
                self.input_queue.put_nowait(
                    SpeechSegment(
                        content=self.input_chunk, tgt_lang=tgt_lang, finished=True
                    )
                )
                self.input_chunk = np.empty(0, dtype=np.int16)
            self.input_queue.put_nowait(EmptySegment(finished=True))
            self.source_finished = True

    def update_source(
        self, segment: Union[np.ndarray[Any, np.dtype[np.float32]], Segment]
    ) -> None:
        """
        Default value for the segment in the update_source method is a segment
        Class, for some reason this interface didn't align with other interfaces
        Adding this change here to support both np.ndarray and Segment class
        """
        tgt_lang = None
        if isinstance(segment, SpeechSegment):
            self.sample_rate = segment.sample_rate
            if hasattr(segment, "tgt_lang") and segment.tgt_lang is not None:
                tgt_lang = segment.tgt_lang
            if isinstance(segment.content, np.ndarray):
                segment = np.array(segment.content, dtype=np.float32)
            else:
                segment = segment.content
        speech_probs = self.get_speech_prob_from_np_float32(segment)
        chunk_size_ms = len(segment) * 1000 / self.sample_rate
        window_size_ms = self.window_size_samples * 1000 / self.sample_rate
        if all(i <= SPEECH_PROB_THRESHOLD for i in speech_probs):
            if self.source_finished:
                return
            self.debug_log("got silent chunk")
            if not self.is_fresh_state:
                self.silence_acc_ms += chunk_size_ms
                self.check_silence_acc(tgt_lang)
            return
        elif speech_probs[-1] <= SPEECH_PROB_THRESHOLD:
            self.debug_log("=== start of silence chunk")
            # beginning = speech, end = silence
            # pass to process_speech and accumulate silence
            self.speech_acc_ms += chunk_size_ms
            self.process_speech(segment, tgt_lang)
            # accumulate contiguous silence
            for i in range(len(speech_probs) - 1, -1, -1):
                if speech_probs[i] > SPEECH_PROB_THRESHOLD:
                    break
                self.silence_acc_ms += window_size_ms
            self.check_silence_acc(tgt_lang)
        elif speech_probs[0] <= SPEECH_PROB_THRESHOLD:
            self.debug_log("=== start of speech chunk")
            # beginning = silence, end = speech
            # accumulate silence , pass next to process_speech
            for i in range(0, len(speech_probs)):
                if speech_probs[i] > SPEECH_PROB_THRESHOLD:
                    break
                self.silence_acc_ms += window_size_ms
            self.check_silence_acc(tgt_lang)
            self.speech_acc_ms += chunk_size_ms
            self.process_speech(segment, tgt_lang)
        else:
            self.speech_acc_ms += chunk_size_ms
            self.debug_log("======== got speech chunk")
            self.process_speech(segment, tgt_lang)

    def debug_write_wav(self, chunk: np.ndarray[Any, Any]) -> None:
        if self.test_input_segments_wav is not None:
            self.test_input_segments_wav.seek(0, SEEK_END)
            self.test_input_segments_wav.write(chunk)


class SileroVADAgent(SpeechToSpeechAgent):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.chunk_size_samples = args.chunk_size_samples
        self.args = args

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--window-size-samples",
            default=512,  # sampling_rate // 1000 * 32 => 32 ms at 16000 sample rate
            type=int,
            help="Window size for passing samples to VAD",
        )
        parser.add_argument(
            "--chunk-size-samples",
            default=5120,  # sampling_rate // 1000 * 320 => 320 ms at 16000 sample rate
            type=int,
            help="Chunk size for passing samples to model",
        )
        parser.add_argument(
            "--silence-limit-ms",
            default=700,
            type=int,
            help="send EOS to the input_queue after this amount of silence",
        )
        parser.add_argument(
            "--speech-soft-limit-ms",
            default=12_000,  # after 15s, increase the speech threshold
            type=int,
            help="after this amount of speech, decrease the speech threshold (segment more aggressively)",
        )
        parser.add_argument(
            "--debug",
            default=False,
            type=bool,
            help="Enable debug logs",
        )

    def build_states(self) -> SileroVADStates:
        return SileroVADStates(self.args)

    def policy(self, states: SileroVADStates) -> Action:
        states.debug_log(
            f"queue size: {states.input_queue.qsize()}, input_chunk size: {len(states.input_chunk)}"
        )
        content: np.ndarray[Any, Any] = np.empty(0, dtype=np.int16)
        is_finished = states.source_finished
        tgt_lang = None
        while not states.input_queue.empty():
            chunk = states.input_queue.get_nowait()
            states.input_queue.task_done()
            if tgt_lang is None:
                tgt_lang = chunk.tgt_lang
            content = np.concatenate((content, chunk.content))

        states.debug_write_wav(content)
        if is_finished:
            states.debug_write_wav(np.zeros(16000))

        if len(content) == 0:  # empty queue
            if not states.source_finished:
                return ReadAction()
            else:
                # NOTE: this should never happen, this logic is a safeguard
                segment = EmptySegment(finished=True)
        else:
            segment = SpeechSegment(
                content=content.tolist(),
                finished=is_finished,
                tgt_lang=tgt_lang,
            )

        return WriteAction(segment, finished=is_finished)

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: None) -> SileroVADAgent:
        return cls(args)
