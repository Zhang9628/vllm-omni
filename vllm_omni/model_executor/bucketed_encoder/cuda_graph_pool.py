# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bucketed CUDA-graph capture/replay with a shared memory pool for encoders.

CUDA graphs require static shapes. A variable-length encoder therefore has to
choose between falling back to eager (slower, launch-bound) every time the
length changes, or capturing a graph per possible shape (memory explosion).
Bucketing the length resolves the dilemma: pad each input to a ladder rung
(see :mod:`vllm_omni.model_executor.bucketed_encoder.bucketing`), so the
length only ever takes ``len(length_buckets)`` values, and capture one graph
per ``(batch_size, length_rung)`` pair.

To keep memory bounded, all graphs are captured into **one shared memory
pool** (``torch.cuda.graph(pool=...)``): the peak activation memory is that
of the single largest capture, not the sum of all captures.

The only model-specific hooks are

* ``model`` — the encoder ``nn.Module`` (a single forward pass over an
  independent batch; no BatchNorm / cross-attention between samples);
* ``make_inputs(batch, length) -> tuple[Tensor, ...]`` — builds dummy capture
  inputs for a given ``(batch, length)``; and
* ``length_buckets`` — the reachable ladder rungs, in the same unit as the
  primary input's length dimension.

Everything else (capture, shared pool, batch padding ladder, dispatch lookup,
chunked eager fallback, oversize-batch chunking) lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

_DEFAULT_NUM_WARMUP_ITERS = 2


def _as_tuple(outputs) -> tuple[torch.Tensor, ...]:
    """Normalize a model's return value to a tuple of tensors."""
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


class CUDAGraphRunner(nn.Module):
    """Captures and replays one encoder forward for one fixed input shape.

    The captured graph is ``nn.Module`` so profilers can correctly group the
    kernels launched inside it. ``capture`` records ``model(*inputs)`` into a
    ``torch.cuda.CUDAGraph`` (optionally into a caller-provided memory pool)
    and pins the input tensors as static buffers; ``forward`` copies fresh
    inputs into those buffers and replays.

    Args:
        model: The encoder module.
        num_warmup_iters: Number of warm-up forward passes before capture, so
            autotuned kernels (e.g. Triton) do not pollute the graph.
    """

    def __init__(self, model: nn.Module, num_warmup_iters: int = _DEFAULT_NUM_WARMUP_ITERS):
        super().__init__()
        self.model = model
        self.num_warmup_iters = num_warmup_iters

        self.input_buffers: list[torch.Tensor] = []
        self.output_buffers: tuple[torch.Tensor, ...] = ()
        self._graph: torch.cuda.CUDAGraph | None = None

    @property
    def graph(self) -> torch.cuda.CUDAGraph:
        assert self._graph is not None, "capture() must be called before graph"
        return self._graph

    @torch.inference_mode()
    def capture(self, *inputs: torch.Tensor, pool=None) -> None:
        """Capture ``model(*inputs)`` into a CUDA graph.

        Args:
            *inputs: Example tensors. Their shapes become the graph's static
                shapes, and the tensors themselves become the static buffers
                ``forward`` copies into.
            pool: Optional memory pool. Pass the pool returned by the first
                captured graph to make all graphs share one pool.
        """
        assert self._graph is None, "capture() may only be called once"

        device = inputs[0].device

        # Warm up outside the graph so autotune / first-touch kernels are not
        # recorded. One iteration is not enough for torch.compile.
        for _ in range(self.num_warmup_iters):
            self.model(*inputs)
        torch.accelerator.synchronize(device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=pool):
            outputs = self.model(*inputs)
        torch.accelerator.synchronize(device)

        self.input_buffers = list(inputs)
        self.output_buffers = _as_tuple(outputs)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Copy ``inputs`` into the static buffers and replay the graph."""
        if len(inputs) != len(self.input_buffers):
            raise ValueError(
                f"Expected {len(self.input_buffers)} inputs, got {len(inputs)}")
        for buffer, tensor in zip(self.input_buffers, inputs):
            buffer.copy_(tensor, non_blocking=True)
        self.graph.replay()
        return self.output_buffers


def _pad_to_batch_size(tensor: torch.Tensor, padded_batch: int) -> torch.Tensor:
    """Pad ``tensor``'s dim-0 to ``padded_batch`` with zeros."""
    current = tensor.size(0)
    if current == padded_batch:
        return tensor
    pad_size = padded_batch - current
    padding = torch.zeros(
        (pad_size,) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)


class BucketedCUDAGraphRunner:
    """Pool of one CUDA graph per ``(batch_size, length_rung)`` pair.

    Capture sizes and length buckets combine into ``len(capture_sizes) *
    len(length_buckets)`` graphs (plus ``large_capture_sizes`` for short
    lengths). All graphs share one memory pool, so peak activation memory is
    bounded by the largest single capture.

    Args:
        model: The encoder module.
        length_buckets: Reachable length rungs (in the primary input's length
            unit). Usually ``LengthBucketingConfig.feature_lengths(...)``.
        make_inputs: ``(batch, length) -> tuple[Tensor, ...]`` building dummy
            capture inputs on the CUDA device.
        capture_sizes: Batch sizes captured for every length bucket.
        large_capture_sizes: Extra batch sizes captured only for lengths
            ``<= large_length_max`` (long input + large batch would exhaust
            activation memory).
        large_length_max: Length threshold below which ``large_capture_sizes``
            are captured. ``None`` disables the large-batch captures.
        eager_chunk: Chunk size for the eager fallback, bounding transient
            activation memory (``activation ~ bs * length``).
        num_warmup_iters: Warm-up iterations per capture.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        length_buckets: Sequence[int],
        make_inputs: Callable[[int, int], Sequence[torch.Tensor]],
        capture_sizes: Sequence[int] = (1, 2, 4, 8),
        large_capture_sizes: Sequence[int] = (),
        large_length_max: int | None = None,
        eager_chunk: int = 8,
        num_warmup_iters: int = _DEFAULT_NUM_WARMUP_ITERS,
    ) -> None:
        self.model = model
        self.make_inputs = make_inputs
        self.num_warmup_iters = num_warmup_iters
        self.eager_chunk = eager_chunk

        self.length_buckets = sorted(set(length_buckets))
        if not self.length_buckets:
            raise ValueError("length_buckets must be non-empty")

        self.capture_sizes = sorted(set(capture_sizes))
        if not self.capture_sizes:
            raise ValueError("capture_sizes must be non-empty")
        self.large_capture_sizes = sorted(set(large_capture_sizes))
        self.large_length_max = large_length_max

        # Any batch <= max_pad_size is padded up to the nearest ladder rung,
        # so a replay always matches a captured batch size.
        pad_ladder = sorted(set(self.capture_sizes) | set(self.large_capture_sizes))
        self.max_pad_size = pad_ladder[-1]
        self._bs_to_padded: list[int] = [0] * (self.max_pad_size + 1)
        for bs in range(1, self.max_pad_size + 1):
            self._bs_to_padded[bs] = next(s for s in pad_ladder if s >= bs)

        self.graph_runners: dict[tuple[int, int], CUDAGraphRunner] = {}
        self.capture_all()

    @torch.inference_mode()
    def capture_all(self) -> None:
        """Capture one graph per ``(batch, length)`` pair into a shared pool."""
        graph_pool = None
        for length in self.length_buckets:
            batch_sizes = list(self.capture_sizes)
            if (self.large_length_max is not None
                    and self.large_capture_sizes
                    and length <= self.large_length_max):
                batch_sizes += list(self.large_capture_sizes)
            for batch in sorted(set(batch_sizes)):
                inputs = tuple(self.make_inputs(batch, length))
                runner = CUDAGraphRunner(self.model, self.num_warmup_iters)
                runner.capture(*inputs, pool=graph_pool)
                if graph_pool is None:
                    graph_pool = runner.graph.pool()
                self.graph_runners[(batch, length)] = runner

    def _pad_inputs(self, inputs: Sequence[torch.Tensor],
                    padded_batch: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            _pad_to_batch_size(t, padded_batch) for t in inputs)

    @torch.inference_mode()
    def execute(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the encoder over ``inputs`` via graph replay (or eager fallback).

        The *primary* input is ``inputs[0]``; its dim-0 is the batch and its
        dim-1 the length used as the ``(batch, length)`` lookup key. Auxiliary
        tensors must already be sliced to match ``length``; only their batch
        dim is padded here.

        Returns:
            A tuple of output tensors, batch-sliced back to the original batch
            size. The caller is responsible for trimming per-row length (e.g.
            ``out[0][:, :real_len]``) before the next replay overwrites the
            graph output buffers.
        """
        if not inputs:
            raise ValueError("execute() requires at least one input tensor")
        batch = inputs[0].size(0)
        length = inputs[0].size(1)

        # Oversized batches are chunked first, so short-T large batches can
        # still hit captured graphs instead of going fully eager.
        if batch > self.max_pad_size:
            step = self.max_pad_size
            chunks = [
                self.execute(*(t[i:i + step] for t in inputs))
                for i in range(0, batch, step)
            ]
            return tuple(
                torch.cat([c[k] for c in chunks], dim=0)
                for k in range(len(chunks[0])))

        padded_batch = self._bs_to_padded[batch]
        runner = self.graph_runners.get((padded_batch, length))
        if runner is not None:
            padded_inputs = self._pad_inputs(inputs, padded_batch)
            outputs = runner(*padded_inputs)
            return tuple(o[:batch] for o in outputs)

        # Unseen length rung: eager in small chunks to bound transient
        # activation memory (~ bs * length per chunk).
        chunks = [
            _as_tuple(self.model(*(t[i:i + self.eager_chunk] for t in inputs)))
            for i in range(0, batch, self.eager_chunk)
        ]
        return tuple(
            torch.cat([c[k] for c in chunks], dim=0)
            for k in range(len(chunks[0])))
