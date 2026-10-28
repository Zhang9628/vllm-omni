# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Micro-benchmark for the bucketed encoder path (Whisper + AudioAdapter).

Compares, on a synthetic CovoAudio-style encoder (no model weights needed):

* ``pad30_eager``  — a short utterance padded to the fixed 30 s cap, eager;
* ``bucket_eager`` — the same utterance padded to its 1 s bucket, eager;
* ``bucket_graph`` — the same, replayed from the bucketed CUDA graph pool.

The 30 s vs 1 s spread isolates the padding waste bucketing removes; the
eager vs graph spread isolates the CUDA-graph launch win.

Example:
    python benchmarks/kernels/benchmark_bucketed_encoder.py --batch 8 --iterations 100
"""

import argparse
import statistics
import time

import torch

from vllm_omni.model_executor.bucketed_encoder import BucketedCUDAGraphRunner
from vllm_omni.model_executor.models.covo_audio.covo_audio_llm import AudioAdapter


def build_encoder(n_mels: int, d_model: int, device: torch.device):
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    cfg = WhisperConfig(
        num_mel_bins=n_mels,
        d_model=d_model,
        encoder_layers=2,
        encoder_attention_heads=4,
        max_source_positions=2048,
    )
    encoder = WhisperEncoder(cfg).to(device)
    adapter = AudioAdapter(d_model, d_model, downsample=8).to(device)

    def encode(features: torch.Tensor) -> torch.Tensor:
        return adapter(encoder(features).last_hidden_state)

    return encode


def measure(operation, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.accelerator.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda:0")
    encode = build_encoder(args.n_mels, args.d_model, device)

    # 30 s cap (Whisper default) vs a 1 s bucket.
    long_frames = 30 * 100  # 3000
    bucket_frames = 1 * 100  # 100

    long_input = torch.randn(args.batch, long_frames, args.n_mels, device=device)
    bucket_input = torch.randn(args.batch, bucket_frames, args.n_mels, device=device)

    def make_inputs(batch: int, length: int):
        return (torch.randn(batch, length, args.n_mels, device=device),)

    runner = BucketedCUDAGraphRunner(
        encode,
        length_buckets=(bucket_frames,),
        make_inputs=make_inputs,
        capture_sizes=(args.batch,),
    )

    with torch.inference_mode():
        pad30 = measure(lambda: encode(long_input), args.warmup, args.iterations, device)
        bucket_eager = measure(lambda: encode(bucket_input), args.warmup, args.iterations, device)
        bucket_graph = measure(lambda: runner.execute(bucket_input), args.warmup, args.iterations, device)

    print(f"batch={args.batch} n_mels={args.n_mels} d_model={args.d_model}")
    print(f"pad30_eager     {pad30:8.3f} ms   (baseline, pad-to-30s)")
    print(f"bucket_eager    {bucket_eager:8.3f} ms   ({pad30 / bucket_eager:5.1f}x vs baseline)")
    print(f"bucket_graph    {bucket_graph:8.3f} ms   ({pad30 / bucket_graph:5.1f}x vs baseline, "
          f"{bucket_eager / bucket_graph:4.2f}x vs bucket-eager)")


if __name__ == "__main__":
    main()
