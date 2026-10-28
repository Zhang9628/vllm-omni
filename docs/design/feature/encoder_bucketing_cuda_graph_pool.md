# Length-bucketing + CUDA graph pooling for variable-length encoders

## Summary

A reusable, encoder-agnostic building block for serving models whose encoder
takes a single forward pass over an independent batch of **variable-length**
inputs (audio frontends today, potentially vision and other modalities). It
combines three mechanisms:

1. **Length-bucketed padding** — pad each input up to the smallest rung of a
   discrete length ladder instead of a single global maximum;
2. **Bucketed CUDA graph capture** — capture one graph per
   `(batch_size, length_rung)` pair into a single **shared memory pool**, then
   dispatch by lookup so the encoder forward stays on graphs without shape
   explosion;
3. **Same-bucket grouped encoding + chunked eager fallback** — group a
   mixed-bucket batch by its native length so no row pays another row's
   padding, and bound transient activation memory on the fallback path.

The three pieces depend only on the input being variable-length, the encoder
running a single forward pass, and samples being independent within a batch
(no BatchNorm / cross-attention). The intent is to contribute this as a
framework-level building block rather than a per-model patch.

## Motivation

Variable-length encoders in vLLM-Omni currently suffer from two structural
inefficiencies:

- **Padding waste.** Inputs are padded to a fixed maximum length, so a 1 s
  utterance in a 30 s audio model is computed as 30 s — tens of x wasted
  encoder FLOPs and H2D traffic. Profiling of a production 8B encoder-decoder
  ASR model showed the encoder occupying ~43% of GPU time, almost all of it
  padded computation.
- **CUDA graph vs. variable-shape dilemma.** CUDA graphs require static
  shapes. With a variable length `T`, an encoder either falls back to eager
  (launch-bound) or captures an infeasible number of graphs.

There is no framework-level answer today: every model re-implements bucket
definition, padding, capture and lookup privately. Bucketing resolves the
dilemma by making the padded length take a small, discrete set of values, so
the graph pool only has to capture one graph per rung.

## Mechanism

### Length bucketing

`LengthBucketingConfig` holds an ascending ladder of rungs (in the caller's
native length unit — waveform samples, mel frames, patches, tokens) plus a
`max_length` cap. `padded_length(length)` returns the smallest rung `>=`
`length`, capped at `max_length`.

```python
from vllm_omni.model_executor.bucketed_encoder import LengthBucketingConfig

cfg = LengthBucketingConfig(buckets=(8000, 16000, 32000, 64000),
                            max_length=64000)
cfg.padded_length(16001)  # -> 32000  (smallest rung >= length)
cfg.padded_length(10**9)  # -> 64000  (capped, never above max_length)
```

For time-based ladders the ladder is usually deploy-tunable via an environment
variable, parsed once at import so the processor and the graph pool read the
same value:

```python
import os
from vllm_omni.model_executor.bucketed_encoder import parse_seconds_ladder

ladder = parse_seconds_ladder(
    os.environ.get("MY_MODEL_BUCKET_SECONDS", ""),
    sample_rate=16000,
    default_seconds=(0.5, 1, 2, 4, 8, 16, 30),
    max_seconds=30.0,
)
```

### Bucketed CUDA graph pool

`BucketedCUDAGraphRunner` captures one graph per `(batch_size, length_rung)`
pair. All graphs are captured with `torch.cuda.graph(..., pool=pool)` into one
shared memory pool, so peak activation memory is bounded by the **largest
single capture**, not the sum of all captures. Capture cost is negligible: a
10-rung batch ladder × 6 length buckets = 60 graphs.

The only model-specific hooks are the encoder module and a `make_inputs`
callback that builds dummy capture inputs for a given `(batch, length)`:

```python
from vllm_omni.model_executor.bucketed_encoder import BucketedCUDAGraphRunner

def make_inputs(batch: int, length: int):
    features = torch.zeros(batch, length, 128, device="cuda")
    mask = torch.zeros(batch, length // 4, dtype=torch.int32, device="cuda")
    return features, mask

runner = BucketedCUDAGraphRunner(
    model=encoder,
    length_buckets=ladder.feature_lengths(to_feature_length),
    make_inputs=make_inputs,
    capture_sizes=(1, 2, 4, 8),
    large_capture_sizes=(16, 32),   # only for lengths <= large_length_max
    large_length_max=798,           # short-audio-only large batches
    eager_chunk=8,                  # bound transient activation memory
)

features, mask = ...               # already bucketed to a rung length
(embeddings,) = runner.execute(features, mask)
embeddings = embeddings[:, :real_len].clone()  # trim before next replay
```

`execute` dispatches by `(batch, length)` lookup:

- batch is padded to the nearest captured batch size, replayed, then sliced
  back to the real batch;
- batches larger than the pad ladder are chunked first so short-`T` large
  batches still hit captured graphs;
- an unseen length rung falls back to **eager in chunks of `eager_chunk`**,
  bounding transient activation memory (`~ batch * length` per chunk).

### Same-bucket grouped encoding

A batch whose rows land in different length buckets is split by native length
and each group runs its own `(batch, length)` graph, instead of padding the
whole batch up to the batch maximum. Because rows are independent, this is
numerically identical to encoding each row solo (verify with solo-vs-batch
byte equality). Grouping is the model's responsibility — it knows the per-row
lengths on the CPU side — while the runner provides the bucketed graph lookup.

## Applicability constraints

The building block is *not* a drop-in for every encoder. It applies only when:

1. the input length is variable (no benefit for fixed-length input);
2. the encoder runs a single forward pass (decoder / KV-cache models are a
   different problem);
3. samples are independent within a batch (no BatchNorm / cross-attention).

Bucketing also only helps if the ladder matches the traffic length
distribution: a ladder too coarse near the P95 length gives back most of the
gain. Tune the ladder per workload and keep it configurable from day one.

## Relationship to other approaches

- **vLLM's encoder CUDA graph manager** (used by MiniCPM-o 4.5, see
  [MiniCPM-o 4.5 input encoder CUDA graphs](../minicpmo_encoder_cuda_graph.md))
  is a *shape cache*: it captures exact shapes on repeat and evicts. Bucketing
  is complementary — it *limits* the shape space to a small ladder up front, so
  a bucketed pool captures the rungs deterministically instead of discovering
  traffic shapes. The two can be combined (bucket the ladder, then let the
  manager cache the rungs).
- **Pad to exact length + eager** — no wasted compute, but every shape is
  dynamic, so graphs miss and kernels become launch-bound.
- **Pad to fixed max length** — graph-friendly but pays tens of x wasted
  encoder compute on short inputs (the baseline bucketing compares against).
- **`torch.compile` with dynamic shapes** — a plausible alternative; not yet
  benchmarked head-to-head for these encoders.
- **Decoder-side CUDA graph only** — vLLM already graphs the LLM decode path;
  the encoder forward (variable `T`, runs once per request) is a separate
  problem this does not cover.

## Evidence and benchmark

Internal production replay (an 8B encoder-decoder ASR chat endpoint, online
audio length P95 ≈ 5 s) measured **+50% QPS** at P95 < 300 ms with bucketing +
bucketed graphs versus pad-to-fixed-max, at −30%~40% latency. These numbers
validate that the optimization is worth generalizing; they are **not**
community-reproducible.

A reproducible vLLM-Omni benchmark on public models with variable-length
encoder stages is planned as the next step.

## Contribution path

1. Land this module and its tests (this change).
2. Agree on the public API (the two hooks: `make_inputs` + `length_buckets`).
3. Wire one public model with a variable-length encoder to it and publish the
   benchmark numbers above.
