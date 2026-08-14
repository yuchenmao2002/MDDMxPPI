# Masked expression diffusion model

This package models one processed PBS cell as the full, fixed sequence of
19,295 genes. Expression values are nonnegative continuous values; forward
corruption replaces selected values with one discrete absorbing `MASK` state.
The decoder defines a hurdle distribution for the clean value at every gene.

This is therefore a continuous marked-unmasking (or reveal-order) density
model. It is not a finite categorical D3PM, and its objective must not be
described as the transition-KL ELBO of a categorical diffusion model.

## Fixed tensor contracts

```text
clean/visible expression       float [B, 19295, 1]
diffusion time                 float32 [B]
diffusion mask                 bool [B, 19295], True = absorbing MASK
Geneformer source table        float32 [19295, 1152]
model hidden states            float [B, 19295, 512]
decoder projection             float [B, 19295, 3]
point prediction               optional float32 [B, 19295, 1]
each distribution parameter    float32 [B, 19295, 1]
```

The gene order is exactly mapping index `0..19294`. There is no padding, gene
subset, or ordinary positional embedding. A diffusion-masked gene is still a
valid attention query, key, and value. A numerical expression value of zero
never means `MASK`; only the boolean diffusion mask defines corruption state.

## Model data flow

```text
Geneformer table [G,1152] -> trainable projection -> gene identity G [G,512]

expression [B,G,1] -> shared 1->32->512 SiLU MLP -> E0 [B,G,512]
                                                mask=True -> learned [512] state

H0 = G[None,:,:] + where(mask[...,None], absorbing_state, E0)
H0 -> L interchangeable denoising blocks -> final LayerNorm -> HL
HL -> shared Linear(512,3) -> hurdle distribution parameters
```

The standard mixer is a non-causal softmax FAVOR+ Performer with eight
64-dimensional heads and 256 random features per head. The external block
boundary remains `[B,G,512] -> [B,G,512]`; later hierarchical blocks may change
resolution internally only.

The aggressive training default uses sequence chunks of 8,192 tokens and does
not enable block-level activation checkpointing. Chunking still bounds the live
forward feature-map workspace. Activation checkpointing remains a supported
fallback for configurations that exceed the available accelerator memory.

Time is deliberately not injected into the expression/identity embeddings,
Performer blocks, or decoder. Every block still receives
`DenoiserContext(diffusion_time, diffusion_mask)` to keep the interface open to
future blocks. The current objective, however, uses the exact supplied time;
time is not inferred from the observed mask count for loss weighting.

## Three-channel hurdle decoder

For each cell `b` and gene `i`, the shared decoder produces three raw channels.
They are exposed as:

```text
detection_logits    a     float32 [B,G,1]
positive_location   mu    float32 [B,G,1]
positive_scale      sigma float32 [B,G,1]
```

The scale is transformed inside the decoder as

```text
sigma = min_scale + softplus(raw_scale)
```

The shared projection and all following distribution transforms run in FP32,
including when the surrounding backbone is under FP16/BF16 autocast. For a
finite raw scale, `sigma` is strictly positive; the loss rejects any non-finite
decoder parameter. With
`rho = sigmoid(detection_logits)`, the conditional clean-expression density is

```text
p(x | h) = (1-rho) * delta_0(x) + rho * f_+(x; mu, sigma).
```

`f_+` is a Gaussian truncated to the strictly positive half-line:

```text
f_+(x; mu, sigma)
    = Normal(x; mu, sigma) / Phi(mu / sigma),  x > 0.
```

When requested, `DecoderOutput.point_prediction` is `[B,G,1]` and is the
distribution expectation rather than a separately trained regression channel:

```text
E[x | h] = rho * (mu + sigma * phi(mu/sigma) / Phi(mu/sigma)).
```

It is intended for MSE/correlation reporting and deterministic imputation. The
training and NLL-validation path consumes only `distribution_parameters` and
therefore returns `point_prediction=None`, avoiding the truncated-Normal mean
calculation over all genes. Direct decoder and denoiser calls compute the point
prediction by default for inference compatibility.

## Forward corruption and training objective

The deterministic denoiser never samples corruption. The training wrapper
samples one independent `t ~ Uniform[0,1)` per cell and conditionally
independent `Bernoulli(t)` mask decisions per gene. It does not force at least
one mask. Callers may explicitly supply a consistent `(t, mask)` pair,
including no masks at `t=0` and all masks at `t=1`.

Let `D = 1[x > 0]`. The complete per-position hurdle negative log likelihood is

```text
ell(x) = (1-D) * softplus(a)
       + D * (softplus(-a) - log f_+(x; mu, sigma)).
```

The zero and positive terms are assembled at each position before reduction.
They must not be normalized independently or recombined with an arbitrary class
weight, because doing so changes the declared probability model.

For the absorbing schedule `alpha(t)=1-t`, the continuous-time weight is

```text
w(t) = -alpha'(t) / (1-alpha(t)) = 1/t.
```

The local training objective is

```text
weighted_nll_sum = sum_b sum_i mask[b,i] * ell[b,i] / t[b]
normalizer       = B * G
loss             = weighted_nll_sum / normalizer
```

The fixed `B*G` denominator is essential. Dividing by the random number of
masked positions would add an unwanted time-dependent weighting. At `t=0`, a
valid row contains no masks and contributes a differentiable zero; no numerical
`0/0` is evaluated. Distribution transforms, NLL evaluation, time weighting,
and reductions are performed in FP32.

For a strongly negative standardized location `z=mu/sigma`, the implementation
does not directly add an `O(z^2)` quadratic term to `log Phi(z)`. It uses the
equivalent `erfcx` form after analytically cancelling the two quadratic terms,
which keeps the positive-value NLL and its gradients stable far into the
negative tail.

The training output also reports local sufficient statistics:

```text
weighted_nll_sum
normalizer
cell_count
masked_count
masked_zero_count
masked_positive_count
weighted_zero_nll_sum
weighted_positive_nll_sum
```

The two weighted branch sums are diagnostics whose sum equals the weighted NLL
sum; they are not separate optimization objectives.

### Distributed reduction

For uneven per-rank batch sizes, reduce the weighted numerator and cell count
globally and use

```text
global_normalizer = global_cell_count * G.
```

Because PyTorch DDP averages gradients across `world_size` ranks, rank `r`
must backpropagate

```text
world_size * local_weighted_nll_sum[r] / global_normalizer.
```

With equal local batch sizes, backpropagating each local `.loss` is equivalent.
Counts and detached diagnostic sums may be all-reduced for logging. Do not
all-reduce a differentiable numerator in place.

## Parameter and checkpoint impact

The probabilistic decoder has `3*512+3 = 1,539` trainable parameters, compared
with 513 in the former scalar linear head: an increase of only 1,026 parameters.
For the fixed dimensions in this package, the total trainable-parameter count is

```text
N(L) = 22,837,699 + 3,150,848 * L,
```

where `L` is the number of Performer blocks. For example, `L=6` gives
41,742,787 trainable parameters. Random-feature projection matrices are fixed
buffers and are not included. Decoder tensor shapes and loss semantics changed,
so the architecture version is
`masked-expression-diffusion-v2-hurdle-truncated-normal`; a former scalar-head
checkpoint must not be loaded as a strict v2 checkpoint.

## Reverse sampling

`reverse_sampler.py` implements the baseline unconditional reverse process.
For a configured number of denoiser evaluations `K`, its linear grid is

```text
1 = t_K > t_{K-1} > ... > t_0 = 0,  t_k = k/K.
```

Every step uses the same frozen denoiser weights. A token that is still masked
at `t_k` is revealed at `t_{k-1}` with conditional probability

```text
r_k = (t_k - t_{k-1}) / t_k.
```

The reveal gate is independent of decoder confidence. Values for newly
revealed tokens are sampled from the declared hurdle distribution: first draw
the positive event from `Bernoulli(sigmoid(a))`; return exact zero when absent,
and otherwise draw from the zero-truncated Normal `f_+(mu,sigma)`. Revealed
values remain fixed forever. Unselected tokens remain in the absorbing state,
their current prediction is discarded, and the denoiser predicts them again
at the next step using the enlarged visible context. There is no re-masking.
The final transition has reveal probability one, so no MASK token remains.

The zero-truncated Normal sampler uses rejection algorithms for both ordinary
and far-tail regimes rather than an unstable rounded inverse CDF. Randomness is
owned by a caller-supplied device-local `torch.Generator`, enabling exact replay
for a fixed software/hardware configuration and batching choice.

`scripts/sample_masked_diffusion.py` loads one explicitly selected trusted
checkpoint, verifies its 19,295-gene order, generates cells in bounded batches,
and atomically writes a CSR-backed `.h5ad`. The output remains in the processed
PBS expression domain used for training; it is not a raw integer-count matrix.
Training checkpoint v3 contains Python/NumPy RNG objects, so the CLI requires
an explicit `--trust-checkpoint` confirmation before unrestricted checkpoint
deserialization. Optimizer, scheduler, scaler, and training RNG states are not
restored for inference.

Example:

```bash
python scripts/sample_masked_diffusion.py \
  --checkpoint outputs/baseline/<run>/checkpoints/best.pt \
  --output outputs/generated/sample_k20.h5ad \
  --num-cells 1000 \
  --num-steps 20 \
  --batch-size 16 \
  --seed 42 \
  --device cuda:0 \
  --precision bf16 \
  --trust-checkpoint
```

For PBS, the companion submission script keeps the checkpoint, cell count,
reverse-step count, physical batch size, seed, precision, and output settings
in one configuration block at the top:

```bash
qsub scripts/submit_sample_masked_diffusion.pbs

# Override only the experiment-specific values without editing the script.
qsub -v CHECKPOINT_PATH=outputs/baseline/<run>/checkpoints/best.pt,NUM_CELLS=5000,NUM_STEPS=20,BATCH_SIZE=8,SEED=42 \
  scripts/submit_sample_masked_diffusion.pbs
```

Each job writes one `.h5ad` and its detailed log under a unique directory in
`outputs/generated/`. The repository's `outputs` symlink places these files on
scratch. The scheduler resolves `#PBS -o outputs/pbs_logs/` before the shell
script starts, so that directory must already exist when `qsub` is called.

The current Performer still does not numerically consume `diffusion_time`; its
state changes across reverse steps only through the monotonically shrinking
mask and newly visible expression values. This is an explicit limitation of
the v2 baseline, not an omission in the sampler API.

## Runtime dependencies

The implementation requires PyTorch and `safetensors`; contract tests use
pytest. The repository intentionally does not pin a CUDA-specific PyTorch wheel
because the execution environment must select a build compatible with its
driver.
