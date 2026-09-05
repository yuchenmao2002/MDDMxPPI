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

## Backbone variants

The backbone is selected with `--backbone-variant`. The four variants share
every dimension above and differ only in the mixer and the feed-forward layer,
so an ablation isolates one change at a time.

| variant | mixer | feed-forward | PPI assets read |
| --- | --- | --- | --- |
| `performer` | FAVOR+ | dense `d -> 4d -> d` | none |
| `ppil_attention` | PPI linear attention | dense `d -> 4d -> d` | spherical embedding |
| `ppil_ffn` | FAVOR+ | statically routed MoE | routing table |
| `ppil_full` | PPI linear attention | statically routed MoE | both |

`blocks/ppil_components.py` holds the only implementation of the two new
components; the three `blocks/ppil_*.py` files assemble them into the shared
pre-normalized residual shell. `blocks/performer.py` is unchanged, and
`ppil_ffn` imports its `PerformerSelfAttention` rather than copying it.

Two gene-indexed artifacts back the PPIL variants:

```text
data/processed/PPI/spherical_ppi_tau700_k4_b1_r64.safetensors
    embedding    float32 [19295,64], every row on the unit sphere
    free_mask    bool    [19295],    3,688 degenerate (ballast) rows
data/processed/PPI/ppi_moe_routing.safetensors
    expert_ids   int64   [19295,2],  ascending, so column 0 is not the top expert
    weights      float32 [19295,2],  rows sum to one
    prototypes   float32 [4,64]
```

Row `i` of both is gene `i` of the same 19,295-gene axis the denoiser uses, so
no identifier remapping exists anywhere in the model; the loader verifies the
`gene_ids == arange(19295)` self-assertion rather than trusting the filename.
The backbone owns one `PPIAssets` module and hands it to every block by
reference, so the state dict holds exactly one 5.17 MiB copy however many
layers are stacked. Those tensors are persistent buffers, so inference never
reopens `data/`.

Routing is fixed offline, per gene: there is no gating network, no router
logits and no load-balancing auxiliary loss, and every PPIL block returns an
empty `aux_losses`. A gene's output is the convex combination of its two
experts under the table's own weights, which sum to one; the free genes carry
exactly `0.5/0.5` there, so the code reads the table rather than special-casing
them.

Two properties of the shipped table are worth knowing when reading ablation
results. The final per-expert load is `[9668, 9555, 9592, 9775]` — max over min
is 1.023 — but experts 1 and 2 are each about 38.5% ballast, because all 3,688
free genes were water-filled into them, which dilutes whatever those two mean.
And although `T_route=0.03` is a sharp temperature, the routing is mostly not
near-deterministic: among real genes the dominant weight has median 0.6055 and
only 2.7% of them exceed 0.99.

## PPI linear attention

The mixer interpolates content attention with the PPI structural prior. Writing
`z_i` for gene `i` on the unit sphere, the approximated kernel is

```text
q~_i . k~_j = (1-lambda) * q_i.k_j  +  lambda * sigma_q * sigma_k * (z_i.z_j)
```

realized by feeding augmented vectors to the usual positive feature map:

```text
q~_i = [ sqrt(1-lambda) * q_i ; sqrt(lambda) * sigma_q * z_i ]   in R^(d_h+r)
k~_j = [ sqrt(1-lambda) * k_j ; sqrt(lambda) * sigma_k * z_j ]
```

so the random projection is `[256,128]` instead of `[256,64]` while the number
of random features is unchanged. `sigma_q` and `sigma_k` are stop-gradient
within-head RMS norms taken over the whole gene axis. They calibrate the PPI
half to the magnitude of the content half, which makes `E||q~_i||^2 = sigma_q^2`
for a fitted gene *independently of lambda*: lambda reallocates share without
changing scale. That is why the augmented vector must never be rescaled a
second time by its own width `d_h+r` — the only dimensional scaling is the
single `d_h^-1/4` on the content half.

`lambda(p_t) in (0,1)^h` is the prior share, one value per attention head per
cell. It is produced by each layer's own gate from the *realized* mask rate
`p_t = mean(diffusion_mask)` — the quantity the reverse sampler can actually
observe at every step — through a Fourier representation with one band per
head:

```text
gamma(p_t) = [sin(2^0 pi p_t), cos(2^0 pi p_t), ..., sin(2^7 pi p_t), cos(2^7 pi p_t)]
lambda     = sigmoid( Linear(64,8) . SiLU . Linear(16,64) ( gamma(p_t) ) )
```

The backbone derives `p_t` and `gamma` once per forward pass, since they are
identical for every layer, and passes them through `DenoiserContext`; the gate
itself is per-layer. Both square roots are evaluated as `exp(0.5*logsigmoid())`
rather than `sqrt(sigmoid())`, which is the same number without the infinite
derivative that FP32 sigmoid saturation would otherwise produce.

Free (ballast) genes get a zero vector in the `r` block. Their rows are unit
vectors in the artifact exactly like the fitted ones, so without this they
would contribute full-magnitude spurious similarity — self-similarity exactly
1.0. Their content half still carries `sqrt(1-lambda)`, so as lambda grows
their logits contract and their attention flattens toward uniform. This is a
deliberate, measured consequence, not an oversight.

Three arithmetic details differ from `blocks/performer.py` and are intentional.
The stabilizing shifts `c_Q` (per token) and `c_K` (one global scalar per batch
and head) are maxima over the *complete* log-feature including its norm term,
which makes the largest exponent exactly zero; both cancel between the
numerator and denominator and so cannot change the result. No per-coordinate
constant is added inside the feature map, because such a constant gives the
approximated kernel a uniform background unrelated to both content and PPI that
then accumulates linearly over all 19,295 keys. The denominator uses `+ epsilon`
rather than `clamp_min(epsilon)`, which is differentiable everywhere.

Because `sigma` is defined over the whole gene axis, the attention makes four
chunked passes rather than the Performer's three: one to accumulate the RMS
norms, one for the global key shift, one for the key/value sufficient
statistics, and one for the query outputs.

## Statically routed feed-forward

Each expert repeats the dense network's shape exactly: up to the expert width,
GELU, the same internal dropout, back down to `d_model`, both projections
biased. Only the sharing pattern differs — the dense network applies one map to
every gene, the routed one applies a gene-dependent convex mixture of four.

The expert width is `expert_ffn_multiplier * d_model` and is set to 2, giving
`512 -> 1024 -> 512`. That choice is what makes the ablation clean: under top-2
routing it holds *three* quantities equal to the dense baseline at once.

```text
                          dense        w=2 experts
per-token MAC             2,097,152    2,097,152     (1.00x)
per-token hidden units    2,048        2,048         (1.00x)
parameters per gene       2,099,712    2,100,224     (1.00x, +0.02% from biases)
distinct parameters       2,099,712    4,200,448     (2.00x)
```

Only the last row grows, and it grows because different genes use different
maps — which is the hypothesis under test, not a confound. The routing is
frozen, so an expert only ever sees its own fixed ~9,600 genes and no gene can
reach beyond its two: the doubled total is the bookkeeping cost of letting gene
groups differ, not capacity any single gene can draw on. That argument would
not hold for a learned gate.

Matching *total* parameters instead would mean width `1d`, which halves the
per-gene parameters, the hidden width and the compute all at once; a loss there
could not be attributed to the routing rather than to the smaller network.

The routing weights multiply the expert outputs in the residual-stream dtype.
Under BF16 autocast that rounds each weight by 0.10% at the median (0.39% worst
case) and leaves a per-gene gain error of at most 0.20%, which sits below the
0.14% median error BF16 already imposes on the expert outputs themselves;
weighting in FP32 would cost a 1.26 GB transient per expert for no real gain.

The aggressive training default uses sequence chunks of 8,192 tokens and does
not enable block-level activation checkpointing. Chunking still bounds the live
forward feature-map workspace. Activation checkpointing remains a supported
fallback for configurations that exceed the available accelerator memory.

Diffusion time is not injected into the expression/identity embeddings, into
any block, or into the decoder. The PPIL attention variants do consume the
context, but through the *realized mask rate* derived from `diffusion_mask`,
never through `diffusion_time` itself; the Performer consumes neither. The
objective uses the exact supplied time, and time is never inferred from the
observed mask count for loss weighting.

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
For the fixed dimensions in this package, the total parameter count is

```text
N(L) = 22,837,699 + P * L
P = 3,150,848   performer
P = 3,152,456   ppil_attention
P = 5,251,584   ppil_ffn
P = 5,253,192   ppil_full
```

where `L` is the number of blocks. For example, `performer` at `L=6` gives
41,742,787 parameters, of which 19,514,947 are trainable; the 19,295x1,152
Geneformer table is frozen. Random-feature projection matrices and the PPI
tables are fixed buffers and are not included in `P`.

`ppil_attention` adds only its 1,608-parameter prior gate per layer
(`16->64->8` with biases): the PPI term itself enters through a wider fixed
random projection, not through new weights. The routed variants add 2,100,736
parameters per layer, because four experts of width `2d` replace one dense
network of width `4d` while per-token compute is unchanged under top-2 routing.

## Architecture identifier

The identifier is three `|`-separated segments with the backbone first:

```text
ppil_full*6|masked-expression-diffusion-v2|hurdle-truncated-normal+inverse-t-nll
```

The backbone segment names the block variant and how many layers of it exist,
and `MaskedDiffusionModelConfig.architecture_version` derives it rather than
storing it, so a configuration cannot carry an identifier that contradicts it.
Nothing compares against a single frozen literal: identifiers are parsed
(`src/models/architecture.py`), and the backbone segment is checked against the
blocks that were actually constructed, both when a backbone is built and after
a checkpoint is loaded.

A stack is always L layers of one variant. Mixing block types was a possibility
an earlier design left open; it has been dropped, and the identifier grammar was
narrowed to match, so what an identifier can express and what the builder can
produce are the same set. A hand-assembled mixed stack is rejected rather than
described.

Whether a checkpoint's weights fit this code is therefore answered in layers:
the format version, the parsed identifier, the identifier against the stored
configuration, the constructed backbone against that identifier, and finally
the strict state-dict load. `sample_masked_diffusion.py` also accepts
`--expect-backbone-variant` so a caller can assert which of the four models it
means to sample.

The checkpoint container is format 4. Its `model_config` holds `backbone`,
`backbone_variant` and `ppi` beside the unchanged component sections;
`architecture_version` is not stored there, because it is derived. A top-level
`ppi_asset_contract` records both artifacts' SHA-256 alongside the routing
shape.

Only the current format is supported. Checkpoints written by earlier revisions
are rejected by the version check rather than migrated, and no migration path
is maintained: the project retrains instead of carrying old weights forward.

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
Training checkpoint v4 contains Python/NumPy RNG objects, so the CLI requires
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

The Performer does not numerically consume `diffusion_time`; its state changes
across reverse steps only through the monotonically shrinking mask and newly
visible expression values. That was an explicit limitation of the v2 baseline
rather than an omission in the sampler API, and the PPIL attention variants
address it: their prior gate reads the realized mask rate, which is visible at
every reverse step, so their behaviour does vary along the trajectory.

## Runtime dependencies

The implementation requires PyTorch and `safetensors`; contract tests use
pytest. The repository intentionally does not pin a CUDA-specific PyTorch wheel
because the execution environment must select a build compatible with its
driver.
