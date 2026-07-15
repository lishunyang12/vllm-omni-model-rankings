# Skip-Softmax + SAGE Attention Backend — Design & Test Plan

*New diffusion attention backend for vLLM-Omni. First algorithms integrated: **Skip-Softmax** (BLASST, arXiv 2512.12087) and **SAGE** (FP8 attention). Verified against FlashInfer 0.6.12 + ModelOpt/TensorRT-LLM sparse-attention docs. Target models: Wan 2.2, Hunyuan Video, Cosmos 3.*

---

# Part A — Design Plan

## A.1 Goal & motivation

Add a new `trtllm` diffusion attention backend (FlashInfer trtllm-gen path) and land two acceleration features on it:

- **SAGE** — FP8 (E4M3) attention with dynamic per-block scales, block `(Q,K,P,V)=(1,4,0,1)`.
- **Skip-Softmax** — dynamic block sparsity; skips P·V + V-load when a running-max predicate says a K-block can't contribute. Speedup ≈ `1 + 0.9·achieved_sparsity`, capped ~2× (QK^T still paid).

Why a new backend and why now: on **B300 the existing `SAGE_ATTN` / `SAGE_ATTN_3` backends do not run** (SAGE_ATTN = SM80–90 lib; SAGE_ATTN_3 = SM120-focused, no GQA). **trtllm-gen is the only SAGE-capable path on Blackwell**, and it also carries Skip-Softmax. One backend, two stacking features.

## A.2 Where it plugs in

```
DiffusionAttentionBackendEnum  +=  TRTLLM
  -> vllm_omni/diffusion/attention/backends/trtllm_attn.py
     -> flashinfer.prefill.trtllm_ragged_attention_deepseek(...)   # trtllm-gen backend
```

The three knobs are **orthogonal** and stack:

| Axis | Quantizes / does | Source |
|------|------------------|--------|
| GEMM quant | linear/proj **weights** → FP8 / NVFP4 | existing ModelOpt loader (#5076/#5087) |
| SAGE | **attention** matmuls → FP8 E4M3 | runtime/dynamic, no calibration |
| Skip-Softmax | attention **block sparsity** | calibrated threshold in checkpoint config.json |

(There is no NVFP4 attention here — NVFP4 is a weight axis; SAGE attention is always FP8.)

## A.3 User config (mirror TensorRT-LLM)

```yaml
attention_backend: trtllm
sparse_attention_config:
  algorithm: skip_softmax
  target_sparsity: 0.5              # or threshold_scale_factor: 5000.0
  disabled_until_timestep: 0.6      # normalized [0,1], early steps stay dense
trtllm:
  sage: true                        # FP8-SAGE, block (1,4,0,1)
```

```python
@dataclass
class SkipSoftmaxConfig:
    sparsity: float | None = None            # target; -> factor via calibration
    calibration: str | None = None           # formula/coeffs from checkpoint
    disabled_until_timestep: float = 0.0      # normalized denoise phase gate
    theta: float | None = None               # manual factor override (skip calib)

@dataclass
class TrtllmAttnConfig:
    sage: bool = True
    sage_block: tuple = (1, 4, 0, 1)
    skip_softmax: SkipSoftmaxConfig | None = None

# AttentionMetadata gains:  skip_softmax_factor: float | None = None
```

## A.4 Kernel call (FlashInfer)

```python
flashinfer.prefill.trtllm_ragged_attention_deepseek(
    ...,
    skip_softmax_threshold_scale_factor = factor,   # threshold = factor / seqlen
    sage_attn_sfs              = (q_sf, k_sf, None, v_sf),   # dynamic FP8 block scales
    num_elts_per_sage_attn_blk = (1, 4, 0, 1),      # (0,0,0,0) = SAGE off
    bmm1_scale = scale_q * scale_k * sm_scale,      # fused by caller
    bmm2_scale = scale_v,
    backend    = "trtllm-gen",                       # cute-dsl backend has NO SAGE
)
```

## A.5 Calibration (checkpoint config.json, from ModelOpt)

```json
"sparse_attention_config": {
  "config_groups": {
    "group_0": {
      "algorithm": "skip_softmax",
      "threshold_scale_factor": {"formula": "a * exp(b * target_sparsity)",
                                 "coefficients": {"a": 1000.0, "b": 5.0}},
      "target_sparsity": 0.5,
      "disabled_until_timestep": 0.8,
      "ignore": ["blocks.0.attn1"]
    }
  }
}
```

- `target_sparsity` → factor via `a*exp(b*target_sparsity)` (numexpr), per `config_group`.
- `ignore` = fnmatch layer patterns that stay dense.
- Kernel divides factor by seqlen at runtime.

## A.6 DiT (fixed L) vs LLM (varied L)

Kernel does `threshold = factor / seqlen`. The `/seqlen` is the only thing that differs.

**DiT — L is FIXED per generation** (known from H, W, frames, VAE downsample, patch):

```python
L = (H//8//patch) * (W//8//patch) * frames    # exact, known before denoise
factor = a * exp(b * target_sparsity)          # per-group a,b from checkpoint
# L fixed -> threshold = factor/L constant across steps;
# compute factor ONCE per generation; only the timestep gate toggles skip.
```

- `/seqlen` is redundant for DiT but lets one calibration transfer across resolutions.
- Calibrate per `(resolution, frames)` bucket — safe sparsity is resolution/content dependent.
- CUDA graphs: key by `(L, skip_enabled_phase)` — static shapes.
- **LLM** (future): L varies → rely on `/seqlen`, recompute threshold per step.

## A.7 Fallback / hardware gate

Fall back to flash-attn (never guess) when outside the validated envelope:

- SM < 100 (non-Blackwell)
- head_dim ≠ 128
- GQA (q_heads ≠ kv_heads)
- no calibration **and** no manual `theta` → skip disabled (dense)
- `sparsity = 0` → dense

## A.8 Work items

1. Register `TRTLLM` backend = FlashInfer trtllm-gen path.
2. Read `sparse_attention_config` from checkpoint config.json; eval formula per group; skip `ignore` layers.
3. Compute L from request; resolve factor **once per generation**; gate skip by normalized timestep.
4. Key CUDA graphs per `(L, phase)`.
5. Hardware/shape gate → else flash-attn fallback.
6. Wire SAGE FP8 scales (runtime); GEMM FP8/NVFP4 via existing ModelOpt loader.
7. Add E2E quality gate (LPIPS/PSNR vs dense).

---

# Part B — Test Plan

## B.1 Levels & ownership

| Level | Owner | Status |
|-------|-------|--------|
| Kernel numerical/perf parity (FI vs TRT, one shape, B300) | NVIDIA DevTech | done (with caveats, B.5) |
| **Integration correctness** (fallbacks, config, calibration, CUDA graphs) | **us** | to do |
| **E2E quality + perf** (per model, on target SKU) | **us** | to do |

## B.2 Coverage scope — operating point, not the full research grid

The deck sweeps `{BF16,FP8,NVFP4} × {dense,SAGE,SAGE+Skip} × sparsity` on 3 models. We do **not** reproduce that whole grid. Per model, cover:

- **BF16 dense** — quality + perf reference.
- The **quant we ship** (FP8 or NVFP4) × **{SAGE, SAGE+Skip}** — the operating points.
- A **sparsity / disabled_until_timestep sweep** to pick the per-model default (Pareto knee).
- An **E2E quality gate** on the shipped config.

## B.3 Metrics

- **E2E latency (s)** and **speedup vs BF16-dense** — the headline.
- **achieved sparsity** — runtime-measured fraction of blocks actually skipped (content-dependent; ≠ target). Needs a counter.
- **attention kernel time** — isolates the attention win from the rest.
- **LPIPS ↓ / PSNR / VBench** — quality vs BF16-dense output.
- **peak VRAM**.

## B.4 Test matrix → fillable tables

Results tables live in `scripts/skip_softmax_test_results.md`:

- **Table 1** — shape/envelope probe (is each model in the D128 + dense-MHA + FP8 + SM103 case?).
- **Table 2** — E2E per model (quant × attention mode → latency/speedup/achieved-sparsity/LPIPS/VRAM/pass).
- **Table 3** — sparsity–fidelity sweep per model → pick default.
- **Table 4** — fallback / correctness gates (10 cases).

## B.5 Pass/fail gates

- **Quality:** LPIPS vs BF16-dense ≤ per-model budget (set from BF16 dense ref); no obvious artifacts on eval prompts.
- **Perf:** net E2E speedup ≥ 1.0× **on the target Blackwell SKU under realistic concurrency** — not the isolated B300 kernel number.
- **Correctness:** all Table-4 gates PASS; `None` factor == `0.0` factor; unsupported cases fall back cleanly with a one-line log.

## B.6 Caveats carried from the kernel validation report

- Kernel parity feeds byte-identical FP8 → the runtime **bf16→fp8 SAGE quant step is untested**; our E2E gate must cover it.
- Reference is TRT-LLM RC20, **not a full-precision oracle** — proves FI≈TRT, not correctness; a shared bug passes.
- At the two deployment thetas, FI/TRT were **~10–13% slower than dense**; only the quality-unsafe theta was faster. Re-measure net win per model.
- The 4-GPU concurrent run **failed**; only isolated-GPU passed. Production is concurrent → test concurrent.
- Narrow envelope: one shape, SM103 only, single seed. **B200 / other shapes need re-validation.**
- `cute-dsl` backend silently asserts SAGE off — only `trtllm-gen` has SAGE. Guard any fallback path.
