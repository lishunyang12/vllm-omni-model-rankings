# SAGE + Skip-Softmax — API Design & Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip → (4) cross-backend.**

---

## API design

### Who controls what

| Knob | Set by | Where |
|------|--------|-------|
| FP8 / NVFP4 **GEMM weights** | **ModelOpt** (offline quant) | checkpoint weights (loader #5076/#5087) |
| **Skip threshold curve** `a, b` (D→factor) | **ModelOpt** (skip calibration) | checkpoint `config.json → sparse_attention_config` |
| **config_groups + `ignore`** (which layers stay dense) | **ModelOpt** | same `config.json` |
| Backend = `trtllm` | **vLLM-Omni** (user) | engine args |
| **SAGE on/off** (FP8 attn, runtime, no calib) | **vLLM-Omni** (user) | engine args |
| **target_sparsity / D** (pick operating point) | **vLLM-Omni** (user) | engine args → `factor = a·exp(b·D)` |
| **disabled_until_timestep** gate | **vLLM-Omni** (denoise loop) | runtime |
| L (seqlen), `factor` once/gen, CUDA-graph keys | **vLLM-Omni** | runtime |
| Fallback (SM<100 / head_dim≠128 / GQA) | **vLLM-Omni** | runtime |

> **ModelOpt = offline, baked into the checkpoint** (weights + the `a,b` curve + ignore list). **vLLM-Omni = runtime**: reads that curve, turns SAGE on/off, picks D, computes the factor, gates by timestep, and calls the kernel. vLLM-Omni never re-fits `a,b`; ModelOpt never runs at inference.

### 1. ModelOpt output — checkpoint `config.json` (vLLM-Omni only reads this)

```json
"sparse_attention_config": {
  "config_groups": {
    "group_0": {
      "method": "flash_skip_softmax",
      "threshold_scale_factor": {"formula": "a*exp(b*target_sparsity)", "a": 1000.0, "b": 5.0},
      "ignore": ["blocks.0.attn1"]
    }
  }
}
```

### 2. vLLM-Omni user config

```yaml
attention_backend: trtllm          # FlashInfer trtllm-gen path
sage: true                         # FP8-SAGE attention (runtime, no calibration)
skip_softmax:                      # optional; needs the ModelOpt curve above
  target_sparsity: 0.5             # or fidelity D
  disabled_until_timestep: 0.6     # normalized [0,1]; early steps stay dense
```

```python
@dataclass
class SkipSoftmaxConfig:
    target_sparsity: float | None = None    # user picks; -> factor via a,b from config.json
    factor: float | None = None             # manual override (bypass calibration)
    disabled_until_timestep: float = 0.0

@dataclass
class TrtllmAttnConfig:
    sage: bool = True                        # FP8-SAGE
    skip_softmax: SkipSoftmaxConfig | None = None
```

### 3. vLLM-Omni runtime → kernel

```python
# once per generation (L fixed for a DiT):
L = tokens_from(H, W, frames, vae_downsample, patch)
a, b   = config_json["sparse_attention_config"][group]["threshold_scale_factor"]  # ModelOpt
factor = a * math.exp(b * target_sparsity)        # vLLM-Omni; None -> no skip
# per denoise step: apply skip only when timestep >= disabled_until_timestep

flashinfer.prefill.trtllm_ragged_attention_deepseek(
    q_fp8, k_fp8, v_fp8, workspace, seq_lens, max_q_len, max_kv_len,
    bmm1_scale = sq * sk * sm_scale,   # float32 tensor = SAGE per-block scales (scalar if plain FP8)
    bmm2_scale = sv,                   # float32 tensor
    ...,
    is_causal = False,                 # DiT is bidirectional
    skip_softmax_threshold_scale_factor = factor,   # threshold = factor / L; omit -> dense
)
```

**Fallback (vLLM-Omni):** SM < 100 / head_dim ≠ 128 / GQA / no calibration & no manual factor → flash-attn.

---

## Models

| Model | HF checkpoint | params | shape |
|-------|---------------|--------|-------|
| Wan 2.2 A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B | 720×1080, 81f, 50 steps |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | 720p |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | 64B | 720p |

---

## ModelOpt skip calibration

Per-model D → `threshold_scale_factor` curve. Sweep `target_sparsity`, fit `factor = a·exp(b·target_sparsity)` per `config_group` (written to `config.json`). Steps 2–3 read D points off this curve. (SAGE needs no calibration.)

**Setup** (per model)

| Model | calib prompts | # samples | res / frames | config_groups | ignore layers |
|-------|---------------|:---------:|--------------|---------------|---------------|
| Wan 2.2 |  |  | 720×1080 / 81 |  |  |
| Hunyuan 1.5 |  |  |  |  |  |
| Cosmos 3 |  |  |  |  |  |

**Sweep** (per model — fit the curve)  ·  *Model: ______*

| target_sparsity | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|-----------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| threshold factor |  |  |  |  |  |  |  |  |  |
| achieved sparsity |  |  |  |  |  |  |  |  |  |
| LPIPS ↓ |  |  |  |  |  |  |  |  |  |

**Fitted coefficients + operating points**

| Model | a | b | factor @ D1.00 | factor @ D0.97 | factor @ D0.94 |
|-------|:-:|:-:|:--------------:|:--------------:|:--------------:|
| Wan 2.2 |  |  |  |  |  |
| Hunyuan 1.5 |  |  |  |  |  |
| Cosmos 3 |  |  |  |  |  |

---

## Step 1 — SAGE (FP8)

FP8-SAGE only, no calibration. trtllm-gen vs current backend: LPIPS parity + speedup.

| Model | config | LPIPS ↓ | speedup | notes |
|-------|--------|:-------:|:-------:|-------|
| Wan 2.2 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Wan 2.2 | FP8-SAGE         |         |        |  |
| Hunyuan 1.5 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Hunyuan 1.5 | FP8-SAGE     |         |        |  |
| Cosmos 3 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Cosmos 3 | FP8-SAGE        |         |        |  |

At BF16 dense: **trtgen == current backend** (LPIPS ≈ 0).

---

## Step 2 — Skip-Softmax

Skip on BF16 attention. D = fidelity (1.00 ≈ no skip → 0.94 = most aggressive).

| Model | D | LPIPS ↓ | speedup |
|-------|:---:|:-------:|:-------:|
| Wan 2.2 | 1.00 | ~0 | ~1.00× |
| Wan 2.2 | 0.97 |    |        |
| Wan 2.2 | 0.94 |    |        |
| Hunyuan 1.5 | 1.00 | ~0 | ~1.00× |
| Hunyuan 1.5 | 0.97 |    |        |
| Hunyuan 1.5 | 0.94 |    |        |
| Cosmos 3 | 1.00 | ~0 | ~1.00× |
| Cosmos 3 | 0.97 |    |        |
| Cosmos 3 | 0.94 |    |        |

---

## Step 3 — SAGE + Skip

FP8-SAGE + Skip at each D.

| Model | D | LPIPS ↓ | speedup |
|-------|:---:|:-------:|:-------:|
| Wan 2.2 | 1.00 |    |        |
| Wan 2.2 | 0.97 |    |        |
| Wan 2.2 | 0.94 |    |        |
| Hunyuan 1.5 | 1.00 |    |        |
| Hunyuan 1.5 | 0.97 |    |        |
| Hunyuan 1.5 | 0.94 |    |        |
| Cosmos 3 | 1.00 |    |        |
| Cosmos 3 | 0.97 |    |        |
| Cosmos 3 | 0.94 |    |        |

---

## Step 4 — cross-backend comparison

Fix one model + shape + seed, swap only the attention backend. Speedup vs current backend. Run per model (Wan / Hunyuan / Cosmos).

| Backend | attention | LPIPS ↓ | speedup | notes |
|---------|-----------|:-------:|:-------:|-------|
| current (default) | dense     | 0 (ref) | 1.00× |  |
| FLASH_ATTN_3      | dense     |         |        |  |
| CUDNN_ATTN        | dense     |         |        |  |
| FLASHINFER        | dense     |         |        |  |
| **TRTLLM**        | FP8-SAGE      |     |        |  |
| **TRTLLM**        | FP8-SAGE+Skip |     |        |  |

---

## Notes

- SAGE = no calibration; Skip = per-model calibration.
- LPIPS measured vs **BF16 dense**; perf on the **target Blackwell SKU under real concurrency**.
