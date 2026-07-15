# SAGE + Skip-Softmax — API Design & Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip → (4) cross-backend.**

---

## API design

**User config**

```yaml
attention_backend: trtllm          # FlashInfer trtllm-gen path
sage: true                         # FP8-SAGE attention, block (1,4,0,1)
skip_softmax:                      # optional
  target_sparsity: 0.5             # or fidelity D → factor
  disabled_until_timestep: 0.6     # normalized [0,1]
```

**Config dataclasses**

```python
@dataclass
class SkipSoftmaxConfig:
    target_sparsity: float | None = None    # → factor from calibration
    factor: float | None = None             # manual override (skip calib)
    disabled_until_timestep: float = 0.0

@dataclass
class TrtllmAttnConfig:
    sage: bool = True
    sage_block: tuple = (1, 4, 0, 1)
    skip_softmax: SkipSoftmaxConfig | None = None
# AttentionMetadata += skip_softmax_factor: float | None
```

**Kernel call** (FlashInfer)

```python
flashinfer.prefill.trtllm_ragged_attention_deepseek(
    ...,
    skip_softmax_threshold_scale_factor = factor,   # threshold = factor / seqlen
    sage_attn_sfs = (q_sf, k_sf, None, v_sf),
    num_elts_per_sage_attn_blk = (1, 4, 0, 1),      # (0,0,0,0) = SAGE off
    bmm1_scale = scale_q * scale_k * sm_scale,
    bmm2_scale = scale_v,
    backend = "trtllm-gen",
)
```

**Calibration source:** per-model `config.json → sparse_attention_config` (`factor = a·exp(b·target_sparsity)`, per `config_group`, `ignore` layers kept dense).

**Fallback:** SM < 100 / head_dim ≠ 128 / GQA / no calibration → flash-attn.

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
