# SAGE + Skip-Softmax — API Design & Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip → (4) cross-backend.**

---

## API design — how the user uses the trtllm-gen backend

### 1. Select the backend (by name, same as every backend)

```bash
# CLI
vllm-omni serve <model> --diffusion-attention-backend TRTLLM_ATTN
# or env
export DIFFUSION_ATTENTION_BACKEND=TRTLLM_ATTN
```

`cuda/platform.py` gates it (mirrors `SAGE_ATTN_3`): needs **Blackwell SM≥100 + flashinfer + head_dim=128 dense MHA** → else falls back to SDPA/flash-attn with a one-line log. New enum member:

```python
# registry.py — DiffusionAttentionBackendEnum
TRTLLM_ATTN = "vllm_omni.diffusion.attention.backends.trtllm_attn.TrtllmAttentionBackend"
```

### 2. SAGE — on by default

Selecting `TRTLLM_ATTN` **already gives FP8-SAGE** (runtime, dynamic, no calibration, no checkpoint change). Nothing else to set.

### 3. Skip-Softmax — opt-in (needs a calibrated checkpoint)

User only picks the operating point; the `a,b` curve is read from the checkpoint (ModelOpt-calibrated). No calibrated checkpoint → skip stays off (dense).

```bash
--trtllm-skip-sparsity 0.5          # target_sparsity / fidelity D
--trtllm-skip-disabled-until 0.6    # normalized timestep [0,1]; early steps dense
```

```python
@dataclass
class TrtllmAttnConfig:
    sage: bool = True                       # FP8-SAGE (default on)
    skip_sparsity: float | None = None      # None -> no skip; else -> factor from checkpoint a,b
    skip_disabled_until_timestep: float = 0.0
```

### 4. What vLLM-Omni does at runtime

- Reads `skip_sparsity` → `factor = a·exp(b·skip_sparsity)` (`a,b` from checkpoint; `None` → dense).
- Computes L once per generation (DiT: L fixed from H,W,frames,VAE,patch).
- Gates skip by denoise timestep (`>= disabled_until_timestep`).
- Keys CUDA graphs per (L, skip on/off).
- Calls `trtllm_ragged_attention_deepseek(..., is_causal=False, skip_softmax_threshold_scale_factor=factor)` — threshold = factor / L.

**Fallback:** SM<100 / head_dim≠128 / GQA / no calibrated checkpoint → flash-attn.

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
