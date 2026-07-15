# SAGE + Skip-Softmax — API Design & Test Plan

## Part 1 — API Design

*How the user uses the trtllm-gen backend.*

### 1. trtllm-gen backend = BF16 dense (baseline)

```bash
vllm-omni serve <model> --diffusion-attention-backend trtllm-gen
# env equivalent:
export DIFFUSION_ATTENTION_BACKEND=trtllm-gen
```

= BF16 dense baseline. SAGE / Skip opt-in on top.

### 2. Add FP8-SAGE

```bash
vllm-omni serve <model> --diffusion-attention-backend trtllm-gen --trtllm-gen-sage
# env: TRTLLM_GEN_SAGE=1
```

FP8-SAGE (no calibration).

### 3. Add Skip-Softmax (needs a calibrated checkpoint)

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend trtllm-gen --trtllm-gen-sage \
  --trtllm-gen-target-sparsity 0.5 \             # skip amount; omit = no skip
  --trtllm-gen-disabled-until-timestep 0.6       # keep initial (noisy) denoise steps dense
```

Needs a ModelOpt-calibrated checkpoint. No calibrated checkpoint → skip stays off (dense).

### 4. vLLM-Omni knobs

```python
@dataclass
class TrtllmGenConfig:                 # mirrors TRT-LLM SkipSoftmaxAttentionConfig, DiT-flavored
    sage: bool = False                 # FP8-SAGE
    target_sparsity: float | None = None        # None -> no skip; needs calibrated checkpoint
    disabled_until_timestep: float = 0.0        # keep initial (noisy) steps dense; per-model preset
```

| CLI flag | default | meaning |
|----------|:-------:|---------|
| `--diffusion-attention-backend trtllm-gen` | — | BF16 dense |
| `--trtllm-gen-sage` | off | FP8-SAGE |
| `--trtllm-gen-target-sparsity <float>` | unset | skip amount (needs calibrated ckpt) |
| `--trtllm-gen-disabled-until-timestep <float>` | preset | keep initial steps dense (t 1→0) |

Env var = flag in UPPER_SNAKE (e.g. `TRTLLM_GEN_SAGE=1`).

**Maps to upstream:**

| vLLM-Omni | TRT-LLM / ModelOpt | FlashInfer |
|-----------|--------------------|------------|
| `sage` | — | `sage_block_sizes` + `bmm1/bmm2_scale` |
| `target_sparsity` | `threshold_scale_factor` (via calibration) | `skip_softmax_threshold_scale_factor` |
| `disabled_until_timestep` | — (DiT-only) | host gate |

Versions: **FlashInfer 0.6.14 · ModelOpt 0.45.0** (runtime deps); TRT-LLM = naming reference (blog16, no runtime dep).

---

## Part 2 — Testing Plan

Hardware: **SM103 (B300)**. Versions: **vLLM-Omni v0.25.0 · FlashInfer 0.6.14 · ModelOpt 0.45.0**.

### Models

| Model | HF checkpoint | params | shape |
|-------|---------------|--------|-------|
| Wan 2.2 A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B | 720×1080, 81f, 50 steps |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | 720p |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | 64B | 720p |

---

### ModelOpt skip calibration

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

**Fitted curve** — `factor = a·exp(b·target_sparsity)` per model. Steps 2–3 fix one `target_sparsity` → one `factor`.

| Model | a | b | run target_sparsity | → factor |
|-------|:-:|:-:|:-------------------:|:--------:|
| Wan 2.2 |  |  |  |  |
| Hunyuan 1.5 |  |  |  |  |
| Cosmos 3 |  |  |  |  |

---

### Step 1 — SAGE (FP8)

FP8-SAGE only, no calibration. trtllm-gen vs current backend: LPIPS parity + speedup.

| Model | config | LPIPS ↓ | speedup | notes |
|-------|--------|:-------:|:-------:|-------|
| Wan 2.2 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Wan 2.2 | FP8-SAGE         |         |        |  |
| Hunyuan 1.5 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Hunyuan 1.5 | FP8-SAGE     |         |        |  |
| Cosmos 3 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Cosmos 3 | FP8-SAGE        |         |        |  |

At BF16 dense: **trtllm-gen == current backend** (LPIPS ≈ 0).

---

### Step 2 — Skip-Softmax

Skip on BF16 attention (no SAGE). Fixed `target_sparsity` (→ factor); sweep **D = `disabled_until_timestep`**.

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

### Step 3 — SAGE + Skip

FP8-SAGE + Skip. Same fixed `target_sparsity`; sweep **D = `disabled_until_timestep`**.

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

### Step 4 — cross-backend comparison

Fix one model + shape + seed, swap only the attention backend. Speedup vs SDPA (= BF16 dense). Run per model (Wan / Hunyuan / Cosmos).

| Backend | attention | LPIPS ↓ | speedup | notes |
|---------|-----------|:-------:|:-------:|-------|
| TORCH_SDPA                   | dense (ref)   | 0 (ref) | 1.00× | reference |
| SAGE_ATTN (SageAttention 2) | FP8-SAGE      |         |        |  |
| **trtllm-gen**              | FP8-SAGE      |         |        |  |
| **trtllm-gen**              | FP8-SAGE+Skip |         |        |  |

