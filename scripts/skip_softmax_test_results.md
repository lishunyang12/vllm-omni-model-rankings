# SAGE + Skip-Softmax — API Design & Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip → (4) cross-backend.**

---

# Part 1 — API Design

*How the user uses the trtllm-gen backend.*

### 1. trtllm-gen backend = BF16 dense (baseline)

```bash
vllm-omni serve <model> --diffusion-attention-backend trtllm-gen
# env equivalent:
export DIFFUSION_ATTENTION_BACKEND=trtllm-gen
```

Plain `trtllm-gen` runs **dense BF16 attention through the trtllm-gen kernel** — the baseline / parity backend (same math as SDPA → LPIPS ≈ 0 vs current backend). SAGE and Skip are **opt-in** on top.

### 2. Add FP8-SAGE

```bash
vllm-omni serve <model> --diffusion-attention-backend trtllm-gen --trtllm-gen-sage
# env: TRTLLM_GEN_SAGE=1
```

FP8-SAGE attention (runtime, dynamic, no calibration, no checkpoint change).

### 3. Add Skip-Softmax (needs a calibrated checkpoint)

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend trtllm-gen --trtllm-gen-sage \
  --trtllm-gen-target-sparsity 0.5 \             # skip amount; omit = no skip
  --trtllm-gen-disabled-until-timestep 0.6       # keep initial (noisy) denoise steps dense
```

Needs a ModelOpt-calibrated checkpoint. No calibrated checkpoint → skip stays off (dense).

### 4. vLLM-Omni knobs

Three vLLM-Omni-native knobs on top of the backend:

```python
@dataclass
class TrtllmGenConfig:                 # mirrors TRT-LLM SkipSoftmaxAttentionConfig, DiT-flavored
    sage: bool = False                 # FP8-SAGE
    target_sparsity: float | None = None        # None -> no skip; needs calibrated checkpoint
    disabled_until_timestep: float = 0.0        # keep initial (noisy) steps dense; per-model preset
```

| CLI flag | env var | default | meaning |
|----------|---------|:-------:|---------|
| `--diffusion-attention-backend trtllm-gen` | `DIFFUSION_ATTENTION_BACKEND=trtllm-gen` | — | select backend (**BF16 dense**) |
| `--trtllm-gen-sage` | `TRTLLM_GEN_SAGE=1` | **off** | FP8-SAGE |
| `--trtllm-gen-target-sparsity <float>` | `TRTLLM_GEN_TARGET_SPARSITY` | unset (no skip) | skip amount; needs calibrated ckpt |
| `--trtllm-gen-disabled-until-timestep <float>` | `TRTLLM_GEN_DISABLED_UNTIL_TIMESTEP` | per-model preset | keep initial noisy steps dense (fidelity guard); normalized denoise t 1→0 |

**Maps to upstream (don't invent):**

| vLLM-Omni knob | TRT-LLM / ModelOpt | FlashInfer kernel arg |
|----------------|--------------------|-----------------------|
| `sage` | SAGE (separate axis; not in `SkipSoftmaxAttentionConfig`) | `sage_block_sizes=(q,k,v)` + `bmm1_scale`/`bmm2_scale` |
| `target_sparsity` | ModelOpt calibration input → config file maps to `threshold_scale_factor` | `skip_softmax_threshold_scale_factor` (threshold = factor / seqlen) |
| `disabled_until_timestep` | — (DiT-only; not in TRT-LLM's LLM config) | host-side gate, no kernel arg |

---

# Part 2 — Testing Plan

## Models

| Model | HF checkpoint | params | shape |
|-------|---------------|--------|-------|
| Wan 2.2 A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B | 720×1080, 81f, 50 steps |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | 720p |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | 64B | 720p |

---

## ModelOpt skip calibration

Per-model curve. Sweep `target_sparsity`, fit `factor = a·exp(b·target_sparsity)` per `config_group` (written to `config.json`). (SAGE needs no calibration.)

Two independent knobs: **`target_sparsity`** (how much to skip → `factor`, from the curve) and **`D = disabled_until_timestep`** (when skip starts; normalized denoise t 1→0). Steps 2–3 **fix `target_sparsity` and sweep D** = 1.00 / 0.97 / 0.94 (1.00 = skip disabled all denoise = dense sanity anchor → 0.94 = skip starts earliest = most aggressive). Calibration here fits `factor(target_sparsity)` only; D is a runtime gate, not calibrated.

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

## Step 3 — SAGE + Skip

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

## Step 4 — cross-backend comparison

Fix one model + shape + seed, swap only the attention backend. Speedup vs SDPA (= BF16 dense). Run per model (Wan / Hunyuan / Cosmos).

| Backend | attention | LPIPS ↓ | speedup | notes |
|---------|-----------|:-------:|:-------:|-------|
| TORCH_SDPA                   | dense (ref)   | 0 (ref) | 1.00× | reference |
| SAGE_ATTN (SageAttention 2) | FP8-SAGE      |         |        |  |
| **trtllm-gen**              | FP8-SAGE      |         |        |  |
| **trtllm-gen**              | FP8-SAGE+Skip |         |        |  |

---

## Notes

- SAGE = no calibration; Skip = per-model calibration.
- LPIPS measured vs **BF16 dense**; perf on the **target Blackwell SKU under real concurrency**.
