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
  --trtllm-gen-skip-sparsity 0.5 \        # target_sparsity / fidelity D; omit = no skip
  --trtllm-gen-skip-disabled-until 0.6    # normalized timestep [0,1]; early steps stay dense
```

The `a,b` curve is read from the checkpoint (ModelOpt-calibrated); the user only picks sparsity. No calibrated checkpoint → skip stays off (dense).

### 4. All knobs

| CLI flag | env var | default | meaning |
|----------|---------|:-------:|---------|
| `--diffusion-attention-backend trtllm-gen` | `DIFFUSION_ATTENTION_BACKEND=trtllm-gen` | — | select backend (**BF16 dense**) |
| `--trtllm-gen-sage` | `TRTLLM_GEN_SAGE=1` | **off** | turn on FP8-SAGE |
| `--trtllm-gen-skip-sparsity <float>` | `TRTLLM_GEN_SKIP_SPARSITY` | unset (no skip) | target_sparsity / D |
| `--trtllm-gen-skip-disabled-until <float>` | `TRTLLM_GEN_SKIP_DISABLED_UNTIL` | per-model preset | timestep gate [0,1] |

**Names map to upstream (reuse TRT-LLM / FlashInfer, don't invent):**

| vLLM-Omni flag | TRT-LLM / ModelOpt | FlashInfer kernel arg |
|----------------|--------------------|-----------------------|
| `--trtllm-gen-sage` | `flash_skip_softmax` sibling — SAGE fp8 | `sage_block_sizes=(q,k,v)` + `bmm1_scale`/`bmm2_scale` (float32 per-block) |
| `--trtllm-gen-skip-sparsity` | `sparse_attention_config: {algorithm: skip_softmax, target_sparsity}` | `skip_softmax_threshold_scale_factor = a·exp(b·sparsity)` (threshold = factor / seqlen) |
| `--trtllm-gen-skip-disabled-until` | — (DiT runtime; not in TRT-LLM's LLM config) | host-side gate, no kernel arg |

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

Fix one model + shape + seed, swap only the attention backend. Speedup vs SDPA. Run per model (Wan / Hunyuan / Cosmos).

| Backend | attention | LPIPS ↓ | speedup | notes |
|---------|-----------|:-------:|:-------:|-------|
| TORCH_SDPA                   | dense (ref)   | 0 (ref) | 1.00× | reference |
| SAGE_ATTN (SageAttention 2) | FP8-SAGE      |         |        |  |
| **TRTLLM**                  | FP8-SAGE      |         |        |  |
| **TRTLLM**                  | FP8-SAGE+Skip |         |        |  |

---

## Notes

- SAGE = no calibration; Skip = per-model calibration.
- LPIPS measured vs **BF16 dense**; perf on the **target Blackwell SKU under real concurrency**.
