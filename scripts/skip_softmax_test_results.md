# SAGE + Skip-Softmax — API Design & Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip → (4) cross-backend.**

---

## API design — how the user uses the trtllm-gen backend

### 1. Just FP8-SAGE (minimal — one flag)

```bash
vllm-omni serve <model> --diffusion-attention-backend trtllm-gen
# env equivalent:
export DIFFUSION_ATTENTION_BACKEND=trtllm-gen
```

That's the whole thing — selecting `trtllm-gen` gives **FP8-SAGE on by default** (runtime, no calibration, no checkpoint change). To turn SAGE off: `--trtllm-gen-sage false`.

### 2. Add Skip-Softmax (needs a calibrated checkpoint)

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend trtllm-gen \
  --trtllm-gen-skip-sparsity 0.5 \        # target_sparsity / fidelity D; omit = no skip
  --trtllm-gen-skip-disabled-until 0.6    # normalized timestep [0,1]; early steps stay dense
# env equivalent:
export DIFFUSION_ATTENTION_BACKEND=trtllm-gen
export TRTLLM_GEN_SKIP_SPARSITY=0.5
export TRTLLM_GEN_SKIP_DISABLED_UNTIL=0.6
```

The `a,b` curve is read from the checkpoint (ModelOpt-calibrated); the user only picks sparsity. No calibrated checkpoint → skip stays off (dense).

### 3. All knobs

| CLI flag | env var | default | meaning |
|----------|---------|:-------:|---------|
| `--diffusion-attention-backend trtllm-gen` | `DIFFUSION_ATTENTION_BACKEND=trtllm-gen` | — | select backend |
| `--trtllm-gen-sage {true,false}` | `TRTLLM_GEN_SAGE={1,0}` | `true` | FP8-SAGE on/off |
| `--trtllm-gen-skip-sparsity <float>` | `TRTLLM_GEN_SKIP_SPARSITY` | unset (no skip) | target_sparsity / D |
| `--trtllm-gen-skip-disabled-until <float>` | `TRTLLM_GEN_SKIP_DISABLED_UNTIL` | per-model preset | timestep gate [0,1] |

No dedicated YAML — these are standard CLI flags / env vars (also settable via vLLM's generic `--config *.yaml`). Enum member + hyphen-normalize:

```python
# registry.py: TRTLLM_GEN = "...backends.trtllm_attn.TrtllmAttentionBackend"
# cuda/platform.py: backend_upper = selected_backend.upper().replace("-", "_")  # trtllm-gen -> TRTLLM_GEN
# gate (mirrors SAGE_ATTN_3): Blackwell SM>=100 + flashinfer + head_dim=128 dense MHA, else fall back.
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
