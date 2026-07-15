# SAGE + Skip-Softmax — Test Plan

Backend: **trtllm-gen** (FlashInfer). Primary metric: **LPIPS ↓** vs BF16 dense. Speedup vs BF16 dense.
Order: **(1) SAGE → (2) Skip → (3) SAGE + Skip.**

## Models

| Model | HF checkpoint | params | shape |
|-------|---------------|--------|-------|
| Wan 2.2 A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B | 720×1080, 81f, 50 steps |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | 720p |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | 64B | 720p (GQA — see notes) |

---

## ModelOpt skip calibration

Skip needs a **per-model** D → `threshold_scale_factor` curve. Calibration sweeps `target_sparsity` on a fine grid, measures achieved sparsity + LPIPS at each point, and fits `factor = a·exp(b·target_sparsity)` **per `config_group`** (written into checkpoint `config.json`, `ignore` layers kept dense). The 3 D points used in Steps 2–3 are operating points read off this curve. (SAGE needs no calibration → Step 1 can start without this.)

**Setup** (per model)

| Model | calib prompts / dataset | # samples | res / frames | timesteps | config_groups | ignore layers (dense) |
|-------|-------------------------|:---------:|--------------|-----------|---------------|-----------------------|
| Wan 2.2 |  |  | 720×1080 / 81 |  |  |  |
| Hunyuan 1.5 |  |  |  |  |  |  |
| Cosmos 3 |  |  |  |  |  | GQA — separate |

**Sweep** (per model; fine grid to fit the curve)

*Model: ______ · config_group: ______*

| target_sparsity | threshold factor | achieved sparsity | LPIPS ↓ | fidelity D |
|:---------------:|:----------------:|:-----------------:|:-------:|:----------:|
| 0.10 |  |  |  |  |
| 0.20 |  |  |  |  |
| 0.30 |  |  |  |  |
| 0.40 |  |  |  |  |
| 0.50 |  |  |  |  |
| 0.60 |  |  |  |  |
| 0.70 |  |  |  |  |
| 0.80 |  |  |  |  |
| 0.90 |  |  |  |  |

**Fitted coefficients + operating points**

| Model | config_group | a | b | factor @ D1.00 | factor @ D0.97 | factor @ D0.94 |
|-------|--------------|:-:|:-:|:--------------:|:--------------:|:--------------:|
| Wan 2.2 |  |  |  |  |  |  |
| Hunyuan 1.5 |  |  |  |  |  |  |
| Cosmos 3 |  |  |  |  |  |  |

---

## Step 1 — SAGE (FP8)

Just FP8-SAGE attention. **No calibration** (SAGE is dynamic). Compare trtllm-gen vs current backend; check LPIPS parity + speedup.

| Model | config | LPIPS ↓ | speedup | notes |
|-------|--------|:-------:|:-------:|-------|
| Wan 2.2 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Wan 2.2 | FP8-SAGE         |         |        |  |
| Hunyuan 1.5 | BF16 dense (ref) | 0 (ref) | 1.00× |  |
| Hunyuan 1.5 | FP8-SAGE     |         |        |  |
| Cosmos 3 | BF16 dense (ref) | 0 (ref) | 1.00× | GQA |
| Cosmos 3 | FP8-SAGE        |         |        | GQA — record if it falls back |

Also confirm: at BF16 dense, **trtgen output == current backend** (LPIPS ≈ 0).

---

## Step 2 — Skip-Softmax

Skip on BF16 attention (isolate the skip effect). Uses the calibrated D → factor above. D = fidelity (1.00 ≈ no skip → 0.94 = most aggressive).

| Model | D | LPIPS ↓ | speedup | achieved sparsity |
|-------|:---:|:-------:|:-------:|:-----------------:|
| Wan 2.2 | 1.00 | ~0 | ~1.00× | ~0 |
| Wan 2.2 | 0.97 |    |        |    |
| Wan 2.2 | 0.94 |    |        |    |
| Hunyuan 1.5 | 1.00 | ~0 | ~1.00× | ~0 |
| Hunyuan 1.5 | 0.97 |    |        |    |
| Hunyuan 1.5 | 0.94 |    |        |    |
| Cosmos 3 | 1.00 | ~0 | ~1.00× | ~0 |
| Cosmos 3 | 0.97 |    |        |    |
| Cosmos 3 | 0.94 |    |        |    |

---

## Step 3 — SAGE + Skip

Both together: FP8-SAGE + Skip at each D.

| Model | D | LPIPS ↓ | speedup | achieved sparsity |
|-------|:---:|:-------:|:-------:|:-----------------:|
| Wan 2.2 | 1.00 |    |        |    |
| Wan 2.2 | 0.97 |    |        |    |
| Wan 2.2 | 0.94 |    |        |    |
| Hunyuan 1.5 | 1.00 |    |        |    |
| Hunyuan 1.5 | 0.97 |    |        |    |
| Hunyuan 1.5 | 0.94 |    |        |    |
| Cosmos 3 | 1.00 |    |        |    |
| Cosmos 3 | 0.97 |    |        |    |
| Cosmos 3 | 0.94 |    |        |    |

## Notes

- **SAGE = no calibration; Skip = per-model calibration** (per `config_group`, per resolution).
- Measure LPIPS vs **BF16 dense**; measure perf on the **target Blackwell SKU under real concurrency**.
- **Cosmos 3 is GQA** → outside dense-MHA validation; calibrate + validate separately.
