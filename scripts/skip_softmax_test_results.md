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

Skip on BF16 attention (isolate the skip effect). **Needs ModelOpt skip calibration per model** (D → threshold). D = fidelity (1.00 ≈ no skip → 0.94 = most aggressive). Start with Wan.

| Model | D | LPIPS ↓ | speedup | achieved sparsity |
|-------|:---:|:-------:|:-------:|:-----------------:|
| Wan 2.2 | 1.00 | ~0 | ~1.00× | ~0 |
| Wan 2.2 | 0.97 |    |        |    |
| Wan 2.2 | 0.94 |    |        |    |

---

## Step 3 — SAGE + Skip

Both together: FP8-SAGE + Skip at each D.

| Model | D | LPIPS ↓ | speedup | achieved sparsity |
|-------|:---:|:-------:|:-------:|:-----------------:|
| Wan 2.2 | 1.00 |    |        |    |
| Wan 2.2 | 0.97 |    |        |    |
| Wan 2.2 | 0.94 |    |        |    |

---

## Fallback gates

| Case | Expected | PASS? |
|------|----------|:-----:|
| unsupported SM / head_dim≠128 / GQA | falls back to flash-attn |  |
| no calibration & no theta | skip disabled → dense |  |
| sparsity = 0 | dense, no skip |  |
| valid case (SM103+FP8+D128+MHA) | trtllm-gen, skip active |  |

## Notes

- **SAGE = no calibration; Skip = per-model calibration.** So Step 1 needs zero setup; Steps 2–3 need Wan's skip calibration first.
- Measure LPIPS vs **BF16 dense**; measure perf on the **target Blackwell SKU under real concurrency**, not the isolated B300 kernel numbers.
- **Cosmos 3 is GQA** (separate `num_key_value_heads`) → outside the dense-MHA validation. Kernel likely runs (trtllm-gen is natively GQA) but SAGE+Skip under GQA is untested — just record run/fallback.
