# Skip-Softmax + SAGE Attention Backend — Test Results

Fill these in during the vLLM-Omni integration. **Single backend under test: `trtllm` = FlashInfer trtllm-gen** (the only SAGE-capable path on B300). Features = `sage` (FP8-SAGE attention) + `skip_softmax`. SAGE attention on B300 is **FP8** (NVFP4 = GEMM/weight axis only). No cross-backend comparison here — trtllm-gen only.

Baseline for all speedups = **BF16 dense**. **Primary quality metric = LPIPS ↓** vs the BF16-dense output (others optional).

**Phased — keep Phase 1 simple (NO calibration, NO skip):**
- **Phase 1 (Table 2)** — backend verification. Per model, only **BF16 dense** and **FP8-SAGE** (SAGE is runtime/dynamic → **no ModelOpt calibration**). Compare **trtllm-gen vs current backend**: LPIPS parity + speedup. This proves the new backend works and FP8-SAGE's cost/quality.
- **Phase 2 (Table 3, later)** — add **Skip-Softmax** at fidelity **D = 1.00 / 0.97 / 0.94**. Skip needs a threshold → **ModelOpt skip calibration per model** (each model has its own D→factor formula). Not started until Phase 1 passes.

> **Calibration:** SAGE = none. Skip = per-model ModelOpt calibration (or a manual factor). So Phase 1 needs zero calibration.

---

## Record — models under test

| Model | HF checkpoint | params | task / shape | target SM |
|-------|---------------|--------|--------------|-----------|
| Wan 2.2 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B (A14B MoE) | T2V, 720p | SM103 (B300) |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | T2V, 720p | SM103 (B300) |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | 64B | T2V, 720p | SM103 (B300) |

---

## Model checkpoints (chosen)

| Model | HF checkpoint | params | task | default shape | notes |
|-------|---------------|--------|------|---------------|-------|
| **Wan 2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B (A14B MoE) | T2V | 720p, 5s | H40/D128 = the FlashInfer B300 validated shape |
| **Hunyuan Video 1.5** | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 8.3B | T2V | 720p | **FP8 E4M3 checkpoints already exist** (`hy15_720p_t2v_fp8_e4m3_lightx2v`) → feeds the FP8 arm directly |
| **Cosmos 3 Super** | `nvidia/Cosmos3-Super` | 64B | T2V | 720p, 5s | vllm-omni `Cosmos3OmniDiffusersPipeline` + FP8 loader #5076 (per supported_models doc) |

**Scope:** all three run **T2V @ 720p only**. Cosmos = Super only; Hunyuan = 720p only (no Nano / 480p runs).

**vLLM-Omni pipelines (already in repo):**

| Model | pipeline class | file | quant status |
|-------|----------------|------|--------------|
| Hunyuan 1.5 T2V | `HunyuanVideo15Pipeline` | `diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py` | **FP8 wired** — stage config `hunyuan_video_15_dit_fp8.yaml` + `examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py` |
| Hunyuan 1.5 I2V | `HunyuanVideo15ImageToVideoPipeline` | `..._1_5_i2v.py` | same FP8 path |
| Cosmos 3 | `Cosmos3OmniDiffusersPipeline` | `diffusion/models/cosmos3/pipeline_cosmos3.py` (uses Wan VAE + UniPC) | FP8 loader #5076 |
| Wan 2.2 I2V | Wan I2V pipeline | `diffusion/models/wan/...` | ModelOpt FP8/NVFP4 loader #5076/#5087 |

---

## Table 1 — Phase 0 (optional): FlashInfer kernel validated-case micro-test

Reproduce the B300 kernel report at the operating shape, at kernel level (no pipeline). Fixed shape from the report: **S=75600 · H=40 · D=128 · FP8 E4M3 · SM103**, single GPU + seed. Reference = the same kernel run **dense (Skip off)** — and, separately, a **full-precision (BF16) oracle** so we test accuracy vs truth, not just FI≈TRT.

| # | Attention mode | threshold θ (factor) | ref = | cosine sim ↑ | relative-L1 ↓ | RMSE ↓ | kernel time (ms) | speedup vs dense | achieved sparsity | PASS? |
|---|----------------|:--------------------:|-------|:------------:|:-------------:|:------:|:----------------:|:----------------:|:-----------------:|:-----:|
| 1 | FP8-SAGE (dense)      | — (skip off) | BF16 oracle |  |  |  |  | 1.00× | — |  |
| 2 | FP8-SAGE + Skip @θ₁   | (deploy θ₁)  | BF16 oracle |  |  |  |  |  |  |  |
| 3 | FP8-SAGE + Skip @θ₂   | (deploy θ₂)  | BF16 oracle |  |  |  |  |  |  |  |
| 4 | FP8-SAGE + Skip @θ_hi | (high-sparsity) | BF16 oracle |  |  |  |  |  |  |  |

Also record **concurrent** (multi-GPU) vs **isolated** kernel time for rows 2–3 (the report's concurrent run regressed) :

| mode | isolated speedup vs dense | 4-GPU concurrent speedup | PASS (≥1.0× under concurrency)? |
|------|:-------------------------:|:------------------------:|:-------------------------------:|
| FP8-SAGE+Skip @θ₁ |  |  |  |
| FP8-SAGE+Skip @θ₂ |  |  |  |

Gate: (a) accuracy vs BF16 oracle within budget (not just FI≈TRT); (b) net speedup ≥ 1.0× **under concurrency**, not only isolated.

---

## Table 2 — Phase 1: backend verification (no skip, no calibration)

Per model, run **BF16 dense** and **FP8-SAGE** on the **current backend** and on **trtllm-gen**. Same prompt + seed. Primary check: **LPIPS parity (trtgen vs current) ≈ 0** and the FP8-SAGE speedup + quality.

| Model | config | current-backend latency (s) | trtgen latency (s) | speedup | LPIPS (trtgen vs current) ↓ | LPIPS (vs BF16 dense) ↓ | notes |
|-------|--------|:---------------------------:|:------------------:|:-------:|:---------------------------:|:-----------------------:|-------|
| Wan 2.2 A14B | BF16 dense |        |        | 1.00× |        | 0 (ref) |  |
| Wan 2.2 A14B | FP8-SAGE   |        |        |        |        |         |  |
| Hunyuan 1.5  | BF16 dense |        |        | 1.00× |        | 0 (ref) |  |
| Hunyuan 1.5  | FP8-SAGE   |        |        |        |        |         |  |
| Cosmos 3 Super | BF16 dense |      |        | 1.00× |        | 0 (ref) | GQA — record if trtgen falls back |
| Cosmos 3 Super | FP8-SAGE   |      |        |        |        |         | GQA — record if trtgen falls back |

---

## Table 3 — Phase 2 (later): Wan 2.2 A14B config × D sweep (needs calibration)

**Only after Phase 1 passes.** Adds Skip-Softmax → **requires ModelOpt skip calibration for Wan** (D→threshold_scale_factor). Shape: **720×1080, 81 frames, 50 steps**. D = skip fidelity (1.00 = near-lossless/no skip → 0.94 = most aggressive).

| config | D = 1.00 |  | D = 0.97 |  | D = 0.94 |  |
|--------|:--------:|:--:|:--------:|:--:|:--------:|:--:|
| | **LPIPS ↓** | **speedup** | **LPIPS ↓** | **speedup** | **LPIPS ↓** | **speedup** |
| BF16 (+Skip)        | 0 (ref) | 1.00× |  |  |  |  |
| FP8 (+Skip)         |  |  |  |  |  |  |
| BF16 + SAGE (+Skip) |  |  |  |  |  |  |
| FP8 + SAGE (+Skip)  |  |  |  |  |  |  |

(Record achieved sparsity per cell in notes. Prereq: `quantize_wan_*_modelopt` skip-calibration produces Wan's D→factor formula.)

---

## Table 4 — Fallback / correctness gates

| # | Case | Config | Expected | Result (PASS/FAIL) | notes |
|---|------|--------|----------|:------------------:|-------|
| 1 | unsupported SM (non-Blackwell) | attention_backend: trtllm | falls back to flash-attn, one-line log |  |  |
| 2 | head_dim ≠ 128 | trtllm | falls back to flash-attn |  |  |
| 3 | GQA (q_heads ≠ kv_heads) | trtllm | falls back to flash-attn |  |  |
| 4 | no calibration & no theta | skip_softmax, sparsity set | skip disabled → dense (no guess) |  |  |
| 5 | sparsity = 0 (or use_trtllm=false) | skip_softmax | dense, no skip |  |  |
| 6 | valid case | SM103+FP8+D128+MHA | runs trtllm-gen, skip active |  |  |
| 7 | None vs 0.0 factor | — | identical output (≤1e-6) |  |  |
| 8 | negative / NaN / inf factor | — | rejected by API |  |  |
| 9 | disabled_until_timestep phase | — | dense early steps, skip later; CUDA graphs keyed per phase |  |  |
| 10 | achieved sparsity realizes target | sparsity=0.65 | achieved ≈ target (within budget) |  |  |

---

## Notes / caveats to record

- E2E quality must be measured against **BF16 dense** — kernel parity (FI vs TRT) does **not** cover the runtime bf16→fp8 SAGE quant step.
- Perf must be measured on the **target Blackwell SKU under realistic concurrency** — do not cite the isolated B300 kernel numbers.
- If any model is not `head_dim=128 + dense MHA`, it is **outside the validated kernel envelope** even on SM103+FP8 → gate it out or request a kernel extension from DevTech (head_dim=64 / GQA).
- **Cosmos 3 is GQA** (`transformer_cosmos3.py` has separate `num_key_value_heads`) → outside the dense-MHA validation. Kernel likely runs (trtllm-gen is natively GQA/MQA) but SAGE+Skip under GQA is untested — in Phase 1, just record whether trtgen runs or falls back.
