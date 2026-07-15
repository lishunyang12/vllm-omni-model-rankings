# Skip-Softmax + SAGE Attention Backend — Test Results

Fill these in during the vLLM-Omni integration. **Single backend under test: `trtllm` = FlashInfer trtllm-gen** (the only SAGE-capable path on B300). Features = `sage` (FP8-SAGE attention) + `skip_softmax`. SAGE attention on B300 is **FP8** (NVFP4 = GEMM/weight axis only). No cross-backend comparison here — trtllm-gen only.

Baseline for all speedups = **BF16 compiled-dense**. Quality reference = **BF16 dense** (LPIPS/PSNR/SSIM measured against it).

Quality metrics follow the video-acceleration convention (Sparse-vDiT / SVG2): reference-based **LPIPS ↓ (primary) / PSNR ↑ / SSIM ↑** vs the BF16-dense output, plus **VBench** for overall + temporal quality. Kernel-level (vs full-precision attention): **cosine sim / relative-L1 / RMSE** (SageAttention convention).

> **SCOPE — validated-kernel test only.** This run covers exactly the FlashInfer-validated case: **SM103 (B300) · FP8 E4M3 · head_dim=128 · dense MHA**. **NVFP4 and GQA are out of scope** (separate follow-up). In-envelope models: **Wan 2.2 (40/40 MHA)**, **Hunyuan 1.5 (16/16 MHA)**. **Cosmos 3 Super is GQA → not in the validated envelope**, so it is a follow-up item, not part of this test.

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

## Table 1 — Shape / envelope probe (is the model in the validated case?)

Validated kernel envelope (from the B300 report): **SM103 · FP8 · head_dim=128 · dense MHA (q_heads=kv_heads)**. Values below read from the actual HF `transformer/config.json` (Wan, Hunyuan) — Cosmos config is gated, confirm on the box.

| Model          | head_dim | heads (q / kv) | attn type | typical L (tokens) | target SM | in envelope? | plan |
|----------------|:--------:|:--------------:|:---------:|:------------------:|:---------:|:------------:|------|
| Wan 2.2 (T2V, A14B)   | **128** | **40 / 40** | **MHA** ✓ | ~75,600 @720p | SM103 | **yes** (= validated shape) | native |
| Hunyuan Video 1.5 (T2V) | **128** | **16 / 16** | **MHA** ✓ | (compute per res) | SM103 | **yes** | native |
| **Cosmos 3 Super 64B** (T2V/I2V/V2V) | 128 (confirm) | **q / kv — likely GQA** | **GQA?** | (compute) | SM103 | **⚠ verify** | see note |

> **⚠ Cosmos 3 GQA finding:** the vllm-omni `transformer_cosmos3.py` attention takes `num_attention_heads` **and a separate `num_key_value_heads`**, projecting K/V with `num_kv_heads` — i.e. the arch is **GQA-capable** (Cosmos-Predict2 public is GQA). The B300 validation only covered **dense MHA**, so Cosmos is **not covered by that report**. The FlashInfer `trtllm_ragged_attention_deepseek` kernel is natively GQA/MQA (it's the DeepSeek path), so GQA likely runs — but **the SAGE per-block-scale layout + Skip predicate under GQA are untested**. Action: confirm Cosmos's actual `num_kv_heads` and validate SAGE+Skip on a GQA shape before trusting it; else fall back to flash-attn for Cosmos.

---

## Table 1b — FlashInfer kernel validated-case micro-test

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

## Table 2 — E2E results (validated case: FP8, T2V @ 720p)

GEMM quant = **FP8** (the validated case, no NVFP4). Attention modes: `dense` (BF16 ref) / `FP8-SAGE` / `FP8-SAGE+Skip`. Only in-envelope models (Wan, Hunyuan).

| Model | GEMM quant | Attention | res / frames | E2E latency (s) | Speedup vs BF16-dense | achieved sparsity | LPIPS ↓ | PSNR/SSIM/VBench | peak VRAM (GiB) | quality OK? | notes |
|-------|:----------:|:---------:|:------------:|:---------------:|:---------------------:|:-----------------:|:-------:|:----------------:|:---------------:|:-----------:|-------|
| Wan 2.2 | BF16 | dense (ref) |         |                 | 1.00×                 | —                 | 0 (ref) | ref              |                 | ref         |       |
| Wan 2.2 | FP8  | SAGE        |         |                 |                       | —                 |         |                  |                 |             |       |
| Wan 2.2 | FP8  | SAGE+Skip   |         |                 |                       |                   |         |                  |                 |             |       |
| Hunyuan 1.5 | BF16 | dense (ref) |     |                 | 1.00×                 | —                 | 0 (ref) | ref              |                 | ref         |       |
| Hunyuan 1.5 | FP8  | SAGE        |     |                 |                       | —                 |         |                  |                 |             |       |
| Hunyuan 1.5 | FP8  | SAGE+Skip   |     |                 |                       |                   |         |                  |                 |             |       |

**Cosmos 3 Super — out of scope for the validated-kernel test** (GQA, not in the SM103/FP8/D128/MHA envelope). Follow-up once GQA SAGE+Skip is validated.

---

## Table 3 — Sparsity–fidelity sweep (per model → Pareto + pick default)

Sweep at fixed GEMM quant (note which). `sparsity=0` = dense. Goal: find the knee and set the per-model default.

**Model: __________  ·  GEMM quant: __________  ·  res/frames: __________**

| target_sparsity | disabled_until_timestep | achieved sparsity | Speedup | LPIPS ↓ | recommended default? |
|:---------------:|:-----------------------:|:-----------------:|:-------:|:-------:|:--------------------:|
| 0.0 (dense)     | —                       | 0                 | 1.00×   | 0 (ref) |                      |
| 0.30            | 0.86                    |                   |         |         |                      |
| 0.50            | 0.86                    |                   |         |         |                      |
| 0.65            | 0.86                    |                   |         |         |                      |
| 0.80            | 0.86                    |                   |         |         |                      |
| 0.65            | 0.70                    |                   |         |         |                      |
| 0.65            | 0.95                    |                   |         |         |                      |

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
