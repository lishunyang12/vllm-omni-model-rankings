# Skip-Softmax + SAGE Attention Backend — Test Results

Fill these in during the vLLM-Omni integration. Backend = `trtllm` (FlashInfer trtllm-gen); features = `sage` (FP8-SAGE attention) + `skip_softmax`. Note: on B300, SAGE attention is **FP8** (NVFP4 = GEMM/weight axis only); the standalone SAGE_ATTN / SAGE_ATTN_3 backends do **not** run on B300.

Baseline for all speedups = **BF16 compiled-dense**. Quality reference = **BF16 dense** (LPIPS/PSNR/SSIM measured against it).

Quality metrics follow the video-acceleration convention (Sparse-vDiT / SVG2): reference-based **LPIPS ↓ (primary) / PSNR ↑ / SSIM ↑** vs the BF16-dense output, plus **VBench** for overall + temporal quality. Kernel-level (vs full-precision attention): **cosine sim / relative-L1 / RMSE** (SageAttention convention).

---

## Record — models under test

| Model | HF checkpoint | tested task / shape | attention | target SM |
|-------|---------------|---------------------|-----------|-----------|
| Wan 2.2 | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | I2V, 720p | 40/40, D128, **MHA** ✓ | SM103 (B300) |
| Hunyuan Video 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | T2V, 720p | 16/16, D128, **MHA** ✓ | SM103 (B300) |
| Cosmos 3 Super | `nvidia/Cosmos3-Super` | T2V + I2V + V2V, 720p | D128, **GQA? ⚠ verify** | SM103 (B300) |

---

## Model checkpoints (chosen)

| Model | HF checkpoint | task(s) | default shape | notes |
|-------|---------------|---------|---------------|-------|
| **Wan 2.2 I2V** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | I2V | 720p (or 480p), 5s | A14B MoE; H40/D128 = the FlashInfer B300 validated shape |
| **Hunyuan Video 1.5** | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | T2V | 720p, 8.3B | 1.5 line; **FP8 E4M3 checkpoints already exist** (`hy15_720p_*_fp8_e4m3_lightx2v`) → feeds the FP8 arm directly. I2V = `...-720p_i2v` |
| **Cosmos 3 Super 64B** | `nvidia/Cosmos3-Super` | **T2V + I2V + V2V** (unified) | 720p, 5s | vllm-omni `Cosmos3OmniDiffusersPipeline` + FP8 loader #5076 (per supported_models doc) |
| Cheap-iter (Wan) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | TI2V | 720p | lighter shape for Wan sweeps / CI |

**Task coverage:** Wan 2.2 = **I2V**; Hunyuan 1.5 = **T2V @ 720p only**; Cosmos 3 Super 64B = **T2V + I2V + V2V @ 720p only** (unified checkpoint, run each mode). Only the three primaries are in scope — **Cosmos = Super only, Hunyuan = 720p only** (no Nano / 480p runs). Wan may use TI2V-5B for cheap sparsity sweeps, then confirm on Wan I2V A14B.

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
| Wan 2.2 I2V A14B      | **128** | **40 / 40** | **MHA** ✓ | ~75,600 @720p | SM103 | **yes** (= validated shape) | native |
| Hunyuan Video 1.5 (T2V) | **128** | **16 / 16** | **MHA** ✓ | (compute per res) | SM103 | **yes** | native |
| **Cosmos 3 Super 64B** (T2V/I2V/V2V) | 128 (confirm) | **q / kv — likely GQA** | **GQA?** | (compute) | SM103 | **⚠ verify** | see note |

> **⚠ Cosmos 3 GQA finding:** the vllm-omni `transformer_cosmos3.py` attention takes `num_attention_heads` **and a separate `num_key_value_heads`**, projecting K/V with `num_kv_heads` — i.e. the arch is **GQA-capable** (Cosmos-Predict2 public is GQA). The B300 validation only covered **dense MHA**, so Cosmos is **not covered by that report**. The FlashInfer `trtllm_ragged_attention_deepseek` kernel is natively GQA/MQA (it's the DeepSeek path), so GQA likely runs — but **the SAGE per-block-scale layout + Skip predicate under GQA are untested**. Action: confirm Cosmos's actual `num_kv_heads` and validate SAGE+Skip on a GQA shape before trusting it; else fall back to flash-attn for Cosmos.

---

## Table 2 — E2E results (operating points, per model)

Fill one block per model×task. Modes: `dense` = baseline; `SAGE`; `SAGE+Skip`. GEMM quant: BF16 / FP8 / NVFP4. Tasks: Wan 2.2 = **I2V**; Hunyuan 1.5 = **T2V**; Cosmos 3 Super = **T2V / I2V / V2V** (one checkpoint, run each).

| Model | task | GEMM quant | Attention | res / frames | E2E latency (s) | Speedup vs BF16-dense | achieved sparsity | LPIPS ↓ | PSNR/SSIM/VBench | peak VRAM (GiB) | quality OK? | notes |
|-------|:----:|:----------:|:---------:|:------------:|:---------------:|:---------------------:|:-----------------:|:-------:|:----------------:|:---------------:|:-----------:|-------|
| Wan 2.2 | I2V | BF16  | dense (ref) |          |                 | 1.00×                 | —                 | 0 (ref) | ref              |                 | ref         |       |
| Wan 2.2 | I2V | FP8   | SAGE        |          |                 |                       | —                 |         |                  |                 |             |       |
| Wan 2.2 | I2V | FP8   | SAGE+Skip   |          |                 |                       |                   |         |                  |                 |             |       |
| Wan 2.2 | I2V | NVFP4 | SAGE+Skip   |          |                 |                       |                   |         |                  |                 |             |       |
| Hunyuan 1.5 | T2V | BF16  | dense (ref) |      |                 | 1.00×                 | —                 | 0 (ref) | ref              |                 | ref         |       |
| Hunyuan 1.5 | T2V | FP8   | SAGE+Skip   |      |                 |                       |                   |         |                  |                 |             |       |
| Hunyuan 1.5 | T2V | NVFP4 | SAGE+Skip   |      |                 |                       |                   |         |                  |                 |             |       |
| Cosmos 3 | T2V | BF16  | dense (ref) |         |                 | 1.00×                 | —                 | 0 (ref) | ref              |                 | ref         |       |
| Cosmos 3 | T2V | FP8   | SAGE+Skip   |         |                 |                       |                   |         |                  |                 |             |       |
| Cosmos 3 | I2V | FP8   | SAGE+Skip   |         |                 |                       |                   |         |                  |                 |             |       |
| Cosmos 3 | V2V | FP8   | SAGE+Skip   |         |                 |                       |                   |         |                  |                 |             |       |
| Cosmos 3 | T2V | NVFP4 | SAGE+Skip   |         |                 |                       |                   |         |                  |                 |             |       |

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

## Table 5 — Attention backend capability matrix (factual, fill "runs on B300?")

Existing vLLM-Omni diffusion backends vs the new `trtllm`. This frames *why* only some are comparable on Blackwell.

| Backend            | lib / path                    | target SM        | attention dtype      | SAGE | Skip | GQA | runs on B300? |
|--------------------|-------------------------------|------------------|----------------------|:----:|:----:|:---:|:-------------:|
| FLASH_ATTN         | flash-attn                    | 80+              | BF16 / FP16          |  no  |  no  | yes |               |
| FLASH_ATTN_3_HUB   | FA3                           | 90 / 100         | BF16 / FP16          |  no  |  no  | yes |               |
| TORCH_SDPA         | torch SDPA                    | any              | BF16 / FP16          |  no  |  no  | yes |               |
| CUDNN_ATTN         | cuDNN fused                   | 90 / 100         | BF16 / FP16          |  no  |  no  | yes |               |
| FLASHINFER_ATTN    | flashinfer (default path)     | 80+              | BF16 / FP16          |  no  |  no  | yes |               |
| SAGE_ATTN          | `sageattention`               | **80–90 only**   | FP8 / INT8           | yes  |  no  | yes | **no** (SM)   |
| SAGE_ATTN_3        | `sageattn3`                   | **120-focused**  | FP8                  | yes  |  no  | **no** | **no** (SM/GQA) |
| **TRTLLM (new)**   | FlashInfer trtllm-gen         | **100+**         | FP8 E4M3 (SAGE)      | yes  | yes  | *(fallback if GQA)* | **yes** |

Takeaway to confirm on the box: on B300 the only SAGE-capable path is **TRTLLM**; SAGE_ATTN / SAGE_ATTN_3 are out (SM / GQA). Dense baselines = FLASH_ATTN / FA3 / cuDNN / FLASHINFER.

---

## Table 6 — Cross-backend head-to-head (vertical perf comparison)

**Fix one model + shape + prompt + seed**, swap only the attention backend. Baseline for speedup = **FLASH_ATTN** (dense). LPIPS measured vs BF16-dense output.

**Model: __________  ·  res/frames: __________  ·  GEMM quant: __________  ·  SM/GPU: __________**

| Backend              | attention mode  | E2E latency (s) | attn kernel time (ms) | speedup vs FLASH_ATTN | achieved sparsity | LPIPS ↓ | peak VRAM (GiB) | notes |
|----------------------|-----------------|:---------------:|:---------------------:|:---------------------:|:-----------------:|:-------:|:---------------:|-------|
| FLASH_ATTN           | dense (ref)     |                 |                       | 1.00×                 | —                 | 0 (ref) |                 |       |
| FLASH_ATTN_3_HUB     | dense           |                 |                       |                       | —                 |         |                 |       |
| CUDNN_ATTN           | dense           |                 |                       |                       | —                 |         |                 |       |
| FLASHINFER_ATTN      | dense           |                 |                       |                       | —                 |         |                 |       |
| TORCH_SDPA           | dense           |                 |                       |                       | —                 |         |                 |       |
| SAGE_ATTN            | FP8-SAGE        | n/a (SM)        | n/a                   | n/a                   | —                 | n/a     | n/a             | not on B300 |
| SAGE_ATTN_3          | FP8-SAGE        | n/a (SM/GQA)    | n/a                   | n/a                   | —                 | n/a     | n/a             | not on B300 |
| **TRTLLM**           | FP8-SAGE        |                 |                       |                       | —                 |         |                 |       |
| **TRTLLM**           | FP8-SAGE + Skip |                 |                       |                       |                   |         |                 |       |

Run the same block per model (Wan 2.2 / Hunyuan / Cosmos 3). This isolates the attention win: `attn kernel time` column shows where the backends actually differ; `E2E` shows what survives the rest of the pipeline.

---

## Notes / caveats to record

- E2E quality must be measured against **BF16 dense** — kernel parity (FI vs TRT) does **not** cover the runtime bf16→fp8 SAGE quant step.
- Perf must be measured on the **target Blackwell SKU under realistic concurrency** — do not cite the isolated B300 kernel numbers.
- If any model is not `head_dim=128 + dense MHA`, it is **outside the validated kernel envelope** even on SM103+FP8 → gate it out or request a kernel extension from DevTech (head_dim=64 / GQA).
