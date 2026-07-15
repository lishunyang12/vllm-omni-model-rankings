# Skip-Softmax + SAGE Attention Backend — Test Results

Fill these in during the vLLM-Omni integration. Backend = `trtllm` (FlashInfer trtllm-gen); features = `sage` (FP8-SAGE attention) + `skip_softmax`. Note: on B300, SAGE attention is **FP8** (NVFP4 = GEMM/weight axis only); the standalone SAGE_ATTN / SAGE_ATTN_3 backends do **not** run on B300.

Baseline for all speedups = **BF16 compiled-dense**. Quality reference = **BF16 dense** (LPIPS/PSNR measured against it).

---

## Table 1 — Shape / envelope probe (is the model in the validated case?)

Validated kernel envelope: **SM103 (B300) · FP8 · head_dim=128 · dense MHA (q_heads=kv_heads)**. Anything else = needs kernel extension or falls back to flash-attn.

| Model          | head_dim | heads (q / kv) | attn type (MHA/GQA) | typical L (tokens) | target SM | in envelope? | plan (native / needs-kernel / fallback) |
|----------------|:--------:|:--------------:|:-------------------:|:------------------:|:---------:|:------------:|-----------------------------------------|
| Wan 2.2 A14B   |          |                |                     |                    |           |              |                                         |
| Hunyuan Video  |          |                |                     |                    |           |              |                                         |
| Cosmos 3       |          |                |                     |                    |           |              |                                         |

---

## Table 2 — E2E results (operating points, per model)

Fill one block per model. Modes: `dense` = baseline; `SAGE`; `SAGE+Skip`. GEMM quant: BF16 / FP8 / NVFP4.

| Model | GEMM quant | Attention | res / frames | E2E latency (s) | Speedup vs BF16-dense | achieved sparsity | LPIPS ↓ | VBench / PSNR | peak VRAM (GiB) | quality OK? | notes |
|-------|:----------:|:---------:|:------------:|:---------------:|:---------------------:|:-----------------:|:-------:|:-------------:|:---------------:|:-----------:|-------|
| Wan 2.2 | BF16  | dense (ref) |            |                 | 1.00×                 | —                 | 0 (ref) | ref           |                 | ref         |       |
| Wan 2.2 | FP8   | SAGE        |            |                 |                       | —                 |         |               |                 |             |       |
| Wan 2.2 | FP8   | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |
| Wan 2.2 | NVFP4 | SAGE        |            |                 |                       | —                 |         |               |                 |             |       |
| Wan 2.2 | NVFP4 | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |
| Hunyuan | BF16  | dense (ref) |            |                 | 1.00×                 | —                 | 0 (ref) | ref           |                 | ref         |       |
| Hunyuan | FP8   | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |
| Hunyuan | NVFP4 | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |
| Cosmos 3| BF16  | dense (ref) |            |                 | 1.00×                 | —                 | 0 (ref) | ref           |                 | ref         |       |
| Cosmos 3| FP8   | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |
| Cosmos 3| NVFP4 | SAGE+Skip   |            |                 |                       |                   |         |               |                 |             |       |

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
