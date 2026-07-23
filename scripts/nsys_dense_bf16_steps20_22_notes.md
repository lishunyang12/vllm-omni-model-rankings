# nsys_dense_bf16_steps20_22

Nsight Systems timeline of Wan2.2-T2V-A14B, 1280×720 / 81f / 50 steps, single B300
(sm_103), backend **TRTLLM_ATTN dense (BF16)**. Capture range = denoising steps 20–22
(`cudaProfilerStart/Stop` gated on the denoise step index; `--capture-range=cudaProfilerApi`).

## GPU kernel breakdown (captured window)

| Kernel | GPU time % | Note |
|---|---|---|
| `fmhaSm103a…QkvBfloat16…H128…Q128Kv128PersistentContext` | **70.3%** | trtllm-gen FMHA (self/cross attention) |
| `nvjet_sm103…128x256` / `256x256` GEMMs | ~21.5% | MoE / linear layers |
| layer norm / rotary / gelu (triton) | ~8% | elementwise |

Attention dominates (~70%), as expected for a 720p×81f DiT (very long sequence). This is
the operation Skip-Softmax targets.

## Latency note (347s vs "500s+")

- Unprofiled steady-state: **6.9 s/step → ~344 s** (matches the ~347s baseline).
- Under nsys: **9.55 s/step** (hooks attached for the whole run), ~11.4 s/step while
  actively recording → projects to **~477 s+**.

So the "500s+" observed for TRTLLM is **profiler overhead**, not a real regression — dense
TRTLLM_ATTN is ~344s, the same as the baseline, when not running under a profiler.
