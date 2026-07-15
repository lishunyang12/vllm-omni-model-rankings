# trtllm-gen Backend — SAGE + Skip-Softmax: Implementation Results

All numbers measured on a real **NVIDIA B300 (SM 10.3 / SM103)**.
Versions: **vLLM-Omni main · FlashInfer 0.6.12 · ModelOpt 0.44.0 · torch 2.11+cu130**.

Branch: `trtllm-gen-attn` (vllm-omni). Design/test plan: `skip_softmax_test_results.md`.

---

## 1. What was built (all B300-validated)

A new diffusion attention backend **`trtllm-gen`** (FlashInfer's TRT-LLM `TllmGenFmhaRunner`),
selectable via `--diffusion-attention-backend trtllm-gen`, with three FP8-attention
granularities + Skip-Softmax, all composable:

| Capability | How |
|-----------|-----|
| BF16 dense (baseline / parity) | `sage=off` |
| per-tensor FP8 | `sage=per_tensor` (reuses vLLM `scaled_fp8_quant`) |
| **per-block SAGE** (true SageAttention) | `sage=per_block` — per-block FP8 Q/K + per-tensor FP8 V |
| Skip-Softmax (sparse) | `target_sparsity` (+ calibrated `a,b`), gated by `disabled_until_timestep` |
| **SAGE + Skip together** | one trtllm-gen kernel call |

**Files** (branch `trtllm-gen-attn`): new `attention/backends/trtllm_gen.py` (backend) +
`tests/diffusion/attention/test_trtllm_gen.py` (24 tests); edits to `registry.py`,
`platforms/cuda/platform.py`, `forward_context.py`, `data.py`, and the wan2_2
(base/i2v/vace) + hunyuan_image3 pipelines (timestep plumbing).

---

## 2. Accuracy — rel error vs BF16 SDPA (B300)

```
mode              rel error
────────────────────────────
BF16 dense        0.0007      ~ bit-parity with SDPA
per-tensor FP8    0.0511      FP8 quantization floor
per-block SAGE    0.0504      true SageAttention
```

per-block SAGE across shapes (all < 0.06, the flashinfer PR #2711 bar is atol/rtol 0.1):

```
B=1 S=256  H=8    0.0504
B=2 S=256  H=8    0.0508
B=2 S=512  H=8    0.0508
B=4 S=256  H=8    0.0497
B=2 S=1024 H=16   0.0519
```

### per-block ≈ per-tensor for FP8 — why (measured)

```
data type                     per_tensor  per_block   ratio
────────────────────────────────────────────────────────────
random gaussian               0.0513      0.0509      1.0x
block-varying magnitude        0.0805      0.0849      0.9x
token outliers                0.0563      0.0582      1.0x
isolated Q quant error         0.0226      0.0220      1.0x
```

**FP8 (e4m3) is floating-point** — its exponent already handles magnitude variation, so
per-block scaling adds little over per-tensor. Per-block's accuracy benefit is an **INT8**
thing (fixed-point). The compiled INT8-QK SAGE kernel is not in this cubin set ("Missing
kernel"); the **FP8-QK** kernel is present and accurate. **Recommendation: for FP8 use
`per_tensor`** (same accuracy as per_block, less quantization overhead).

---

## 3. Skip-Softmax behavior (B300)

```
tiny threshold factor (1e-6)   rel vs dense = 0.0016   ~= dense (skip barely engages)
large threshold factor (5000)  rel = 1.67, finite      real skipping, no NaN
no calibration curve           factor = None            safe -> dense
timestep gate  t=0.9 (>0.86)   -> None (dense guard)    early/noisy steps stay dense
               t=0.3 (<=0.86)  -> factor                late steps skip
SAGE + Skip (tiny)             = SAGE-only              compose in one kernel ✓
```

---

## 4. Kernel-level latency (B300, single attention call)

```
shape (B S H D)     off(BF16)  per_tensor  per_block  sage+skip
──────────────────────────────────────────────────────────────
1 4096  16 128       0.099 ms   0.179 ms    0.376 ms    —
1 16384 16 128       1.32 ms    1.38 ms     2.08 ms     2.31 ms
```

Skip on **random** data (S=16384) is *slower* (0.82–0.88x): random attention has no
skippable blocks, so the predicate is pure overhead.

**Key finding — random-tensor microbenchmarks cannot show the deck's page-8 speedup.**
The techniques exploit real DiT data structure that random tensors lack:

| technique | needs | on random | on real DiT |
|-----------|-------|-----------|-------------|
| per-block SAGE | Q/K outliers | no benefit | better accuracy |
| Skip-Softmax | attention sparsity (near-zero blocks) | overhead, slower | skips → faster |
| FP8 kernel | long-seq matmul | trtllm BF16 already fast → parity | — |

Page-8 numbers (525s→486s SAGE, →461s SAGE+Skip on Wan 2.2 A14B 720p) are **E2E on the
real model** and require running the actual pipeline (`bench_trtllm_gen_sage_skip.py`).

---

## 5. ModelOpt Skip-Softmax calibration (B300)

Used ModelOpt's **official** script `examples/diffusers/sparsity/wan22_skip_softmax.py`
(method `triton_skip_softmax`) — not a hand-written one. Only patch: `load_calib_prompts`
streams OpenVid-1M (first N) with a built-in fallback (full download infeasible offline),
plus a persistence step to write `sparse_attention_config` into `transformer/config.json`
(`export_hf_checkpoint` omits it for Wan/diffusers).

**Skip-Softmax calibration does NOT modify weights** (deck page 6: "No weight
modification") — it fits `scale_factor = a·exp(b·target_sparsity)` and writes a config table.

Wan 2.2 **TI2V-5B** proof run (2 prompts, 480p, 9f, 4 steps, target 0.5):

```
Collected 176 (scale_factor, sparsity) pairs, log-space fit:
  scale_factor = 4.300e-07 · exp(65.422 · sparsity)   →  a = 4.3e-07, b = 65.42
```

Exported `transformer/config.json`:
```json
"sparse_attention_config": {
  "config_groups": {"group_0": {"sparse_algo": "softmax_skip", "targets": ["WanAttention"]}},
  "threshold_scale_factor": {"formula": "a * exp(b * target_sparsity)",
                             "prefill": {"a": 4.2997e-07, "b": 65.422}},
  "producer": {"name": "modelopt", "version": "0.44.0"}
}
```

**End-to-end loop verified** — the backend reads exactly this:
```
ModelOpt calibration → config.json threshold_scale_factor.prefill.{a,b}
   → backend reads skip_a/skip_b → factor = a·exp(b·target_sparsity)
   → kernel skip_softmax_threshold_scale_factor (threshold = factor / seqlen)
```

> Proof-run `a,b` (b≈65) come from a tiny calibration and are not production-usable — a
> real curve needs more prompts + target resolution/frames/steps + the full threshold
> sweep. A14B 720p production calibration was launched; results appended below when done.

---

## 6. Tests

- **24** backend tests pass on B300 (`test_trtllm_gen.py`): BF16 parity, FP8, per-block SAGE
  accuracy, SAGE+Skip compose, skip logic, timestep gate, calibration-config read, GQA/mask
  fallback, seqlen-indivisible fallback.
- **37** existing `test_attention_config.py` pass — no regression.

---

## 7. Key references found

- **flashinfer PR #2711** "DiT-oriented kernels" + `tests/attention/test_trtllm_ragged_dit.py`
  — the authoritative SAGE recipe (V per-tensor via bmm2 + `v_sfs=ones` was the fix that took
  rel 0.19 → 0.05).
- **ModelOpt** `examples/diffusers/sparsity/wan22_skip_softmax.py` — official calibration.
- Deck: `skip_softmax_dit.pdf` (BLASST, arXiv 2512.12087).

---

## 8. Status / remaining

- ✅ Backend (BF16 / FP8 / per-block SAGE / Skip / compose) — implemented + B300-validated.
- ✅ Calibration → config.json → backend read — end-to-end verified.
- ✅ Timestep plumbing (wan2_2 family + hunyuan_image3).
- ⏳ A14B production calibration — running.
- ⏳ E2E page-8 speedup — needs the full vLLM-Omni generation run on a calibrated checkpoint
  (`bench_trtllm_gen_sage_skip.py`); cannot be shown by microbenchmarks.
