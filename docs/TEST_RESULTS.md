# Automated test results

## Phase 1 — 2026-11-26

Environment: CPU only (4 cores), Python 3.13, torch 2.15.0, transformers 5.24.1, float32.
Command: `cd backend && pytest -v`

**57 passed, 0 failed, 0 skipped.**

Tiny random models (Llama, Qwen2, Qwen3 configs; no download) cover most tests. Tests marked
`test_real_model` use the real `Qwen/Qwen2.5-0.5B` checkpoint (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`).

| Check | Result |
|---|---|
| Hooked forward == uninstrumented forward | bitwise equal (all 3 architectures + real model) |
| Σ a_i w_i (+ bias) == MLP output | max rel. error < 1e-5 (tiny), 2.9e-7 (Qwen2.5-0.5B), 3.3e-7 (Qwen3-0.6B, manual check) |
| MLP bias handled; reconstruction without bias fails | yes |
| Recomputed gate/up/SiLU/activation == hooked values | error < 1e-4 |
| x = RMSNorm(resid_mid) recomputed from the raw residual | error < 1e-4 |
| Zero / scale / negate / set deltas == (a′−a)·w at chosen positions only | atol 1e-5 |
| Scaling is proportional | yes |
| Hook ablation == independent weight edit on a deep copy; original weights unchanged | yes |
| Hooks removed after normal runs and after exceptions | yes |
| Repeated baseline runs identical | bitwise |
| Token ↔ string round trip incl. leading whitespace (" John" ≠ "John") | yes |
| Layer/neuron/position validation; paper-neuron checkpoint guard | yes |
| Attention capture: rows sum to 1, causal, Transformers' own hooks don't alter logits | yes |
| Concurrent API requests (16 mixed requests across 8 threads) | all 200, 0 hooks left |

A bug found and fixed during Phase 1: the API's post-request hook-leak check counted hooks without
holding the model lock, so concurrent requests could see each other's hooks and fail with 500
(the check refused to return results, as designed). It is fixed, and the concurrency test above guards it.

UI end-to-end check (headless Chromium against the running backend with Qwen2.5-0.5B): the explorer shows
the real prediction " Rip" (p = 0.9394) for "The password cracking tool is called John the". The microscope
shows a reconstruction error of 2.93e-7. A zero-ablation of L10 n0 at the last token gives KL 7.65e-8.
No console errors.

<details><summary>Full pytest listing</summary>

```
tests/test_correctness.py::test_hooked_matches_plain[llama] PASSED
tests/test_correctness.py::test_hooked_matches_plain[qwen2] PASSED
tests/test_correctness.py::test_hooked_matches_plain[qwen3] PASSED
tests/test_correctness.py::test_reconstruction[llama] PASSED
tests/test_correctness.py::test_reconstruction[qwen2] PASSED
tests/test_correctness.py::test_reconstruction[qwen3] PASSED
tests/test_correctness.py::test_reconstruction_with_bias PASSED
tests/test_correctness.py::test_inspector_recompute[llama] PASSED
tests/test_correctness.py::test_inspector_recompute[qwen2] PASSED
tests/test_correctness.py::test_inspector_recompute[qwen3] PASSED
tests/test_correctness.py::test_output_vector_is_down_proj_column[llama] PASSED
tests/test_correctness.py::test_output_vector_is_down_proj_column[qwen2] PASSED
tests/test_correctness.py::test_output_vector_is_down_proj_column[qwen3] PASSED
tests/test_correctness.py::test_intervention_delta[llama-zero-0.0-0.0] PASSED
tests/test_correctness.py::test_intervention_delta[llama-scale-2.5-2.5] PASSED
tests/test_correctness.py::test_intervention_delta[llama-negate-0.0--1.0] PASSED
tests/test_correctness.py::test_intervention_delta[llama-scale-0.0-0.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen2-zero-0.0-0.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen2-scale-2.5-2.5] PASSED
tests/test_correctness.py::test_intervention_delta[qwen2-negate-0.0--1.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen2-scale-0.0-0.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen3-zero-0.0-0.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen3-scale-2.5-2.5] PASSED
tests/test_correctness.py::test_intervention_delta[qwen3-negate-0.0--1.0] PASSED
tests/test_correctness.py::test_intervention_delta[qwen3-scale-0.0-0.0] PASSED
tests/test_correctness.py::test_set_override[llama] PASSED
tests/test_correctness.py::test_set_override[qwen2] PASSED
tests/test_correctness.py::test_set_override[qwen3] PASSED
tests/test_correctness.py::test_scaling_is_proportional[llama] PASSED
tests/test_correctness.py::test_scaling_is_proportional[qwen2] PASSED
tests/test_correctness.py::test_scaling_is_proportional[qwen3] PASSED
tests/test_correctness.py::test_ablation_matches_weight_edit[llama] PASSED
tests/test_correctness.py::test_ablation_matches_weight_edit[qwen2] PASSED
tests/test_correctness.py::test_ablation_matches_weight_edit[qwen3] PASSED
tests/test_correctness.py::test_compare_runs_layer_checks[llama] PASSED
tests/test_correctness.py::test_compare_runs_layer_checks[qwen2] PASSED
tests/test_correctness.py::test_compare_runs_layer_checks[qwen3] PASSED
tests/test_correctness.py::test_hooks_removed_on_error[llama] PASSED
tests/test_correctness.py::test_hooks_removed_on_error[qwen2] PASSED
tests/test_correctness.py::test_hooks_removed_on_error[qwen3] PASSED
tests/test_correctness.py::test_repeat_baseline_identical[llama] PASSED
tests/test_correctness.py::test_repeat_baseline_identical[qwen2] PASSED
tests/test_correctness.py::test_repeat_baseline_identical[qwen3] PASSED
tests/test_correctness.py::test_validation[llama] PASSED
tests/test_correctness.py::test_validation[qwen2] PASSED
tests/test_correctness.py::test_validation[qwen3] PASSED
tests/test_correctness.py::test_paper_neuron_checkpoint_guard PASSED
tests/test_correctness.py::test_dimensions_read_from_model[llama] PASSED
tests/test_correctness.py::test_dimensions_read_from_model[qwen2] PASSED
tests/test_correctness.py::test_dimensions_read_from_model[qwen3] PASSED
tests/test_correctness.py::test_attention_capture_and_foreign_hooks[llama] PASSED
tests/test_correctness.py::test_attention_capture_and_foreign_hooks[qwen2] PASSED
tests/test_correctness.py::test_attention_capture_and_foreign_hooks[qwen3] PASSED
tests/test_real_model.py::test_real_hooked_matches_plain_and_reconstructs PASSED
tests/test_real_model.py::test_tokens_leading_whitespace PASSED
tests/test_real_model.py::test_api_end_to_end PASSED
tests/test_real_model.py::test_api_concurrent_requests PASSED
============================= 57 passed in 11.46s ==============================
```
</details>
