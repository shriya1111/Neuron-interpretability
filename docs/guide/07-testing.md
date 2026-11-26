# 7. Testing and verification

## 7.1 Running the tests

```bash
cd backend && pytest -v
```

To run the real-model tests against the cross-check model:

```bash
cd backend && NEURONLAB_TEST_MODEL=Qwen/Qwen2.5-0.5B pytest -v tests/test_real_model.py
```

* **Tiny models.** Most tests build tiny random Llama, Qwen2 and Qwen3 models (3 layers, $d=64$, 96 neurons) in
  memory, so they are fast and need no download. The weights are scaled up so that SiLU's negative region and
  closed gates actually occur.
* **Real model.** The `test_real_model.py` tests use the real checkpoint (Qwen3-0.6B by default) and are skipped if it
  isn't downloaded.

Current status: **80 passed** (see [`../TEST_RESULTS.md`](../TEST_RESULTS.md)).

## 7.2 What each test proves

| Property | Test(s) | Why it matters |
|---|---|---|
| Instrumentation doesn't change the model | `test_hooked_matches_plain`, `test_real_hooked_matches_plain_and_reconstructs` | Otherwise every measurement would be of a different model |
| Neuron contributions sum to the MLP output | `test_reconstruction`, `test_reconstruction_with_bias` | The neuron decomposition is exact, not approximate |
| Displayed gate/up/SiLU/$a$ are what the model computed | `test_inspector_recompute` | The microscope's numbers are real |
| $w$ is the right column | `test_output_vector_is_down_proj_column` | The paper's definition of output vector |
| Interventions change exactly $(a'-a)w$ | `test_intervention_delta`, `test_set_override`, `test_scaling_is_proportional` | No hidden side effects |
| Hook ablation == weight edit | `test_ablation_matches_weight_edit` | Two independent implementations agree; weights untouched |
| Zero ablation == adding $-a w$ | `test_zero_ablation_equals_subtracting_aw` | Validates the vector-add path used by controls |
| Vector add hits only chosen positions | `test_vector_add_resid_post_and_positions` | Steering/controls are localized |
| Controls have matched norm and are reproducible | `test_random_directions_matched_norm` | Fair comparisons |
| Hooks are always cleaned up | `test_hooks_removed_on_error`, `test_api_concurrent_requests` | No leakage between runs or users |
| Determinism | `test_repeat_baseline_identical` | Same input → same output, bit for bit |
| Tokens and whitespace | `test_tokens_leading_whitespace` | `" John"` ≠ `"John"` |
| Validation | `test_validation`, `test_paper_neuron_checkpoint_guard` | Bad layer/neuron/position rejected; paper IDs refused on other models |
| Logit lens maths | `test_folded_scores_match_manual`, `test_minus_readout_is_negation`, `test_residual_lens_last_layer_is_model_prediction` | Chapter 4 formulas hold |
| DLA | `test_dla_equals_direct_effect` | The attribution means what it claims |
| Neuron search | `test_neuron_search_matches_bruteforce` | The fast scan finds the true best |
| Attention capture | `test_attention_capture_and_foreign_hooks` | Rows sum to 1, causal; Transformers' own hooks don't alter outputs |
| End to end API | `test_api_end_to_end`, `test_phase2_api` | Every endpoint works on the real model |

## 7.3 Checks you can do yourself in the app

1. **Reconstruction:** Microscope → "MLP reconstruction check" should show a relative error around $10^{-7}$
   (fp32) or $10^{-3}$ (bf16).
2. **Recompute:** expand "hooked values & recompute error". The errors should be tiny.
3. **No upstream effect:** after any intervention, the bar chart "Change in residual stream" must be zero
   before the edited layer.
4. **Restore:** click *restore*. The modified panel disappears, and the baseline is unchanged.
5. **Lens consistency:** Vocabulary Lenses → residual state. The last row equals the Model Explorer's prediction,
   and the page shows the measured difference.
6. **Hooks:** `GET /health` must report `neuronlab_hooks_attached: 0`.

## 7.4 Bugs the tests caught

* **Phase 1.** The API's hook-leak check ran without holding the model lock, so two simultaneous requests could see each
  other's hooks and fail with an error. It is fixed, and `test_api_concurrent_requests` guards it.
* **Phase 1.** Transformers ≥ 5 permanently installs its own hooks the first time attention weights are requested.
  The leak check now counts only NeuronLab's hooks; the others are reported and verified harmless.
