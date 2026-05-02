# Reproducibility Appendix: Claim-to-Artifact Evidence Map

This appendix maps manuscript-safe claims to concrete repository artifacts and reproducible commands. Unsupported original claims are explicitly marked as unresolved.

| Claim | Status | Evidence Artifact | Command to Reproduce | Notes / Limitations |
|---|---|---|---|---|
| Real gradient boosting model | Supported | `src/core/cognitive_load_model.py`, `tests/test_cognitive_load.py` | `pytest tests/test_cognitive_load.py` | Implemented as optional `GradientBoostingRegressor` with explicit fallback when dependency is unavailable. |
| Kalman smoothing | Supported | `src/core/kalman_filter.py`, `src/core/cognitive_load_model.py`, `tests/test_cognitive_load.py` | `pytest tests/test_cognitive_load.py` | Implemented as optional 1D smoothing in prediction path. |
| Isolation Forest anomaly detection | Supported | `src/core/anomaly_detector.py`, `experiments/run_experiment.py`, `tests/test_run_experiment_llm.py` | `pytest tests/test_run_experiment_llm.py` | Implemented with dependency-gated Isolation Forest and documented fallback behavior. |
| LLM decision layer | Supported | `src/agents/llm_agent.py`, `experiments/run_experiment.py`, `tests/test_llm_agent.py`, `tests/test_run_experiment_llm.py` | `pytest tests/test_llm_agent.py tests/test_run_experiment_llm.py` | Decision-layer module supports OpenAI mode and deterministic mock mode. |
| Optional LSTM module | Supported | `src/core/sequence_model.py`, `src/core/cognitive_load_model.py`, `tests/test_cognitive_load.py` | `pytest tests/test_cognitive_load.py` | Experimental and dependency-gated (`torch` required). |
| Synthetic demo experiment pipeline | Supported | `experiments/run_experiment.py`, `data/README.md` | `python experiments/run_experiment.py --synthetic-demo --n-samples 500 --seed 42 --output-dir experiments/results_phase5` | Synthetic data validates pipeline behavior only, not human outcomes. |
| Ablation study | Supported | `experiments/run_experiment.py`, `tests/test_run_experiment_llm.py` | `python experiments/run_experiment.py --synthetic-demo --run-ablation --seed 42 --output-dir experiments/results_phase5` | Generates ablation outputs for engineering validation in synthetic mode. |
| LLM decision summary | Partially Supported | `experiments/run_experiment.py`, `experiments/generate_tables.py` | `python experiments/run_experiment.py --synthetic-demo --run-ablation --use-llm-agent --seed 42 --output-dir experiments/results_phase5` | Produced when ablation is run with LLM enabled; output is synthetic/demo evidence. |
| Paper table generation | Supported | `experiments/generate_tables.py`, `tests/test_generate_tables.py` | `pytest tests/test_generate_tables.py` | Generates tables only from available CSV inputs and skips missing unsupported sources. |
| User study template | Supported | `experiments/user_study_template.py`, `tests/test_user_study_template.py` | `pytest tests/test_user_study_template.py` | Provides intake template/analysis utilities; does not provide real participant data by itself. |
| Ethics/privacy documentation | Supported | `docs/ETHICS_AND_PRIVACY.md`, `src/core/privacy_manager.py`, `README.md` | `pytest tests/test_agents.py` | Documentation and privacy module exist; this is not an IRB approval artifact. |
| Real human participant results | Not Supported | `data/README.md`, `README.md` | `N/A` | Repository does not include real participant dataset/results artifacts. |
| NASA-TLX results | Not Supported | `README.md`, `experiments/generate_figures.py` | `N/A` | Figure/table tooling exists, but no bundled verified participant NASA-TLX result dataset in repo. |
| 120-participant study | Not Supported | `README.md`, `data/README.md` | `N/A` | No evidence artifact for a 120-participant dataset or validated analysis outputs. |
| Physiological measures | Not Supported | `README.md`, `src/core/cognitive_load_model.py` | `N/A` | Feature placeholders exist, but no validated participant physiological dataset/results in repository. |
| Cognitive-load reduction percentage claims | Not Supported | `README.md`, `REALITY_CHECK.md` | `N/A` | Percentage-improvement claims are unsupported without real participant evidence. |

## Reviewer-safe interpretation

- Supported items can be presented as implemented engineering artifacts with reproducible code paths.
- Partially supported items must be scoped to synthetic/demo or dependency-gated conditions.
- Not supported items should be removed from present-tense empirical claims or moved to future work until evidence is collected.
