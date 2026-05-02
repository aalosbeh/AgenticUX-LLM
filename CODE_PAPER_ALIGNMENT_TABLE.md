# Code-Paper Alignment Table (Reviewer Response Audit)

Audit scope: repository artifacts only. No manuscript PDF was found in this repo, so manuscript section/page references are marked from available docs or as `Needs manual verification`.

## 1) Claimed Methods vs Code Implementation

| Manuscript claim | Paper section/page if identifiable | Current status | Code file(s) | Function/class name(s) | Evidence command or test | Notes / required manuscript revision |
|---|---|---|---|---|---|---|
| LLM agents | `README.md` (`Optional LLM Agent`) | Partially Supported | `src/agents/llm_agent.py`, `experiments/run_experiment.py` | `LLMAgent`, `analyze_user_state()`, `run_single_configuration()` | `pytest tests/test_llm_agent.py tests/test_run_experiment_llm.py` | Implemented as optional decision layer with mock fallback; avoid claiming fully deployed central LLM orchestration. |
| Executive Agent | `README.md` (`Executive Agent`) | Supported | `src/agents/executive_agent.py` | `ExecutiveAgent`, `orchestrate_adaptation()` | `pytest tests/test_agents.py` | Coordination exists in Python prototype. |
| Behavior Analysis Agent | `README.md` (`Behavior Analysis Agent`) | Supported | `src/agents/behavior_analysis_agent.py` | `BehaviorAnalysisAgent`, `estimate_cognitive_load()`, `detect_anomalies()` | `pytest tests/test_agents.py` | Behavioral/NASA-style score logic is implemented. |
| Interface Agent | `README.md` (`Interface Agent`) | Supported | `src/agents/interface_agent.py` | `InterfaceAgent`, `adapt_interface()` | `pytest tests/test_agents.py` | CSS/DOM adaptation generation exists; expected benefit values are heuristic. |
| Workflow Agent | `README.md` (`Workflow Agent`) | Supported | `src/agents/workflow_agent.py` | `WorkflowAgent`, `analyze_task()`, `get_next_step_guidance()` | `pytest tests/test_agents.py` | Workflow planning/guidance implemented as rule/template engine. |
| Learning Module | `README.md` (`Learning Module`) | Supported | `src/agents/learning_module.py` | `LearningModule`, `analyze_user()`, `get_personalized_strategy()` | `pytest tests/test_agents.py` | Personalization/learning logic exists, but no real study validation bundled. |
| Gradient Boosting | `README.md`, `REPRODUCIBILITY_APPENDIX.md` | Supported | `src/core/cognitive_load_model.py` | `RealGradientBoostingModel`, `CognitiveLoadModel._build_gb_model()` | `pytest tests/test_cognitive_load.py` | Real `sklearn` GB is dependency-gated; fallback exists. |
| Kalman filtering | `README.md`, `REPRODUCIBILITY_APPENDIX.md` | Supported | `src/core/kalman_filter.py`, `src/core/cognitive_load_model.py` | `KalmanFilter`, `CognitiveLoadModel(...use_kalman=True)` | `pytest tests/test_cognitive_load.py` | Implemented as 1D smoothing; no advanced state model benchmark evidence. |
| LSTM | `README.md`, `REPRODUCIBILITY_APPENDIX.md` | Partially Supported | `src/core/sequence_model.py` | `LSTMRegressor` | `pytest tests/test_cognitive_load.py` | Optional/experimental and dependency-gated (`torch`), not central runtime path. |
| Isolation Forest | `README.md`, `REPRODUCIBILITY_APPENDIX.md` | Supported | `src/core/anomaly_detector.py`, `experiments/run_experiment.py` | `BehaviorAnomalyDetector`, `score()`, `predict()` | `pytest tests/test_cognitive_load.py tests/test_run_experiment_llm.py` | Implemented with explicit z-score fallback. |
| Anomaly detection | `README.md` | Supported | `src/core/anomaly_detector.py`, `src/agents/behavior_analysis_agent.py` | `BehaviorAnomalyDetector`, `BehaviorAnalysisAgent.detect_anomalies()` | `pytest tests/test_cognitive_load.py tests/test_agents.py` | Two implementations exist (model-based + rule-threshold agent checks). |
| Browser extension / DOM adaptation | `README.md` (`Browser Extension`) | Supported | `src/browser_extension/content_script.js`, `src/browser_extension/background.js`, `src/browser_extension/manifest.json` | `BehavioralDataCollector`, `BehavioralDataAggregator` | Needs manual verification (manual extension load/runtime not covered by automated tests) | Extension scaffolding and DOM event capture/adaptation messaging exist. |
| Privacy manager | `README.md`, `docs/ETHICS_AND_PRIVACY.md` | Supported | `src/core/privacy_manager.py` | `PrivacyManager`, `record_consent()`, `sanitize_data()`, `request_data_deletion()` | `pytest tests/test_agents.py` | Privacy module exists; IRB artifact is still missing. |
| Weighted round-robin | Needs manual verification | Not Supported | N/A | N/A | `rg "weighted round[- ]robin|round_robin" .` | No weighted round-robin scheduler implementation found; remove claim or mark future work. |
| Priority mutex | Needs manual verification | Not Supported | N/A | N/A | `rg "priority mutex|mutex" .` | No priority mutex primitive found; remove/soften claim. |
| Real-time/sub-100ms latency | `README.md` (`Performance Characteristics`) | Partially Supported | `README.md`, `src/core/behavior_processor.py` | `BehaviorProcessor` | Needs manual verification (no latency benchmark script/test) | Claimed in docs only; no benchmark artifact proving sub-100ms E2E latency. |

## 2) Reported Results / Tables / Figures vs Reproducible Outputs

| Reported result/table/figure | Manuscript claim/value | Current reproducibility status | Script that generates it | Output file | Dataset required | Notes / action needed |
|---|---|---|---|---|---|---|
| Cognitive load reduction | `36.97%` | Not Supported | N/A | N/A | Real participant dataset | No artifact in repo ties this exact percentage to a validated participant analysis. |
| Task completion improvement | `27.57%` | Not Supported | N/A | N/A | Real participant dataset | No reproducible script/output in repo for this value. |
| Satisfaction improvement | `21.33%` | Not Supported | N/A | N/A | Real participant dataset | No reproducible script/output in repo for this value. |
| NASA-TLX tables | NASA-TLX outcomes | Not Supported | `experiments/generate_figures.py` (legacy placeholders only) | None generated for TLX tables | Participant NASA-TLX data tables | Script explicitly marks TLX figure inputs as missing legacy files. |
| Physiological metrics | HRV/pupil outcomes | Not Supported | N/A (only feature placeholders in model inputs) | None | Participant physiological data + analysis pipeline | Repo has placeholders/features, not participant-derived physiological results. |
| 120-participant study | `n=120` | Not Supported | N/A | None | Human-subject dataset + protocol artifacts | No 120-participant dataset or results package in repo. |
| Table 1 | Model metrics table | Supported (synthetic/demo pipeline) | `experiments/generate_tables.py` | `experiments/final_tables/table1_model_metrics.csv` | `results.csv` from experiment run | Reproducible from available outputs; synthetic evidence only in current checked files. |
| Table 2 | Ablation table | Supported (synthetic/demo pipeline) | `experiments/generate_tables.py` | `experiments/final_tables/table2_ablation_study.csv` | `ablation_results.csv` | Present and reproducible from run outputs. |
| Table 3 | LLM decision summary | Supported (synthetic/demo pipeline) | `experiments/generate_tables.py`, `experiments/run_experiment.py` | `experiments/final_tables/table3_llm_decision_summary.csv` | `llm_decision_summary.csv` | Reproducible; indicates decision-layer action counts, not human efficacy. |
| Table 4 | User study condition comparison | Partially Supported | `experiments/generate_tables.py`, `experiments/user_study_template.py` | Only generated if `condition_comparison.csv` exists | Real user-study CSV | Tooling exists; participant result data not bundled. |
| Table 5 | Demographic analysis | Not Supported | N/A | None | Participant demographic dataset | No table-generation artifact for demographic results in repo outputs. |
| Table 6 | System performance table | Not Supported | N/A | None | Technical benchmark dataset/logs | No reproducible table output found in repo outputs. |
| Current model metrics table | MAE/MSE/RMSE/R2 | Supported | `experiments/run_experiment.py` + `experiments/generate_tables.py` | `experiments/final_check/results.csv`, `experiments/final_tables/table1_model_metrics.csv` | Synthetic demo (current files) or user-provided input CSV | Fully reproducible for available data source. |
| Ablation table | Multiple configs (`baseline`, `real_gb`, `+kalman`, `+anomaly`, `+llm`) | Supported | `experiments/run_experiment.py --run-ablation` | `experiments/final_check/ablation_results.csv` | Synthetic demo (current files) | Reproducible engineering ablation only. |
| LLM decision summary table | action counts | Supported | `experiments/run_experiment.py --use-llm-agent` | `experiments/final_check/llm_decision_summary.csv` | Synthetic demo (current files) | Reproducible, but not human-study evidence. |
| Generated figures | paper figures | Partially Supported | `experiments/generate_figures.py` | No figure files currently present under `experiments/` | `results.csv` + `predictions.csv` for 4 implemented figures | Script can generate 4 current figures; legacy Fig1-11 participant-style figures are explicitly skipped without missing legacy tables. |

## 3) Unsupported Manuscript Claims

- Claim: Exact participant efficacy percentages (`36.97%`, `27.57%`, `21.33%`).
  - Why unsupported: No participant dataset, no statistical report artifact in repo.
  - Recommended action: `remove` or `mark future work` unless real evidence package is added.
- Claim: 120-participant user study completed/analyzed.
  - Why unsupported: No `n=120` raw/processed data, no participant-level analysis outputs.
  - Recommended action: `remove` or `soften` to planned study.
- Claim: NASA-TLX participant results are reported/reproduced.
  - Why unsupported: TLX tables/legacy inputs are missing; figure script marks these as skipped.
  - Recommended action: `soften` + state missing data, or `mark future work`.
- Claim: Physiological outcome results (not just placeholders) are validated.
  - Why unsupported: Only model input fields/placeholders; no physiological dataset/results.
  - Recommended action: `mark future work`.
- Claim: Weighted round-robin scheduling implemented.
  - Why unsupported: No implementation found in code search.
  - Recommended action: `remove` or `implement`.
- Claim: Priority mutex mechanism implemented.
  - Why unsupported: No implementation found in code search.
  - Recommended action: `remove` or `implement`.
- Claim: Verified real-time sub-100ms latency.
  - Why unsupported: Claim appears in docs but no benchmark script/report artifact.
  - Recommended action: `soften` and add benchmark evidence before asserting.

## 4) Non-Reproducible Experimental Results

- Missing dataset:
  - No bundled real participant dataset in `data/` (only `data/README.md` schema guidance).
- Missing script-linked legacy inputs:
  - `experiments/generate_figures.py` references missing legacy table inputs for NASA-TLX/physiological/demographic/system-comparison figures.
- Missing raw logs:
  - No participant-session raw logs or provenance package for human-study claims.
- Missing statistical analysis for participant outcomes:
  - `experiments/statistical_analysis.py` summarizes model metrics from `results.csv`; no participant hypothesis-testing artifact tied to manuscript percentages.
- Missing ethics/IRB support if human data is claimed:
  - `docs/ETHICS_AND_PRIVACY.md` provides guidance only; no IRB approval artifact/protocol ID/consent package found.

## 5) LLM / Agent Component Assessment

- Is LLM implemented?
  - Yes, as `src/agents/llm_agent.py`.
- Where?
  - `LLMAgent` is invoked in `experiments/run_experiment.py` when `--use-llm-agent` is enabled.
- Is it central or optional?
  - Optional (feature-flagged), with deterministic mock fallback when OpenAI client/key is unavailable.
- Is it a decision layer or regression model?
  - Decision layer for UX action selection; not the cognitive-load regression model.
- Should manuscript title/abstract be reframed?
  - Yes, if currently framed as fully LLM-centered system. Current repo supports a multi-agent prototype with optional LLM decision module, not mandatory LLM core for all runs.

## 6) Dataset / Logs / Generated Outputs

| Artifact | Exists? yes/no | Path | Purpose | Missing pieces |
|---|---|---|---|---|
| Real participant dataset | No | N/A | Human-study evidence | De-identified dataset + provenance |
| Synthetic data generator | Yes | `experiments/run_experiment.py` (`make_synthetic_demo`) | Pipeline/demo validation | Not human evidence |
| Core experiment results | Yes | `experiments/final_check/results.csv` | Model metric outputs | Participant-mode run artifacts |
| Predictions with row-level outputs | Yes | `experiments/final_check/predictions.csv` | Per-sample predictions | Participant labels/evidence |
| Ablation outputs | Yes | `experiments/final_check/ablation_results.csv`, `experiments/final_check/ablation_summary_table.csv` | Engineering ablation | Real-data ablation |
| LLM decision summary | Yes | `experiments/final_check/llm_decision_summary.csv` | Action distribution counts | Participant-grounded interpretation |
| Paper tables (1-3) | Yes | `experiments/final_tables/table1_model_metrics.csv`, `experiments/final_tables/table2_ablation_study.csv`, `experiments/final_tables/table3_llm_decision_summary.csv` | Publication-ready reproducible tables | Tables 4-6 complete with real data |
| User study template | Yes | `experiments/user_study_template.py`, `experiments/final_user_study/user_study_template.csv` | Intake schema/template | Actual study data and analysis outputs |
| Generated figure files under experiments | No | N/A | Visual result outputs | Run `generate_figures.py`; provide required data for legacy figures |
| Ethics/privacy guidance doc | Yes | `docs/ETHICS_AND_PRIVACY.md` | Ethics guidance text | IRB approval artifacts |

## 7) Privacy / Ethics / User Data Handling

- What exists in code/docs:
  - `src/core/privacy_manager.py` includes consent levels, sanitization, retention policy metadata, deletion/export helpers.
  - `src/browser_extension/content_script.js` includes privacy-conscious collection behavior (normalized coordinates, sensitive-input filtering).
  - `docs/ETHICS_AND_PRIVACY.md` documents consent/anonymization recommendations and synthetic-data disclaimer.
- What is missing from manuscript-supporting artifact set:
  - Formal IRB/ethics approval package.
  - Explicit participant consent form/version and retention/deletion process records.
  - Data governance provenance linking any claimed human results to approved collection protocol.
- What must be added before resubmission (if human claims remain):
  - IRB protocol ID/approval evidence (or equivalent ethics committee documentation).
  - Participant consent documentation summary and data handling SOP.
  - De-identification and data lineage/provenance appendix for every reported participant result.

## 8) Final Recommendation

- What is safe to claim now:
  - Multi-agent prototype architecture with implemented Executive/Behavior/Interface/Workflow/Learning components.
  - Optional algorithms implemented in code: real GB (dependency-gated), Kalman smoothing, Isolation Forest fallback path, optional LSTM module.
  - Reproducible synthetic/demo experiment pipeline with generated model metrics and ablation/LLM decision summary tables.
- What must be removed/softened:
  - Any present-tense participant efficacy percentages and 120-participant claims.
  - NASA-TLX/physiological outcome claims unless tied to reproducible participant artifacts.
  - Claims of weighted round-robin, priority mutex, and verified sub-100ms latency unless implemented/benchmarked.
- What requires new implementation:
  - Weighted round-robin and priority mutex if these are to remain method claims.
  - Benchmark harness/report proving latency claims.
  - Full runtime integration evidence if positioning LLM as central.
- What requires real user study/data collection:
  - All human-subject efficacy claims (cognitive load reduction, task completion, satisfaction, NASA-TLX, physiological metrics, participant-count claims).
  - Full participant-level statistical analysis and ethics-compliant evidence package.
