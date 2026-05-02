# Reviewer Response Checklist

Use this checklist to track reviewer-facing closure items. Keep entries evidence-based; do not claim completion without repository artifacts.

## 1) LLM integration
- [ ] **Done/Not Done:** Partially done
  - **Current status:** Optional OpenAI-backed `LLMAgent` exists with deterministic mock fallback and tests; full production orchestration and external validation are not complete.
  - **Required fix:** Keep manuscript claims scoped to optional module status unless full runtime integration and validation are added.
  - **Evidence/file path:** `src/agents/llm_agent.py`, `experiments/run_experiment.py`, `tests/test_llm_agent.py`, `tests/test_run_experiment_llm.py`, `REPRODUCIBILITY_APPENDIX.md`

## 2) Algorithm implementation
- [ ] **Done/Not Done:** Partially done
  - **Current status:** Added optional real GradientBoostingRegressor, Kalman smoothing, Isolation Forest anomaly detector, and experimental LSTM module with explicit dependency-gated availability/fallback.
  - **Required fix:** Add benchmark/evaluation evidence on real participant data before strong performance claims.
  - **Evidence/file path:** `src/core/cognitive_load_model.py`, `src/core/kalman_filter.py`, `src/core/anomaly_detector.py`, `src/core/sequence_model.py`, `tests/test_cognitive_load.py`, `REPRODUCIBILITY_APPENDIX.md`

## 3) Experimental data
- [ ] **Done/Not Done:** Partially done (pipeline only)
  - **Current status:** Synthetic demo mode exists for pipeline validation, now with component flags and anomaly scores; no bundled real participant dataset.
  - **Required fix:** Clearly label synthetic-only evidence in manuscript now; for stronger revision, add real dataset pipeline + approved data handling.
  - **Evidence/file path:** `data/README.md`, `experiments/run_experiment.py`, `experiments/user_study_template.py`, `README.md`, `REPRODUCIBILITY_APPENDIX.md`

## 4) References
- [ ] **Done/Not Done:** Not done (verification pending)
  - **Current status:** Reference list may include broad placeholders not tightly mapped to implemented methods.
  - **Required fix:** Ensure citations map to implemented methods and any future-work methods are labeled as non-implemented.
  - **Evidence/file path:** `README.md` ("References"), `REPRODUCIBILITY_APPENDIX.md`, manuscript bibliography source files (outside this repo if applicable)

## 5) Ethics/privacy
- [ ] **Done/Not Done:** Partially done
  - **Current status:** Privacy module and data minimization concepts are present; formal ethics/IRB/process documentation is not fully packaged for reviewer response.
  - **Required fix:** Add explicit ethics/privacy statement, consent/retention/deletion workflow, and synthetic-vs-real data boundaries in manuscript/docs.
  - **Evidence/file path:** `docs/ETHICS_AND_PRIVACY.md`, `src/core/privacy_manager.py`, `README.md`, `ROADMAP.md`, `REPRODUCIBILITY_APPENDIX.md`

## 6) Reproducibility
- [ ] **Done/Not Done:** Partially done
  - **Current status:** Reproducible commands exist for synthetic runs and figure generation.
  - **Required fix:** Add a reproducibility note tying each figure/result claim to source data type (synthetic vs real) and exact commands.
  - **Evidence/file path:** `README.md` ("Reproducible Experiment Commands"), `REPRODUCIBILITY_APPENDIX.md`, `experiments/generate_figures.py`, `experiments/statistical_analysis.py`, `tests/test_generate_tables.py`

## 7) Figures/results
- [ ] **Done/Not Done:** Not done (claim alignment pending)
  - **Current status:** Figure generation exists, but reviewer risk remains if synthetic outputs are presented as participant outcomes.
  - **Required fix:** Relabel or regenerate figure captions/results sections to explicitly indicate synthetic-demo provenance until real dataset evidence is added.
  - **Evidence/file path:** `experiments/generate_figures.py`, `experiments/generate_tables.py`, `README.md`, `REPRODUCIBILITY_APPENDIX.md`, manuscript figures/captions (outside this repo if applicable)

---

## Explicit unresolved items (must remain unresolved unless new evidence is added)

- [ ] **Real participant dataset:** unresolved
  - **Evidence gap:** No bundled validated participant dataset/results artifacts.
  - **Expected future evidence:** de-identified dataset manifest, analysis outputs, provenance docs.
- [ ] **IRB/ethics approval artifact:** unresolved
  - **Evidence gap:** Guidance exists, but no formal IRB/ethics approval package is included.
  - **Expected future evidence:** approval letter/reference, protocol ID, consent documents.
- [ ] **Original claimed performance percentages:** unresolved
  - **Evidence gap:** repository does not contain reproducible participant-level evidence for those percentages.
  - **Expected future evidence:** pre-registered analysis or equivalent reproducible participant-study report.
- [ ] **NASA-TLX/physiological claims:** unresolved
  - **Evidence gap:** tooling exists but no validated participant NASA-TLX/physiological dataset/results are bundled.
  - **Expected future evidence:** participant-derived tables, scripts, and reproducible result artifacts.

---

## Completion gate for resubmission
- [ ] All "Not done" items are either closed with implementation evidence or removed from claims.
- [ ] All "Partially done" items include explicit limitation language and traceable artifacts.
- [ ] Reviewer response letter references concrete file paths and reproducible commands only.
