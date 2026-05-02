# Engineering Roadmap

This roadmap provides two explicit paths for resubmission planning. It is documentation-only and does not claim additional implementation beyond the current repository state.

## Track A - Truth-aligned revision

Goal: align manuscript claims with what is currently implemented and reproducible in this codebase.

### A1) Rename/describe current system accurately
- Reframe the system as a **prototype multi-agent UX personalization framework** with:
  - Python agent orchestration
  - behavioral feature processing
  - cognitive-load prediction via residual linear ensemble + feedforward NN
  - synthetic-demo experiment pipeline
- Remove language implying deployed autonomous LLM operations or production-grade validated outcomes.

### A2) Remove unsupported claims
- Remove or rewrite claims for methods not yet implemented in code:
  - production LLM integration
  - Kalman filtering
  - LSTM sequence modeling
  - Isolation Forest anomaly detection
  - LightGBM/tree-boosting production model
  - real participant dataset results
- Replace with "future work" or "planned implementation" wording.

### A3) Explain synthetic-demo limitations
- State clearly that synthetic demo data validates pipeline behavior only.
- Add explicit caveat: no human-subject generalization claims are supported by synthetic outputs.
- Ensure any figures produced from synthetic runs are labeled as non-participant, non-clinical, and non-deployment evidence.

### A4) Manuscript sections that must change
- **Title/Abstract**: remove "LLM-powered" or equivalent if presented as implemented.
- **Methods**: restrict to implemented modules and algorithms only.
- **System Architecture**: remove components not present in code.
- **Experiments/Dataset**: identify synthetic-demo setup; remove participant-study language.
- **Results/Discussion**: remove unsupported performance/causal claims.
- **Limitations**: add explicit prototype and synthetic-data constraints.
- **Future Work**: move currently unimplemented claimed methods here.
- **Reproducibility Statement**: align commands, data assumptions, and artifact scope with repository reality.

### Track A Exit Criteria
- Manuscript text and figures contain no unsupported method/result claims.
- Every method claim is traceable to code in this repository.
- Reviewer-facing statement includes explicit synthetic-data caveats.

---

## Track B - Full claimed-system implementation

Goal: implement and validate the originally claimed stack for a stronger future submission.

### B1) Real LLM integration
- Status: **Partially addressed in code** (optional OpenAI-backed module + deterministic mock fallback + tests).
- Remaining: deeper runtime integration into full agent orchestration and broader prompt-eval coverage.

### B2) Kalman filter
- Status: **Partially addressed in code** (implemented simple 1D Kalman smoothing in cognitive-load predictions).
- Remaining: richer state models and dedicated benchmark protocol.

### B3) LSTM sequence model
- Status: **Partially addressed in code** (optional experimental `LSTMRegressor` when PyTorch is available).
- Remaining: end-to-end sequence data pipeline and comparative evaluation.

### B4) Isolation Forest anomaly detector
- Status: **Partially addressed in code** (Isolation Forest module with explicit fallback; integrated anomaly score into experiment outputs).
- Remaining: threshold calibration and false-positive analysis on real participant data.

### B5) LightGBM/sklearn GradientBoostingRegressor
- Status: **Partially addressed in code** (optional `GradientBoostingRegressor` enabled with explicit fallback and model-type reporting).
- Remaining: hyperparameter search, CV protocol, and robust calibration analysis.

### B6) Real experiment dataset pipeline
- Define ingestion, de-identification, split protocol, and data quality checks.
- Add participant metadata schema, provenance tracking, and versioned manifests.
- Enforce separation between synthetic and real-data experiment outputs.

### B7) Privacy/ethics documentation
- Add IRB/ethics process summary template (if applicable).
- Document consent model, retention schedule, and deletion workflow.
- Add risk assessment for LLM outputs and adaptive UI side effects.

### Track B Exit Criteria
- All claimed methods are implemented with tests.
- Real dataset pipeline and documentation are reproducible.
- Results are generated from participant-approved data with ethics/privacy coverage.

---

## Recommended sequencing
1. Complete Track A first to stabilize a truthful near-term resubmission.
2. Execute Track B in milestones, each with tests + documentation before manuscript claim updates.

---

## Immediate manuscript-safe route

- Remove unsupported human-study claims from present-tense results and abstract language.
- Present synthetic-demo and ablation outputs as engineering validation only (not participant efficacy evidence).
- Frame real user study outcomes as future work unless real participant data and approvals are collected and documented.
- Avoid unsupported percentage-improvement statements (including cognitive-load reduction claims) until reproducible participant evidence exists.
- Use `REPRODUCIBILITY_APPENDIX.md` as the claim-to-artifact source of truth for reviewer alignment.
