# Agentic UX: Reproducibility-Aligned Implementation

## Project Status

This repository has been updated for reproducibility and manuscript-code alignment.
Current evaluation uses synthetic demo data.
Real user-study validation is not included yet.

## Implemented Components

- Reproducible experiment pipeline (`experiments/run_experiment.py`)
- Synthetic demo data generation (`--synthetic-demo`)
- Real Gradient Boosting model (`--model-type real_gb`, when sklearn backend is available)
- Kalman filter smoothing (`--use-kalman`)
- Isolation Forest anomaly detector (`--use-anomaly-detector`)
- Optional LLM decision layer (`--use-llm-agent`)
- Ablation study (`--run-ablation`)
- Figure generation (`experiments/generate_figures.py`)
- Table generation (`experiments/generate_tables.py`)
- User study template tooling (`experiments/user_study_template.py --demo-template`)

## Not Yet Implemented / Not Yet Validated

- Real participant dataset in this repository
- IRB-approved user study artifacts in this repository
- Real-world cognitive load reduction claims
- NASA-TLX / SUS / physiological validation from real participant data
- Production deployment claims

## Reproducibility Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run synthetic experiment (pipeline run)
python experiments/run_experiment.py --synthetic-demo --n-samples 500 --seed 42 --use-kalman --use-anomaly-detector --model-type auto --output-dir experiments/results_p3

# Run synthetic experiment with LLM decision layer
python experiments/run_experiment.py --synthetic-demo --n-samples 500 --seed 42 --use-kalman --use-anomaly-detector --use-llm-agent --model-type auto --output-dir experiments/results_phase2

# Run ablation study
python experiments/run_experiment.py --synthetic-demo --n-samples 500 --seed 42 --run-ablation --output-dir experiments/results_phase2

# Run statistical analysis
python experiments/statistical_analysis.py --results-csv experiments/results_phase2/results.csv --output-dir experiments/results_phase2

# Generate figures
python experiments/generate_figures.py --results-dir experiments/results_phase2 --output-dir experiments/figures

# Generate user study template (empty headers only)
python experiments/user_study_template.py --demo-template --output-dir experiments/user_study_results
```

## LLM Usage

- The LLM module is an interface adaptation decision layer.
- The LLM module does not directly improve regression prediction metrics.
- OpenAI mode requires `OPENAI_API_KEY` and the `openai` package.
- Mock mode is deterministic and labeled as mock output.

## Results Summary (from existing CSV outputs only)

### Model Performance

| Source | model_type | kalman_enabled | anomaly_enabled | llm_enabled | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|---:|---:|---:|---:|
| `experiments/results_p3/results.csv` | real_gb | True | True | False | 6.635687063348048 | 67.00187671515162 | 8.185467409693329 | -0.04598576583273539 |
| `experiments/results_phase2/results.csv` | real_gb | True | True | True | 6.635687063348048 | 67.00187671515162 | 8.185467409693329 | -0.04598576583273539 |

### Ablation Results (`experiments/results_p3/ablation_results.csv`)

| config_name | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| baseline (residual_linear) | 7.870473395688209 | 87.41419934516867 | 9.349556104177816 | -0.7233104178392378 |
| real_gb | 5.624972444576851 | 44.96611523564154 | 6.7056778356584905 | 0.11352417094896616 |
| +kalman | 6.297116177221474 | 58.470707346282815 | 7.646614109936686 | -0.1527095124488782 |
| +anomaly | 6.501181756748796 | 62.36884327978053 | 7.897394714700572 | -0.2295585634574473 |
| +llm | 6.410912513626985 | 60.637063319096086 | 7.786980372332788 | -0.19541772055081186 |

### Ablation Results (`experiments/results_phase2/ablation_results.csv`)

| config_name | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| baseline (residual_linear) | 7.373071796671868 | 81.94946041649422 | 9.052594126353739 | -0.27933683824623556 |
| real_gb | 5.300676043586282 | 42.68489575823288 | 6.533367872562579 | 0.3336336895609796 |
| +kalman | 6.578125104816931 | 65.77394506949253 | 8.110113751945315 | -0.02681616811780252 |
| +anomaly | 6.698198660968514 | 68.35316822228089 | 8.267597487921197 | -0.06708117018919668 |
| +llm | 6.636023706796224 | 67.17087379596231 | 8.195783903688671 | -0.048624028365989114 |

### LLM Decision Summary (`experiments/results_phase2/llm_decision_summary.csv`)

| config_name | simplify_ui | highlight_relevant | no_change | request_human_review | total_decisions |
|---|---:|---:|---:|---:|---:|
| +llm | 141 | 109 | 250 | 0 | 500 |

## Disclaimer

Synthetic demo results are not evidence of real-world user performance.
Unsupported manuscript claims should not be made unless backed by data.

## Reality and Planning Documents

- [Reality Check](REALITY_CHECK.md)
- [Engineering Roadmap](ROADMAP.md)
- [Reviewer Response Checklist](REVIEWER_RESPONSE_CHECKLIST.md)
- [Reproducibility Appendix](REPRODUCIBILITY_APPENDIX.md)
- [Data README](data/README.md)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{agentic-ux-2026,
  title={Agentic UX: Autonomous LLM Agents Reshaping Web Interface Architecture},
  author={Research Team},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Contributors

- Research Team
- Subhanjan Bikram K C — Student Research Assistant; reproducibility alignment, ML pipeline, ablation study, LLM decision layer, documentation support.
