# Data Inputs for Experiments

## Expected Input Schema (`--input-csv`)

The experiment runner expects one row per sample with these required columns:

- `mouse_velocity` (float)
- `click_frequency` (float)
- `time_between_actions` (float)
- `error_count` (int)
- `correction_count` (int)
- `page_visits` (int)
- `heart_rate` (float)
- `pupil_dilation` (float)
- `blink_rate` (float)
- `task_complexity` (float, 0-1)
- `task_familiarity` (float, 0-1)
- `time_pressure` (float, 0-1)
- `element_density` (float, 0-1)
- `color_complexity` (float, 0-1)
- `cognitive_load` (float target label, 0-100)

## Synthetic Demo Mode

Use `--synthetic-demo` to generate deterministic synthetic data for pipeline checks and local reproducibility. The generated rows are explicitly labeled as synthetic in output files (`data_source` and `is_synthetic_demo` fields).

This mode is for demonstration and testing only.

## Warning About Claims

Synthetic demo data is **not evidence** for real-world user outcomes and must not be presented as participant-derived findings.

## Adding Real Datasets Safely

- Store private participant data outside version control.
- Add local file paths through `--input-csv` when running experiments.
- Keep sensitive files out of commits (for example via `.gitignore` and secure storage practices).
- Share only de-identified, permission-approved aggregates when publishing artifacts.
