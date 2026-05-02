# Ethics and Privacy for User Studies

This project can support small pilot user studies, but real human-subject data collection must follow ethical and privacy safeguards.

## Behavioral Data That May Be Collected

Collect only data needed for the stated research question, such as:

- task completion times
- aggregate error counts
- coarse study condition labels (for example: baseline or adaptive)
- participant-reported satisfaction ratings
- optional non-identifying study notes

## Data That Should NOT Be Collected

Do not collect direct identifiers or sensitive personal data unless explicitly approved by an ethics process and legally required safeguards are in place. Avoid collecting:

- real names, personal email addresses, phone numbers, home addresses
- government identifiers
- passwords, authentication secrets, or private account content
- raw keystroke logs or free-text fields that can leak identity without strong safeguards
- unnecessary browser history or unrelated personal activity data

## Consent Requirement

Before collecting any real participant data:

- provide clear informed consent language
- explain what data is collected, why, and for how long
- explain withdrawal/deletion options where applicable
- collect explicit participant consent before recording study data

## Anonymization Guidance

- use pseudonymous participant IDs (for example `P001`, `P002`)
- store any re-identification key separately with restricted access
- remove or redact potentially identifying notes before sharing datasets
- report aggregate statistics whenever possible instead of row-level identifying detail

## IRB / Ethics Review Recommendation

Seek Institutional Review Board (IRB) or equivalent ethics approval before conducting any real human-subject study or publishing human-subject claims.

## Synthetic Data Disclaimer

Synthetic demo data is useful for pipeline validation only. It is not human-subject evidence and must not be presented as real participant outcomes.
