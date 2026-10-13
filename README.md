# ICITIIT 2027 — Scenario B ETL Benchmark

**Research title:** *AI-Augmented Marketing Analytics: Evaluating Large Language Models in ETL Pipelines for Accuracy, Cost, and Scalability in Small-Scale Deployments*

This is a separate, Scenario B experimental app. Scenario A and its completed audit exports remain in the original repository:
https://github.com/Anish0605/LLM-Experiment-Accuracy-Cost-Scalability

## Research purpose

Scenario A tested known, reversible formatting changes, where both the rule pipeline and Grok achieved 100% accuracy across 20, 50 and 100 records. Scenario B broadens the task to controlled semantic category aliases and alternate column names.

Scenario B compares three arms:

1. **Frozen rules:** the Scenario A parsing logic without new header/value mappings.
2. **Alias-aware rules:** deterministic parsing plus explicit, predeclared alias dictionaries.
3. **Grok-assisted:** Grok receives canonical field meanings and canonical vocabularies, but not the alias lookup table.

The second rule-based arm is intentional: a frozen baseline's failure on new inputs must not be treated as proof that an LLM is better if a simple lookup table solves the task. The aliases are controlled synthetic perturbations, not a claim about error frequencies in real production data.

## Dataset

- Kaggle competition: https://www.kaggle.com/competitions/retail-return-risk-modeling
- File: `train.csv`
- The app uses the Kaggle API for a direct download; accept competition terms first.
- The dataset origin and real-world provenance are not independently verified here.
- The eligibility rules mirror the Scenario A app to aid comparison. This filters records to a defined subset; document this in the paper.

## Deploy on Streamlit Community Cloud

Entry point: `app.py`.

Add these values under Streamlit app **Settings → Secrets** (use the actual keys; never paste them into GitHub):

```toml
XAI_API_KEY = "your-xai-api-key"
XAI_MODEL = "grok-4.3"
KAGGLE_USERNAME = "anish6"
KAGGLE_KEY = "your-kaggle-legacy-key"
```

Keep the Kaggle key from the legacy `kaggle.json` fields expected by the app. Keep every secret out of the repository. Use a fresh legacy key if Kaggle returns 401.

## Run procedure

1. Download or upload `train.csv`.
2. Inspect eligibility rules, then build a reproducible benchmark (start with 100 records and seed 42).
3. Run frozen rules and alias-aware rules.
4. Run Grok first on 20 records, inspect results/cost/errors, then extend to 50 and 100.
5. Export audit JSON and record-level CSV after completed runs. Save these privately, outside public GitHub.
6. Repeat with multiple predeclared random seeds for stronger evidence.

Do not rebuild the benchmark between 20/50/100 extensions; a rebuild clears session outputs.

## Measures

Field accuracy, full-record accuracy, observed runtime, records/second, Grok token counts, reference-priced service-equivalent cost, and API errors. Rule token cost is zero, but these metrics do not include engineering or maintenance effort. Timings compare local code with hosted API latency and are not a matched-hardware comparison.

## Reproducibility and limitations

- All aliases and header variants are listed in the audit export.
- Ground truth is frozen before perturbations.
- Scenario B tests only the defined controlled aliases/schema variations; do not generalize its results to all ETL.
- Grok's API price can change; verify the current price and actual invoice/credit consumption before publication.
- A single seed is exploratory. Report multiple runs and API errors.
- Never report generated results before a successful run or reuse Scenario A's metrics as Scenario B results.

## Dependencies

See `requirements.txt`; includes Streamlit, Pandas, NumPy, Plotly, the OpenAI-compatible xAI SDK, and Kaggle CLI.
