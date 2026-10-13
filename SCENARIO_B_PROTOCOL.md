# Scenario B — Predeclared Experimental Protocol

## Research question
Under controlled semantic value aliases and header-name variations, how do an unchanged deterministic rule parser, an alias-aware deterministic parser, and Grok-assisted normalization compare on accuracy, time, token use, and service-equivalent cost?

## Arms
- A: frozen Scenario A parser, with no Scenario B alias tables.
- B: same deterministic parser with explicit field-name and semantic value-alias dictionaries.
- C: Grok 4.3 via xAI API. The prompt includes field meanings and canonical categorical vocabularies, but not the explicit lookup mappings.

## Data and ground truth
Use Kaggle competition train.csv after competition rules are accepted. Apply the same eligibility criteria as Scenario A. Sample eligible rows with a fixed seed and save canonical values before generating the altered input. The dataset provenance has not been independently verified.

## Perturbations
- Alternate names for canonical fields, such as age_years/customer_age and item_category/product_category.
- Value synonyms for categories, device, shipping and payment fields, chosen from fixed lists in app.py.
- The same reversible numeric, flag and order-ID formatting variations used in Scenario A.

All aliases are controlled and predeclared; they are not asserted to be observed errors in a real deployment.

## Metrics
Field-level accuracy; complete-record accuracy; elapsed processing time; records per second; input and output tokens; reference-priced token cost; missing/invalid output; API failures.

## Run order
1. Build a benchmark with seed 42 and 100 records.
2. Run frozen rules and alias-aware rules over all sampled records.
3. Run Grok for 20 records, review/export. Extend to 50 then 100 without rebuilding.
4. Repeat with multiple predeclared seeds to estimate variability.
5. Preserve each audit JSON and CSV privately.

## Interpretation guardrails
A better score for Grok than the frozen rule baseline does not by itself establish that the LLM is preferable to a well-maintained mapping table. Compare all three arms. Do not claim broad real-world generalization or scalability beyond the tested sizes. Local rule execution time and hosted API wall time are not a matched-hardware comparison.
