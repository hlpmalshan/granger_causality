Project Conventions

This repository is an active research codebase. Scientific correctness is always more important than software elegance.

GENERAL PRINCIPLES

- Preserve reproducibility.
- Make the smallest change necessary to implement a new idea.
- Do not perform unrelated refactoring.
- Prefer extending the existing architecture rather than redesigning it.
- When uncertain about a design decision, ask before making major structural changes.

--------------------------------------------------

EXPERIMENTS

The experiments directory documents the chronological evolution of the research.

Treat completed experiments as historical records.

Do not modify completed experiments unless explicitly instructed.

When implementing a new idea:

- identify the closest previous experiment
- copy that experiment
- implement only the new methodological change
- preserve existing outputs whenever possible

Experiment numbering is chronological and should remain sequential.

--------------------------------------------------

SOURCE CODE

Reusable algorithms belong in src/.

Experiment scripts should primarily:

- define parameters
- generate data
- call reusable functions
- summarize results
- save outputs

Avoid placing large reusable algorithms directly inside experiment scripts.

If functionality will likely be reused by future experiments, move it into src/.

--------------------------------------------------

BACKWARD COMPATIBILITY

Avoid breaking previous experiments.

Do not rename functions that are already used by multiple experiments unless explicitly requested.

Avoid changing public function signatures unnecessarily.

Preserve existing imports whenever possible.

--------------------------------------------------

OUTPUTS

Preserve existing output formats.

CSV column names should remain unchanged unless explicitly requested.

Existing plotting behavior should remain unchanged unless the experiment specifically studies visualization.

Continue writing outputs to the results directory.

--------------------------------------------------

REPRODUCIBILITY

Preserve random seeds.

Avoid introducing nondeterministic behavior.

Do not silently change simulation parameters.

Do not silently change Monte Carlo settings.

--------------------------------------------------

IMPLEMENTATION STYLE

Prefer readable code over clever code.

Prefer small helper functions.

Avoid deeply nested logic.

Reuse existing utilities before creating new ones.

Avoid duplicated implementations.

--------------------------------------------------

DEBUGGING

When fixing bugs:

first identify the underlying cause

do not suppress warnings simply to remove errors

preserve scientific meaning

explain the cause of the bug before proposing a fix whenever possible

--------------------------------------------------

SCIENTIFIC WORKFLOW

Every experiment should test one primary research hypothesis.

Avoid combining multiple methodological changes into one experiment.

When implementing a new estimator:

preserve the previous estimator

implement the new estimator separately

allow fair comparison between methods

--------------------------------------------------

VALIDATION

Whenever practical:

compare new results against the previous experiment

reuse existing evaluation metrics

preserve Monte Carlo summaries

preserve statistical comparisons

--------------------------------------------------

PREFERRED WORKFLOW

Research discussion happens outside this repository.

This repository implements those research decisions.

Avoid introducing new research ideas unless explicitly instructed.

If requirements are ambiguous, ask for clarification instead of making scientific assumptions.

When making changes:

High confidence:
- fixing syntax
- imports
- type errors
- refactoring
- logging

Medium confidence:
- implementing algorithms explicitly requested

Low confidence:
- changing statistical methodology
- changing mathematical formulas
- changing evaluation metrics
- changing Monte Carlo settings

For low-confidence changes, explain the proposed modification before implementing it.

## Experiment Pattern

Most experiments follow the same high-level workflow.

1. Define experiment parameters.
2. Generate simulated or observed data.
3. Fit the appropriate model.
4. Compute Granger causality or related statistics.
5. Perform statistical inference.
6. Evaluate performance using Monte Carlo simulations.
7. Save numerical summaries.
8. Save plots.
9. Compare with previous experiments.

Not every experiment contains every stage, but new experiments should preserve this general structure whenever appropriate.

Each new experiment should introduce one primary methodological change while preserving as much of the surrounding workflow as possible.

## Research Progression

Every experiment should answer one scientific question.

When creating a new experiment:

- identify the hypothesis being tested
- preserve previous methodology whenever possible
- modify only the components necessary to test the new hypothesis
- ensure that comparisons with previous experiments remain scientifically meaningful

Scientific comparability is more important than minimizing code duplication.

Experiments are not only executable scripts.

They also serve as:

- research records,
- validation studies,
- reproducible benchmarks,
- documentation of methodological evolution.

Preserve readability and reproducibility even if this results in some duplication between experiments.