# Score calculations

## Evaluation Parameters

SEA-HELM evaluations are conducted using the following parameters:

- **Number of evaluation runs**: 8 independent runs per model
- **Number of bootstrap replicates**: 2,000 replicates per task, drawn by resampling the _questions_ of that task (see [Uncertainty estimation](#uncertainty-estimation))
- **Reported interval**: the 2.5th and 97.5th percentiles of the bootstrap distribution, i.e. a 95% percentile bootstrap interval
- **Generation parameters**: We use the model-specific defaults when available in the model configurations. For any unspecified parameters, we apply the vLLM default settings.

> [!Note]  
> All prompts in SEA-HELM are presented in their native languages using zero-shot prompting for instruct/reasoning models and five-shot prompting for base models.

Aggregation is implemented in [src/aggregation/](../src/aggregation/) and driven by `MultiRunAggregator`
(see [helpers/results_aggregation.ipynb](../helpers/results_aggregation.ipynb)). The bootstrap is seeded
(`seed=94370244` by default), so re-aggregating the same result folder reproduces the same score file
byte for byte.

## Normalization process

Score normalization is done to account for the different score scales, difficulties and random baseline
scores of each task. It happens in two steps.

**1. Put the per-question scores on a 0–1 scale.** Each task declares its metric in its `config.yaml`,
and each metric declares its native maximum in
[`METRIC_NATIVE_MAX`](../src/aggregation/constants.py) — `1.0` for a metric stored as a fraction
(e.g. `accuracy_score`) and `100.0` for one already stored as a percentage (e.g.
`normalized_rougel_f1`). The per-question scores are divided by that native maximum.

> [!Note]  
> The native scale is looked up by metric name and is never inferred from the observed values. The
> earlier "multiply by 100 if the largest score is ≤ 1" heuristic could flip scale between bootstrap
> replicates, mixing 1× and 100× replicates inside a single task. A metric with no declared scale is a
> hard error rather than a guess, since a wrong guess silently rescales a whole task by 100×.

**2. Normalize against the random baseline:**

```math
\text{normalised\_score} = \max\left(\frac{\text{score} - \text{baseline}}{\text{maximum} - \text{baseline}},\ 0\right) * 100
```

where $\text{baseline}$ is equal to:

- Multiple choice tasks: $`1/\text{n\_classes}`$, where $`\text{n\_classes}`$ is the number of distinct
  gold labels of the task
- Generative tasks: $0$

and $\text{maximum}$ is equal to:

- $1$, the maximum possible score after the rescaling in step 1

Normalized scores are floored at $0$, so a model scoring at or below random chance reports $0$.

## Uncertainty estimation

Intervals come from a bootstrap over the **questions** of each task, run once per task and then rolled
up through the score tree.

### 1. Averaging per prompt score

Each prompts' score is the mean of the scores across the 8 runs.

### 2. Resample the questions, 2,000 times

- **Generative tasks**: prompts are resampled i.i.d. with replacement.
- **Multiple choice tasks**: positions are resampled with replacement _within each class_, so every
  replicate keeps the original class supports.

### 3. Calculate mean/CIs

Each replicate yields one normalized 0–100 score for the task. The task reports the `mean` of those
2,000 values and the `ci` at the 2.5th/97.5th percentiles.

## Balanced accuracy for multiple choice tasks

Multiple choice tasks are scored with a **probability-aware balanced accuracy**
([src/aggregation/probabilistic_accuracy.py](../src/aggregation/probabilistic_accuracy.py)).

Each question contributes the probability it placed on the correct class — its mean correctness across
the 8 runs, or its cumulative probability for a logprob task — instead of a 0/1 outcome. The per-class
score becomes an expected recall, and balanced accuracy remains their unweighted mean:

```math
\text{recall}_i = \frac{1}{|\{n : y_n = i\}|}\sum_{n : y_n = i} P[n, i]
\qquad
\text{balanced\_accuracy} = \frac{1}{\text{n\_classes}}\sum_i \text{recall}_i
```

Feeding one-hot probabilities reproduces `sklearn.metrics.balanced_accuracy_score` exactly, so this is a
strict generalisation of it.

Pooling every run's responses into one flat list and scoring that with
`sklearn.metrics.balanced_accuracy_score` gives an identical number. Balanced accuracy is linear in the
per-question outcome and every run shares the same class supports, so averaging the runs first changes
nothing. What it does change is the input to the bootstrap: one score per question rather than 8
correlated rows, so the interval reflects the sampling of questions rather than the repetition of runs.

The formulation also accepts probabilities that no count of runs can express. Logprob tasks are therefore
scored straight from the model's own probability for the correct answer, with no need to threshold it
into a hard prediction first.

The resulting balanced accuracy is normalized against a baseline of $1/\text{n\_classes}$ as described
above.

## Aggregation process

Each task in SEA-HELM is grouped into one of the following competencies - NLU, NLG, NLR,
Instruction-Following, Multi-Turn, Cultural, Safety.

Our scoring system follows a hierarchical approach, aggregating results from individual tasks up to the
overall SEA score. The aggregation operates on the **bootstrap replicates**.

### 📋 Task Level

Individual task scores are the mean of the task's 2,000 bootstrap replicates, each computed from the
per-prompt scores averaged over the 8 evaluation runs.

> [!Note]  
> **Sub-task aggregation**  
> For tasks with sub-tasks (e.g. translation - translation-xx-en and translation-en-xx), the sub-tasks
> declare a shared `aggregation_group` in their config. Each sub-task is still scored and reported on
> its own, but is marked `ignore: true` so that it does not enter the competency mean twice: the group's
> score is the unweighted mean of its member sub-tasks, and it is that group score which the competency
> averages.

### 🎯 Competency Level

Competency scores are the unweighted mean of the task scores (and aggregation-group scores) within that
competency area, taken replicate by replicate.

Example:

> Competency: NLR  
> Tasks to average: NLI, Causal

### 🌏 Language Level

Language scores aggregate all competency scores available for that specific language, calculated using
the same approach as the competency-level aggregation.

Example:

> Language: ID  
> Competencies to average: NLU, NLG, NLR, Instruction-Following, Multi-Turn, Safety

### 🏆 SEA Level (Overall Score)

Two top-level aggregates are reported, both averaging the language scores replicate by replicate:

- `mean` / `ci`: over every language present in the results folder.
- `sea_mean` / `sea_ci`: over the Southeast Asian subset only — `id`, `vi`, `th`, `ta`, `tl`, `ms`, `my`
  (see [`SEA_LANGUAGES`](../src/aggregation/constants.py)).

Example:

> SEA Average  
> Languages to average: FIL, ID, MS, MY, TA, TH, VI

## Score file contents

Aggregation writes `<model>/<model>_scores.yaml`. Every node of the tree — task, aggregation group,
competency, language, and the two top-level aggregates — carries:

| Key    | Meaning                                                   |
| ------ | --------------------------------------------------------- |
| `mean` | Mean of the node's bootstrap replicates, on a 0–100 scale |
| `ci`   | `[low, high]` percentile interval of the same replicates  |

with the following annotations where they apply:

| Key                                      | Meaning                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| `is_incomplete`                          | The task is missing at least one run (also set at the root of the tree)                   |
| `low_class_support`                      | Per-class question counts of a multiple choice task with a class below 10 questions       |
| `ignore`, `aggregation_group`, `remarks` | The task reaches its competency through the named aggregation group instead of on its own |
