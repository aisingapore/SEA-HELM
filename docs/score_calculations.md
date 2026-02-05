# Score calculations

## Evaluation Parameters

SEA-HELM evaluations are conducted using the following parameters:

- **Number of evaluation runs**: 8 independent runs per model
- **Number of bootstrap runs**: 30 bootstrapped runs that are drawn from the 8 independent runs per model
- **Generation parameters**: We use the model-specific defaults when available in the model configurations. For any unspecified parameters, we apply the vLLM default settings.

> [!Note]  
> All prompts in SEA-HELM are presented in their native languages using zero-shot prompting for instruct/reasoning models and five-shot prompting for base models.

## Normalization process

Score normalization is done to account for different difficulties and random baseline scores for each task.

The calculation of the normalized scores is:

```math
\text{normalised\_score} = (\text{score} - \text{baseline}) / (\text{maximum} - \text{baseline}) * 100
```

where $\text{baseline}$ is equal to:

- Multiple choice tasks: $`1/\text{n\_options}`$
- Generative tasks: $0$

and $\text{maximum}$ is equal to:

- maximum possible score that an answer can get (typically $1$)

## Aggregation process

Each task in SEA-HELM is grouped into one of the following competencies - NLU, NLG, NLR, Instruction-Following, Multi-Turn, Cultural, Safety.

Our scoring system follows a hierarchical approach, aggregating results from individual tasks up to the overall SEA score:

### 📋 Task Level

Individual task scores are calculated as the mean across all 8 evaluation runs. Standard errors use the clustered standard error methodology as detailed in [Miller (2024)](https://arxiv.org/abs/2411.00640).

> [!Note]  
> **Sub-task aggregation**  
> For tasks with sub-tasks (e.g. translation - translation-xx-en and translation-en-xx), the scores for each sub-task are first averaged to get the task score. This task score is then used in the calculation of the competency score.

### 🎯 Competency Level

For each evaluation run, we calculate competency scores by averaging all task scores within that competency area. The final competency score and its standard error are derived from the mean and standard error of these per-run scores.

Example:

> Competency: NLR  
> Tasks to average: NLI, Causal

### 🌏 Language Level

Language scores aggregate all competency scores available for that specific language, calculated using the same approach as the competency-level aggregation.

Example:

> Language: ID  
> Competencies to average: NLU, NLG, NLR, Instruction-Following, Multi-Turn, Safety

### 🏆 SEA Level (Overall Score)

The SEA score represents the performance across all Southeast Asian languages and is calculated as the aggregate of the individual language scores with their respective standard errors. This calculation follows the same approach as the competency-level aggregation.

Example:

> SEA Average  
> Languages to average: FIL, ID, TA, TH, VI
