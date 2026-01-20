# SEA-HELM Copilot Instructions

## Project Overview

SEA-HELM (SouthEast Asian Holistic Evaluation of Language Models) is a comprehensive LLM evaluation framework spanning 5 pillars: NLP Classics, LLM-specifics, SEA Linguistics, SEA Culture, and Safety. It evaluates models across Southeast Asian languages (ID, VI, TH, TA, TL, MS, MY, etc.) using a task-based architecture.

## Architecture

### Core Components

- **`seahelm_evaluation.py`**: Main orchestrator - loads tasks, manages inference, and calculates metrics
- **`seahelm_tasks/`**: Task definitions organized by competency (nlu, nlg, nlr, instruction_following, multi_turn, cultural, safety, etc.)
- **`src/dataloaders/`**: Data loading abstraction layer - all dataloaders inherit from `AbstractDataloader`
- **`src/metrics/`**: Metric calculation - all metrics inherit from `SeaHelmMetric`
- **`src/serving/`**: Model serving abstractions - supports vLLM, LiteLLM, OpenAI, Anthropic, VertexAI

### Task Configuration Pattern

1. **Task Structure**: Each task follows a standard structure:

   ```
   seahelm_tasks/<competency>/<task_name>/
   ├── data/                    # Test data files
   ├── config.yaml              # Task configuration
   ├── readme.md                # Task description & changelog
   └── <task_name>.py           # Metric implementation
   ```

2. **Dataloader Pattern**: All tasks inherit from `AbstractDataloader` in `src/dataloaders/base_dataloader.py`
3. **Metric Pattern**: All metrics inherit from `SeaHelmMetric` in `src/metrics/seahelm_metric.py`
4. **Serving Pattern**: All model backends inherit from `BaseServing` in `src/serving/base_serving.py`

**Critical**: `config.yaml` defines:

- `metric_file`: Path to metric implementation (relative to repo root)
- `metric_class`: Class name inheriting from `SeaHelmMetric`
- `metric`: Primary metric for aggregation
- `languages.<lang>.filepath`: Data file path
- `languages.<lang>.prompt_template`: Template with `{fewshot_examples}` placeholder
- `fewshot_num_examples.base` and `fewshot_num_examples.instruct`: Few-shot config
- `aggregation_group`: Groups sub-tasks (e.g., translation) for score averaging

### Model Serving Hierarchy

- `BaseServing` (abstract) → `generate()`, `batch_generate()`, `generate_responses()`
- `VLLMServing`: Offline inference via vLLM engine
- `LiteLLMServing`: API gateway (base for OpenAI, Anthropic, VertexAI)
  - Pass engine args via `--model_args "key=value,key2=value2"` (no spaces!)

**Model Serving Options** (`--model_type`):

- `vllm` - For local HF models (default, most common)
- `openai` - For OpenAI API
- `litellm` - For unified API access to multiple providers
- `anthropic` - For Anthropic API
- `vertexai` - For Google Vertex AI
- `none` - For testing without inference

## Common Workflows

### Running Evaluations

**IMPORTANT**: This project uses `uv` as the package manager and runtime. ALWAYS prefix Python commands with `uv run`:

```bash
# ✅ CORRECT - Always use uv run
uv run python seahelm_evaluation.py --tasks seahelm --output_dir results --model_type vllm --model_name <model>

# ❌ WRONG - Never use python directly
python seahelm_evaluation.py
```

```bash
# Standard vLLM run
uv run python seahelm_evaluation.py --tasks seahelm --output_dir results --model_type vllm \
  --model_name <model_path_or_hf_id> --model_args "tensor_parallel_size=1"

# Base model (skips MT-Bench, Kalahi, IF-Eval)
uv run python seahelm_evaluation.py --tasks seahelm --is_base_model ...

# Reasoning model (DeepSeek-style, temp=0.6)
uv run python seahelm_evaluation.py --tasks seahelm --is_reasoning_model ...
```

**Alternative Scripts**:

- `run_evaluation.sh` - Basic bash wrapper for local execution
- `run_evaluation.pbs` - PBS job scheduler script
- `run_evaluation.slurm` - SLURM job scheduler script

**Key flags**:

- `--tasks`: Task set from `seahelm_tasks/task_config.yaml` (e.g., `seahelm`, `english_evals`)
- `--is_base_model` - Use for base models (skips MT-Bench, Kalahi, IF-Eval). Applies generic base model chat template.
- `--is_reasoning_model` - For DeepSeek-style reasoning models (adds thinking tokens, sets temp=0.6)
- `--is_vision_model` - For vision-capable models (currently unused)
- `--rerun_cached_results` - Force rerun even if cached results exist
- `--limit <n>` - Limit examples per task (TESTING ONLY)

### Adding a New Task

1. Create folder: `seahelm_tasks/<competency>/<task_name>/`
2. Add `config.yaml` following existing patterns (see `instruction_following/ifeval/config.yaml`)
3. Create metric class inheriting `SeaHelmMetric` with `calculate_metrics()` method
4. Place data files in `data/` subdirectory
5. Update `seahelm_tasks/task_config.yaml` to include task in evaluation sets
6. **Version bumping**: Increment `metadata.version` in config for any data/metric changes

### Modifying Existing Tasks

- **ALWAYS** increment the `version` field in `config.yaml`
- Document changes in `readme.md`
- Update aggregation logic in `seahelm_tasks/aggregate_metrics.py` if needed

## Critical Conventions

### Task Execution Order

- **First**: Remote LLM judge tasks (`mt-bench`, `mental-health-safety`, `arena-hard-v2`) - sends async inference requests to remote judges while other evaluations run
- **Second**: Normal evaluation tasks (all non-judge tasks)
- **Third**: Local LLM judge tasks - tasks that use local models for evaluation (e.g., `translation` with `MetricX`, `xm3600` with `CLIP ViT-H/14 frozen xlm roberta large`)
- **Last**: Remote LLM judge evaluation collection - collects async results from remote judges

**Configuring Task Execution Order**:
Tasks are automatically categorized based on their configuration in `seahelm_tasks/constants.yaml`:

1. **Normal tasks** (default): No special configuration needed - all tasks run as normal evaluations by default

2. **Remote judge tasks**: Add task to `tasks_with_judge_models` list in `constants.yaml`, then in the task's `config.yaml` set:

   ```yaml
   judge_model_type: openai # or anthropic, vertexai, litellm
   ```

3. **Local judge tasks**: Add task to `tasks_with_judge_models` list in `constants.yaml`, then in the task's `config.yaml` set:
   ```yaml
   judge_model_type: vllm
   ```
   Note: If `judge_model_type` is not set, the tasks is categorized as a local judge task by default.

The `model_types` mapping in `constants.yaml` determines whether a judge type runs locally or remotely.

### Score Normalization & Aggregation

- **Normalization**: `(score - baseline) / (max - baseline) * 100`
  - Multiple choice baseline: `1/n_options`
  - Generative baseline: `0`
- **Aggregation hierarchy**:
  1. Sub-tasks → Task (via `aggregation_group`)
  2. Tasks → Competency average
  3. Competencies → Language score
  4. Languages → SEA Average

### Prompt Template Requirements

- Use YAML literal block style (`|-`) for readability:
  ```yaml
  prompt_template:
    preamble: |-
      This is the preamble.
    task_template: |-
      This is the task template with {fewshot_examples} if applicable.
  ```

### File Paths

- All paths in `config.yaml` are relative to repo root
- Inference results cached at: `results/<model>/inferences/<lang>/<task>/`
- Use `model_args` with comma-separated kwargs (no spaces!): `"key=value,key2=value2"`

## Environment Setup

**Initial Setup Steps**:

1. Install dependencies: The project uses `requirements.txt` (no `setup.py` or `pyproject.toml`)
2. Install pre-commit hooks: `uv run pre-commit install`
3. Set required environment variables:
   - `HF_TOKEN` - Required for gated Hugging Face models
   - `OPENAI_API_KEY` - Required for MT-Bench LLM-as-a-Judge evaluation
   - Optional: `ANTHROPIC_API_KEY`, `GCS_BUCKET_NAME`, `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`

## Testing

- Use `--limit 5` for quick validation before full runs
- Check `logs/` for detailed execution logs
- Cached results enable resumption of interrupted runs

## Special Cases

- **Reasoning models**: Number of tokens overridden by `seahelm_tasks/reasoning_generation_config.yaml` (default additional 20000 tokens)
- **Vision models**: Use `--is_vision_model` (currently experimental)
- **API models without tokenizers**: Auto-sets `--skip_tokenize_prompts` for VertexAI/Anthropic
- **Recalculate metrics only**: Set `--model_type none` (no inference, uses cached results)

## References

- Full CLI docs: `docs/cli.md`
- Task creation guide: `docs/new_task_guide.md`
- Model serving details: `docs/serving_models.md`
- Score calculations: `docs/score_calculations.md`
