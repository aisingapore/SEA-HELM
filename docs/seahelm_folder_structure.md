# SEA-HELM Folder structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .github/
├── .gitignore
├── .markdownlint.yaml
├── .markdownlintignore
├── .pre-commit-config.yaml
├── .prettierignore
├── chat_templates/             # Folder with chat templates for different model types
│   ├── base_model.jinja        # Base model chat template
│   └── llama_template_wo_sys_prompt.jinja
├── docs/                       # Documentation folder
│   ├── README.md
│   ├── cli.md
│   ├── datasets_and_prompts.md
│   ├── new_task_guide.md
│   ├── score_calculations.md
│   ├── seahelm_folder_structure.md
│   ├── serving_models.md
│   └── assets/
├── elo/                        # ELO rating system for model comparison
│   ├── elo_outcomes.py
│   ├── elo_overrides_template.json
│   ├── elo_results_loader.py
│   ├── elo_runner.py
│   └── elo_utils.py
├── helpers/                    # Helper notebooks for analysis
│   ├── results_aggregation.ipynb
│   └── results_integrity_status.ipynb
├── logs/                       # Execution logs
├── results/                    # Evaluation results storage
├── run_evaluation.pbs          # PBS job scheduler script
├── run_evaluation.sh           # Shell script for running evaluation
├── run_evaluation.slurm        # SLURM job scheduler script
├── seahelm_tasks/              # Task definitions and data for SEA-HELM. Only a few notable examples are expanded.
│   ├── constants.yaml          # Constants for task execution order and judge models
│   ├── task_config.yaml        # Task configuration mapping
│   ├── reasoning_generation_config.yaml  # Config for reasoning models
│   ├── cultural/
│   │   └── kalahi/
│   │       ├── data/           # Folder containing the data for Kalahi
│   │       ├── config.yaml     # YAML containing the task configuration
│   │       └── kalahi.py       # Python file containing the metrics used in Kalahi
│   ├── instruction_following/
│   │   └── ifeval/
│   │       ├── data/
│   │       ├── config.yaml
│   │       ├── if_eval.py
│   │       └── instruction_checkers.py    # Python file containing the instruction checkers for each constraint
│   ├── knowledge/              # Knowledge-based tasks
│   ├── lindsea/                # LindSEA linguistic tasks
│   ├── multi_turn/
│   │   └── mt_bench/
│   │       ├── data/
│   │       ├── config.yaml
│   │       ├── mt_bench_prompts.py      # Python file containing the LLM-as-a-Judge prompts
│   │       └── mt_bench.py
│   ├── nlg/                    # Natural Language Generation tasks
│   ├── nlr/                    # Natural Language Reasoning tasks
│   ├── nlu/                    # Natural Language Understanding tasks
│   ├── safety/                 # Safety evaluation tasks
│   └── vision/                 # Vision-language tasks
└── src/                        # Source code modules
    ├── __init__.py
    ├── aggregate_metrics.py    # Metric aggregation logic
    ├── base_logger.py          # Logging utilities
    ├── collect_env.py          # Environment collection script
    ├── queue_manager.py        # Async task queue management
    ├── seahelm_evaluation.py   # Main SEA-HELM evaluation script
    ├── task_config.py          # Task configuration loader
    ├── utils.py                # Utility functions for SEA-HELM
    ├── dataloaders/            # Data loading abstractions
    │   ├── base_dataloader.py  # AbstractDataloader base class
    │   └── ...
    ├── judges/                 # LLM judge implementations
    │   └── ...
    ├── metrics/                # Metric calculation modules
    │   ├── seahelm_metric.py   # SeaHelmMetric base class
    │   └── ...
    ├── rouge_score/            # ROUGE metric implementation
    │   └── ...
    └── serving/                # Model serving framework wrappers
        ├── __init__.py
        ├── batch/              # Batch API serving (remote providers)
        │   ├── __init__.py
        │   ├── base_batch_serving.py  # Base class for batch serving
        │   ├── anthropic_serving.py   # Anthropic batch API
        │   ├── openai_serving.py      # OpenAI batch API
        │   └── vertexai_serving.py    # VertexAI batch API
        └── local/              # Local/online serving
            ├── __init__.py
            ├── base_serving.py         # BaseServing abstract class
            ├── vllm_serving.py         # vLLM local serving
            ├── litellm_serving.py      # LiteLLM unified API
            ├── local_openai_serving.py # Local OpenAI-compatible API
            ├── metricx_serving.py      # MetricX model serving
            ├── metricx_models.py       # MetricX model definitions
            └── openclip_serving.py     # OpenCLIP model serving

```
