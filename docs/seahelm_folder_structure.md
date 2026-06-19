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
│   ├── code_generation/        # Code generation tasks
│   ├── cultural/
│   │   └── kalahi/
│   │       └── config.yaml     # YAML containing the task configuration
│   ├── instruction_following/
│   │   └── ifeval/
│   │       ├── config.yaml
│   │       ├── if_eval.py
│   │       └── instruction_checkers.py    # Python file containing the instruction checkers for each constraint
│   ├── knowledge/              # Knowledge-based tasks
│   ├── lindsea/                # LINDSEA linguistic tasks
│   ├── long_context/           # Long-context evaluation tasks
│   ├── multi_turn/
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
    │   ├── huggingface_dataloader.py
    │   ├── huggingface_image_dataloader.py
    │   ├── huggingface_audio_dataloader.py
    │   ├── seahelm_local_dataloader.py
    │   └── judges/             # Dataloaders for LLM judge inputs
    │       ├── criteria_dataloader.py
    │       ├── judge_dataloader.py
    │       └── pairwise_dataloader.py
    ├── inference_strategy/     # Model inference strategies
    │   ├── base_model_inference_strategy.py
    │   ├── batched_inference_strategy.py
    │   ├── default_inference_strategy.py
    │   ├── logprobs_inference_strategy.py
    │   └── utils.py
    ├── metrics/                # Metric calculation modules
    │   ├── seahelm_metric.py   # SeaHelmMetric base class
    │   ├── f1_acc_metric.py
    │   ├── logprob_metric.py
    │   ├── math_metric.py
    │   ├── question_answering.py
    │   └── llm_judges/         # LLM judge metric implementations
    │       ├── criteria_judge_metric.py
    │       ├── pairwise_llm_judge_metric.py
    │       └── pairwise_finegrained_llm_judge_metric.py
    ├── rouge_score/            # ROUGE metric implementation
    │   └── ...
    ├── sandbox/                # Sandboxed code execution for code-gen tasks
    │   ├── base_sandbox.py
    │   ├── docker_sandbox.py
    │   ├── enroot_sandbox.py
    │   ├── podman_sandbox.py
    │   └── singularity_sandbox.py
    └── serving/                # Model serving framework wrappers
        ├── __init__.py
        ├── batch/              # Batch API serving (remote providers)
        │   ├── __init__.py
        │   ├── base_batch_serving.py  # Base class for batch serving
        │   ├── anthropic_serving.py   # Anthropic batch API
        │   ├── openai_serving.py      # OpenAI batch API
        │   └── vertexai_serving.py    # VertexAI batch API
        ├── offline/            # Offline/local batch inference
        │   ├── __init__.py
        │   ├── base_offline_serving.py
        │   ├── vllm_serving.py         # vLLM offline serving
        │   ├── metricx_serving.py      # MetricX model serving
        │   ├── metricx_models.py       # MetricX model definitions
        │   └── openclip_serving.py     # OpenCLIP model serving
        └── online/             # Online/API-based serving
            ├── __init__.py
            ├── base_online_serving.py
            ├── litellm_serving.py      # LiteLLM unified API
            ├── local_openai_serving.py # Local OpenAI-compatible API
            ├── online_sglang_serving.py
            └── online_vllm_serving.py

```
