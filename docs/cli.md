# Command Line Interface

Running the SEA-HELM evaluation from the command line can be done using this command.

> ```bash
> uv run seahelm_evaluation.py --tasks seahelm --output_dir <output_dir> --model_type vllm --model_name <model_name> --model_args "enable_prefix_caching=True,tensor_parallel_size=auto"
> ```

## Command Line Arguments

The following command line arguments are exposed to customize the running of the model evaluations:

- `--tasks`: Specifies which set of tasks to run. Default is `seahelm` which runs the tasks found in the seahelm suite.
  - You can specify multiple instances of this and the evaluation will run for all the tasks specified in the various instances.
  - Please see [seahelm_tasks/task_config.yaml](../seahelm_tasks/task_config.yaml) for the list of defined task sets.
- `--output_dir`: Path where the logs, evaluation config, results and inferences are stored for the particular evaluation run
- `--model_type`: Specifies the backend used to serve the model. Accepted values are [`vllm`, `litellm`, `openai`, `none`]. For more details on each model serving type, see [Serving Models](serving_models.md)
- `--model_name`: Name of model to run inference on. Can be a path to a model directory if `model_type` is `vllm`.
- `--model_args`: Comma-separated kwargs that are passed on the respective model serving class. Please ensure that there are no spaces between each kwargs. An example for vllm is as follows: `enable_prefix_caching=True,tensor_parallel_size=auto`
- `--batch-api-calls`: Whether to use the Batch APIs provided by API providers. Default is `True`.
- `--is_base_model`: Include this flag if the model is a base model. The model's chat template (if available) will be not be applied. (A generic base model chat template will this be applied to allow for ease of use with the inference frameworks. This template is equivalent to the case where no chat template is used). Some tasks are not run (e.g. MT-Bench, Kalahi).
- `--is_vision_model`: [Unused for now] Include this flag if the model is a vision model. Setups the chat template to accept vision and text inputs.
- `--is_reasoning_model`: Include this flag if the model is a reasoning model. Currently, this is setup for the DeepSeek family of models with the think tokens `<think>` and `<\think>`. A total of `20000` additional thinking tokens are allowed currently.
- `--rerun_cached_results`: Flag to specify if the inference results should be rerun if the cached results are found.
- `--skip_tokenize_prompts`: Skip the tokenization of the prompts.
- `--seed`: Random seed used for all operations.
- `--run`: Evaluation run number. Only accepts `int`.
- `--skip_tasks`: Comma separated list of tasks that should be skipped. Default is `None`.
- `--limit`: Limit the number of evaluation questions per task to run (only use this for testing). Only accepts `int`.
- `--no_batching`: Include this flag to turn off batching and run the prompts sequentially.
- `--ignore_missing_files`: Specify whether to ignore any missing inference files. Default is `False`
