# New Task Guide

## Folder setup

To create a new task, we will need to create a new folder. For example,

```bash
mkdir seahelm_tasks/<competency>/<task_name>
```

Each folder should minimally contain the following folders/files:

```text
<task_name>
├──data             # Folder containing the data for the task
│   └── ...
├──config.yaml      # YAML containing the task configuration
└──task_metric.py   # Python file containing the metrics used for the task
```

## Creating a new `config.yaml`

<details>
<summary>Example config file structure</summary>

```yaml
<task_name>:
  metadata:
    version: ...
    changes:
      <version>: ...
  name: ...
  competency: ...
  aggregation_group: ...
  dataloader_file: ...
  dataloader_class: ...
  metric_file: ...
  metric_class: ...
  metric: ...
  <additional kwargs>: ...
  use_judges: ...
  judge_file: ...
  judge_class: ...
  judge:
    judge_model_name: ...
    judge_model_type: ...
    batch_api_calls: ...
    judge_init_args:
      <kwargs>: ...
    judge_generation_kwargs:
      <kwargs>: ...
    judge_prompts:
      <template>: ...
  temperature: ...
  languages:
    <lang>:
      filepath: ...
      example_filepath: ...
      max_tokens: ...
      prompt_template:
        template: ...
        fewshot_example: ...
        fewshot_label: ...
```

</details>

### Basic task descriptions

The description of the tasks should be defined as follows.

```yaml
<task_name>:
  metadata:
    version: ... # version number of the task
    changes:
      <version>: # description of the changes made in the specified version
  name: ... # name of the task
  competency: ... # competency which the task should be under
  aggregation_group: ... # <optional> Set multiple tasks to the same aggregation group for the scores to be averaged together e.g. translation
```

> [!Note]  
> **Versioning of config/data**  
> To encourage transparency and reproducibility of the results, any changes to the config, data or metric calculation should be accompanied by an increase in the version number.
>
> Please also indicate in the metadata any changes that are made to the evaluation.

> [!tip]  
> **Multiple tasks in a single config file**  
> The config file allows for multiple tasks to be defined by specifying more than one `<task_name>` and filling in the various keys. For an example of this, check out the translation task

### Dataloader class

A custom dataloader can be specified as follows. The filepath to the dataloader should be relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>`.

```yaml
dataloader_file: ... # filepath to the dataloader python file
dataloader_class: ... # name of the dataloader class
```

### Metric class

The metrics can be specified as follows. The filepath to the metric should be relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>`.

```yaml
metric_file: ... # filepath to the metric python file
metric_class: ... # name of the metric class
metric: ... # metric used for the calculation of the aggregated scores
<additional kwargs>: ... # Additional kwargs that can be accessed by `metric_class`
```

> [!tip]  
> **Additional parameters requiried for the metric class**
>
> - Additional parameters can be defined in the config file and accessed by `metric_class`.
> - Examples of such parameters include: `null_label` for multiple choice tasks or `judge_models` for LLM-as-a-judge based tasks.

### Specifying Judge parameters

The judge model can be specified as follows. The filepath to the judge class hsould eb relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>`.

```yaml
use_judges: ... # set to true if the task uses a judge model
judge_file: ... # filepath to the judge python file
judge_class: ... # name of the judge class
judge:
  judge_model_name: ... # name of the judge model
  judge_model_type: ... # type of model serving to use
  batch_api_calls: ... # whether to use the Batch API or fall back to the default chat completion API
  judge_init_args:
    <kwargs>: ... # model initializaiton arguments to pass to the serving class
  judge_generation_kwargs:
    <kwargs>: ... # generation kwargs to pass to the judge
  judge_prompts:
    <template>: ... # prompt template use to format the judge prompt
```

> [!tip]  
> **Parameters to setup the judge model**
>
> - For commercial APIs that have a Batch API, set `batch_api_calls` to `true`. If not, set to `false`.
> - For more details on the arguments to specify in `judge_init_args`, see [docs/serving_models.md](docs/serving_models.md).
> - The `judge_prompts` template can take any form as long as the judge class is able to parse the template.

### Languages

For each language in each task, please fill in the following parts. Filepaths are relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>\<data>`.

```yaml
languages:
  <lang>: # Language. Current code only support the two letter isocodes (ID, TA, TH, VI, TL, ...)
    filepath: ... # filepath the test data. Can also be a huggingface dataset when paired with the huggingface dataloader.
    example_filepath: ... # filepath containing the fewshot examples
    max_tokens: ... # max number of tokens allowed for each generation
    prompt_template:
      preamble: ... # text blob that is placed before the task_template
      task_template: ... # task template
      answer_template: ... # template used for the answer of the model
      answer_tag: ... # tag used to find the answer of the model
```

- `preamble` and `task_template` should contain the actual prompt template used to format the data.
- `answer_template` should contain the expected model response.

> [!tip]  
> **Using the YAML block syntax for more readable prompt templates**  
> Please use the following syntax to use the block style
>
> ```yaml
> template: |-
>   text here
>   next line of text
> ```

<details>
<summary>Example `prompt_template` configuration (Indonesian sentiment task)</summary>

````yaml
prompt_template:
  preamble: |-
    Apa sentimen dari kalimat berikut ini? Gunakan salah satu dari pilihan di bawah ini: Positif, Negatif, atau Netral.

    Jawablah hanya dengan menggunakan format berikut ini:
    Jawaban: $OPTION
    Ganti $OPTION dengan jawaban yang telah dipilih.{fewshot_examples}
  task_template: |-
    Kalimat:
    ```
    {text}
    ```
    answer_template: |-
      {answer_tag} {label}
    answer_tag: "Jawaban:"
````

</details>

### Generation params

The generation parameters follows the default values determined by the model providers.

> [!note]  
> **Temperature and max token count for reasoning models**  
> The max token count is increased by the amount specified in `seahelm_tasks/reasoning_generation_config.yaml` (default is 20000). The max number of tokens is thus set to `20000 + max_tokens` where `max_tokens` is specified in the task config file.

## Writing the dataloader class

Available dataloaders:

1. `AbstractDataloader` (`src/dataloaders/base_dataloader.py`) - Base class that all dataloaders inherit from. Provides core functionality for loading data, formatting prompts, and managing few-shot examples.
1. `HuggingFaceDataloader` (`src/dataloaders/huggingface_dataloader.py`) - Loads datasets directly from Hugging Face Hub using dataset identifiers. This expects the dataloader to be formatted in the SEA-HELM data format.
1. `HuggingFaceImageDataloader` (`src/dataloaders/hugggingface_image_dataloader.py`) - Base class that extends the `AbstractDataloader` to allow for image datasets.
1. `SeaHelmLocalDataloader` (`src/dataloaders/seahelm_local_dataloader.py`) - Loads local datasets that have been formatted in the SEA-HELM data format.

If a new dataloader class is needed, please create the new dataloader class in the task folder and inherit from the base metric class `AbstractDataloader` or one of the dataloaders above. The dataloader should create a `dataset` class with at least the following columns:

- `id` or `question_id`: should be a `str` identifier
- `prompts`: should be a `list` of `dict` where each key-value pair is used to perform a string format for the prompt template
- `metadata`: should be a `dict` that contains the relevant metadata for each prompt

Refer to the respective metric classes for other columns that are needed e.g `label`, `baseline`, `reference`.

## Writing the metric class

For each task, please create a new metric class and inherit from the base metric class `SeaHelmMetric`.

### Overview for the evaluation of responses

The metrics are calculated using the following steps in the `evaluate_responses()` function:

1. (Either) Drop error responses using `drop_error_responses()` or replace error responses using `replace_error_responses()`
2. Postprocess of responses `postprocess_responses()`
   - Default post processing steps are to extract out the answer using regex
   - Strip out "$" signs at the start and end of the text
   - Strip out any leading and trailing white spaces
3. Calculate the counts of unique responses
4. Calculate the metric using `calculate_metrics()`

### Calculation of metrics

Please ensure that `calculate_metrics()` is defined in the new metric class. Output expected from the function is a dictionary containing the various metrics and the inference pandas dataframe.

```python
def calculate_metrics(self) -> tuple[dict, pd.DataFrame]:
    predictions = self.inference_df[self.postprocessed_response_column] # get the processed responses
    references = self.inference_df[self.label_column] # get the references from the label_column

    # perform metric calculations here
    metric_a = ...

    # perform score normalization of metrics
    # min is 0 for generative task and 1/n_options for multiple choice tasks
    normalized_metric_a = 100 * self.normalize_score(metric_a, min, max)

    # calculate number of null cases (for multiple choice tasks)
    null_count = ...

    metrics = {
      "normalized_metric_a": normalized_metric_a,
      "null_count": null_count
    }
    return metrics, self.inference_df
```
