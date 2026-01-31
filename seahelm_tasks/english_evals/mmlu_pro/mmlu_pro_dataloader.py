from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class MMLUProDataloader(HuggingFaceDataloader):
    """Dataloader for the MMLUProDataloader table parsing dataset.

    MMLUPro is a dataset for evaluating language models on a variety of professional and academic subjects.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ):
        """Initialize the MMLUProDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )

    def load_dataset(self, limit: int = None):
        """Load the MMLUPro dataset from HuggingFace datasets.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and image decoding.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]
            logger.info("Drawing and preparing data from %s", filepath)
            dataset = load_dataset(filepath, split="test")

            if limit is not None:
                dataset = dataset.select(range(limit))

            category = self.task_config.task_name.removeprefix(
                "mmlu_pro_"
            ).removesuffix("-logprobs")
            category = category.replace("_", " ")
            dataset = dataset.filter(lambda x: x["category"] == category)

            # Map dataset fields to standard fields used in SEA-HELM

            def map_columns(row):
                letters = "ABCDEFGHIJ"
                options = row["options"]

                output_options = []
                for letter, option in zip(letters, options, strict=False):
                    output_options.append(f"{letter}. {option.strip()}")
                return {
                    "id": row["question_id"],
                    "label": row["answer"],
                    "prompts": [
                        {
                            "question": row["question"],
                            "options": "\n".join(output_options),
                        }
                    ],
                    "metadata": {
                        "language": "en",
                        "category": row["category"],
                        "src": row["src"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset

    def load_example_dataset(self, limit: int | None = None) -> None:
        """Load the example dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)
            dataset = load_dataset(example_filepath, split="validation")

            if limit is not None:
                dataset = dataset.select(range(limit))

            category = self.task_config.task_name.removeprefix(
                "mmlu_pro_"
            ).removesuffix("-logprobs")
            category = category.replace("_", " ")
            dataset = dataset.filter(lambda x: x["category"] == category)

            # Map dataset fields to standard fields used in SEA-HELM

            def map_columns(row):
                letters = "ABCDEFGHIJ"
                options = row["options"]

                output_options = []
                for letter, option in zip(letters, options, strict=False):
                    output_options.append(f"{letter}. {option.strip()}")
                return {
                    "id": row["question_id"],
                    "label": row["answer"],
                    "prompts": [
                        {
                            "question": row["question"],
                            "options": "\n".join(output_options),
                            "cot_content": row["cot_content"],
                        }
                    ],
                }

            dataset = dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=dataset.column_names,
            )

            self.example_dataset = dataset

            if limit is not None:
                if len(self.example_dataset) < limit:
                    logger.warning(
                        "Not enough examples! Expected %d examples but only received %d.",
                        limit,
                        len(self.example_dataset),
                    )
                    limit = len(self.example_dataset)
                self.example_dataset = self.example_dataset.select(range(limit))
