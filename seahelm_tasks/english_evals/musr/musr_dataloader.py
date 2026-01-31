import ast

from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class MuSRDataloader(HuggingFaceDataloader):
    """Dataloader for the MuSR dataset.

    MuSR is a dataset for evaluating language models on multistep reasoning tasks.
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
        """Initialize the MuSRDataloader.

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
        self.seed = 976843

    def load_dataset(self, limit: int = None):
        """Load the MuSR dataset from HuggingFace datasets.

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

            category = self.task_config.task_name.removeprefix("musr_").removesuffix(
                "-logprobs"
            )
            dataset = load_dataset(filepath, split=category)

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                choices = [
                    f"{i + 1} - {choice}"
                    for i, choice in enumerate(ast.literal_eval(row["choices"]))
                ]

                return {
                    "id": index,
                    "label": str(row["answer_index"] + 1),
                    "prompts": [
                        {
                            "narrative": row["narrative"],
                            "question": row["question"],
                            "choices": "\n".join(choices),
                        }
                    ],
                    "metadata": {
                        "language": "en",
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Returns:
            datasets.Dataset: The loaded example dataset.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)

            self.example_dataset = load_dataset(
                "json", split="train", data_files=example_filepath
            )

            if limit is not None:
                if len(self.example_dataset) < limit:
                    logger.warning(
                        "Not enough examples! Expected %d examples but only received %d.",
                        limit,
                        len(self.example_dataset),
                    )
                    limit = len(self.example_dataset)
                self.example_dataset = self.example_dataset.select(range(limit))
