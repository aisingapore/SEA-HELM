from datasets import Sequence, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class TranslationDataloader(HuggingFaceDataloader):
    """Dataloader for HuggingFace translation datasets."""

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
        num_workers: int = 16,
    ):
        """Initialize the TranslationDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples (int): Default number of few-shot examples to use.
            is_base_model (bool): Whether this is a base model (vs instruction-tuned).
            model_name (str, optional): Name/path of the model being evaluated. Defaults to "".
            run_base_path (str, optional): Base path for storing inference results. Defaults to "".
            inference_file_type (str, optional): File format for inference results ('jsonl' or 'csv'). Defaults to "jsonl".
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
            num_workers=num_workers,
        )

        self.dataset_map = {
            "translation-en-xx": "en_to_{lang}",
            "translation-xx-en": "{lang}_to_en",
            "translation-id-xx": "id_to_{lang}",
            "translation-xx-id": "{lang}_to_id",
        }

    def load_dataset(self, limit: int | None = None) -> None:
        """Load the HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.

        Note:
            The dataset is stored in self.dataset after loading.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            dataset_split = self.dataset_map[self.task_name].format(lang=self.lang)
            self.dataset = load_dataset(
                self.specific_task_config["filepath"], dataset_split, split="eval"
            )
            if limit is not None:
                self.dataset = self.dataset.select(range(limit))

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

            dataset_split = self.dataset_map[self.task_name].format(lang=self.lang)
            self.example_dataset = load_dataset(
                example_filepath, dataset_split, split="examples"
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

            # check if label is of type list and convert it to string
            if isinstance(self.example_dataset.features["label"], Sequence):
                self.example_dataset = self.example_dataset.map(
                    lambda x: {"label": x["label"][0]}, num_proc=16
                )
