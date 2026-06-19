from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class BBHDataloader(HuggingFaceDataloader):
    """Dataloader for the BBH dataset.

    BBH is a dataset for evaluating language models on big-bench hard tasks.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the BBH dataset from HuggingFace datasets.

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

            dataset = load_dataset(
                filepath,
                self.task_config.task_name.removeprefix("bbh_").removesuffix(
                    "-logprobs"
                ),
                split="eval",
            )

            if limit is not None:
                dataset = dataset.select(range(limit))

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
                example_filepath,
                self.task_config.task_name.removeprefix("bbh_").removesuffix(
                    "-logprobs"
                ),
                split="examples",
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
