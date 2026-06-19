from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class ThaiExamDataloader(HuggingFaceDataloader):
    """Dataloader for the ThaiExam dataset.

    ThaiExam is a dataset for evaluating language models on Thai language exam questions.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the ThaiExam dataset from HuggingFace datasets.

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
            category = self.task_config.task_name.removeprefix(
                "thai_exam_"
            ).removesuffix("-logprobs")
            dataset = load_dataset(filepath, category, split="test")

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                return {
                    "id": index,
                    "label": row["answer"].upper(),
                    "prompts": [
                        {
                            "question": row["question"],
                            **{x: row[x] for x in "abcde" if x in row},
                        }
                    ],
                    "metadata": {
                        "language": "th",
                        "subject": row["subject"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
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
            category = self.task_config.task_name.removeprefix(
                "thai_exam_"
            ).removesuffix("-logprobs")
            dataset = load_dataset(example_filepath, category, split="test")

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                return {
                    "id": index,
                    "label": row["answer"].upper(),
                    "prompts": [
                        {
                            "question": row["question"],
                            **{x: row[x] for x in "abcde" if x in row},
                        }
                    ],
                    "metadata": {
                        "language": "th",
                        "subject": row["subject"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
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
