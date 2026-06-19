from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class SuperGPQADataloader(HuggingFaceDataloader):
    """Dataloader for the SuperGPQA dataset.

    SuperGPQA is a dataset for evaluating language models on google proof question answering tasks.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the SuperGPQA dataset from HuggingFace datasets.

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

            dataset = load_dataset(filepath, split="train")
            discipline = (
                self.task_name.removeprefix("supergpqa-")
                .removesuffix("-logprobs")
                .replace("-", " ")
            )
            dataset = dataset.filter(
                lambda example: example["discipline"].lower() == discipline
            )

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                options = "\n\n".join(
                    f"{i}: {option.strip()}"
                    for i, option in zip("ABCDEFGHIJ", row["options"], strict=False)
                )

                return {
                    "id": index,
                    "label": row["answer_letter"],
                    "prompts": [{"question": row["question"], "options": options}],
                    "metadata": {
                        "language": "en",
                        "discipline": row["discipline"],
                        "field": row["field"],
                        "subfield": row["subfield"],
                        "difficulty": row["difficulty"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Returns:
            datasets.Dataset: The loaded example dataset.
        """
        raise NotImplementedError("No example dataset provided for SuperGPQA")
