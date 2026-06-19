from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class MathArenaDataloader(HuggingFaceDataloader):
    """Dataloader for the MathArena datasets.

    MathArena is a collection of datasets for evaluating language models on various math problems.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the MathArena datasets from HuggingFace datasets.

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

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row):
                if "problem_idx" in row:
                    id = row["problem_idx"]
                else:
                    id = row["id"]

                return {
                    "id": id,
                    "label": str(row["answer"]),
                    "prompts": [{"problem": row["problem"]}],
                    "metadata": {
                        "language": "en",
                        "dataset": self.task_config.task_name,
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
            )
            self.dataset = dataset
