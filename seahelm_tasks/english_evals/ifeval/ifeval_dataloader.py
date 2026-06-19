from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class IFEvalDataloader(HuggingFaceDataloader):
    """Dataloader for the IFEval dataset.

    IFEval is a dataset for evaluating language models on their instruction-following capabilities.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the IFEval dataset from HuggingFace datasets.

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

            # perform edit for id 1174
            def correct_data(row):
                if row["key"] == 1174:
                    old_kwargs = row["kwargs"].copy()
                    old_kwargs[0]["let_relation"] = "at least"
                    return {"kwargs": old_kwargs}
                else:
                    return {"kwargs": row["kwargs"]}

            dataset = dataset.map(correct_data, num_proc=self.num_workers)

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row):
                return {
                    "id": row["key"],
                    "prompts": [{"text": row["prompt"]}],
                    "metadata": {
                        "language": "en",
                        "instruction_id_list": row["instruction_id_list"],
                    },
                    "kwargs": row["kwargs"],
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset
