from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class MMLUReduxDataloader(HuggingFaceDataloader):
    """Dataloader for the MMLUReduxDataloader"""

    def load_dataset(self, limit: int = None):
        """Load the MMLURedux dataset from HuggingFace datasets.

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
                "mmlu_redux_"
            ).removesuffix("-logprobs")
            dataset = load_dataset(filepath, category, split="test")

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM

            def map_columns(row, idx):
                letters = "ABCD"
                options = row["choices"]

                label = letters[row["answer"]]
                dropped = True
                if row["error_type"] == "ok":
                    dropped = False
                elif row["error_type"] == "wrong_groundtruth":
                    if row["correct_answer"] in [0, 1, 2, 3]:
                        label = letters[row["correct_answer"]]
                        dropped = False

                output_options = []
                for letter, option in zip(letters, options, strict=True):
                    output_options.append(f"{letter}. {option.strip()}")
                return {
                    "id": idx,
                    "label": label,
                    "prompts": [
                        {
                            "question": row["question"],
                            "options": "\n".join(output_options),
                        }
                    ],
                    "metadata": {
                        "language": "en",
                        "category": category,
                        "src": row["source"],
                    },
                    "dropped": dropped,
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )
            dataset = dataset.filter(
                lambda x: not x["dropped"], num_proc=self.num_workers
            )
            dataset = dataset.remove_columns("dropped")

            self.dataset = dataset

    def load_example_dataset(self, limit: int | None = None) -> None:
        """Load the example dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        raise NotImplementedError("No example dataset for MMLURedux")
