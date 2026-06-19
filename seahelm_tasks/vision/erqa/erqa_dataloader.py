import re

from datasets import Image, Sequence, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader

logger = get_logger(__name__)


class ERQADataloader(HuggingFaceImageDataloader):
    """Dataloader for the ERQA multilingual vision-language dataset."""

    def load_dataset(self, limit: int = None):
        """Load the ERQA dataset from HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples. Images are decoded from their stored format to PIL Images.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and image decoding.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            dataset = load_dataset(self.specific_task_config["filepath"], split="test")
            if limit is not None:
                dataset = dataset.select(range(limit))

            dataset = dataset.cast_column("images", Sequence(Image(decode=False)))
            dataset = dataset.rename_column("images", "image_list")

            # Decode images
            def map_columns(row):
                question = row["question"]
                question = question.removesuffix(
                    " Please answer directly with only the letter of the correct option and nothing else."
                )
                question = question.replace(" Choices:", "\nChoices:")
                question = re.sub(r" (A|B|C|D). ", r"\n\1.  ", question)

                return {
                    "id": row["question_id"],
                    "label": row["answer"],
                    "prompts": [{"question": question}],
                    "images": [
                        row["image_list"][i]["bytes"]
                        for i in range(len(row["image_list"]))
                    ],
                    "metadata": {
                        "language": "en",
                        "question_type": row["question_type"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset
