from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader

logger = get_logger(__name__)


class MMStarDataloader(HuggingFaceImageDataloader):
    """Dataloader for the MMStar multilingual vision-language dataset."""

    def load_dataset(self, limit: int = None):
        """Load the MMStar dataset from HuggingFace datasets.

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
            dataset = load_dataset(self.specific_task_config["filepath"], split="val")
            if limit is not None:
                dataset = dataset.select(range(limit))

            dataset = dataset.cast_column("image", Image(decode=False))

            # Decode images
            def map_columns(row):
                return {
                    "id": row["index"],
                    "label": row["answer"],
                    "prompts": [{"question": row["question"]}],
                    "images": [row["image"]["bytes"]],
                    "metadata": {
                        "language": "en",
                        "l2_category": row["l2_category"],
                        **row["meta_info"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset
