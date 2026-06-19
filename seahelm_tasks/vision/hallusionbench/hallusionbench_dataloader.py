from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader

logger = get_logger(__name__)


class HallusionBenchDataloader(HuggingFaceImageDataloader):
    """Dataloader for the HallusionBench multilingual vision-language dataset."""

    def load_dataset(self, limit: int = None):
        """Load the HallusionBench dataset from HuggingFace datasets.

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
            subset = self.task_name.removeprefix("hallusionbench_")
            if subset == "image":
                dataset = load_dataset(
                    self.specific_task_config["filepath"], split="image"
                )
            else:
                dataset = load_dataset(
                    self.specific_task_config["filepath"], split="non_image"
                )

            if limit is not None:
                dataset = dataset.select(range(limit))

            labels = ["FALSE", "TRUE"]
            if subset == "image":
                dataset = dataset.cast_column("image", Image(decode=False))

                # Decode images
                def map_columns(row, idx):
                    return {
                        "id": idx,
                        "label": labels[int(row["gt_answer"])],
                        "prompts": [{"question": row["question"]}],
                        "images": [row["image"]["bytes"]],
                        "metadata": {
                            "language": "en",
                            "category": row["category"],
                            "subcategory": row["subcategory"],
                        },
                    }
            else:

                def map_columns(row, idx):
                    return {
                        "id": idx,
                        "label": labels[int(row["gt_answer"])],
                        "prompts": [{"question": row["question"]}],
                        "images": [],
                        "metadata": {
                            "language": "en",
                            "category": row["category"],
                            "subcategory": row["subcategory"],
                        },
                    }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset
