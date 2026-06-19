import ast

from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader

logger = get_logger(__name__)


class MMMUProDataloader(HuggingFaceImageDataloader):
    """Dataloader for the MMMU-Pro vision-language dataset."""

    def load_dataset(self, limit: int = None):
        """Load the MMMU-Pro dataset from HuggingFace datasets.

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
            subset_map = {
                "10_options": "standard (10 options)",
                "4_options": "standard (4 options)",
                "vision": "vision",
            }
            subset = subset_map[self.task_name.removeprefix("mmmu_pro_")]

            dataset = load_dataset(
                self.specific_task_config["filepath"], subset, split="test"
            )
            if limit is not None:
                dataset = dataset.select(range(limit))

            if subset == "vision":
                dataset = dataset.cast_column("image", Image(decode=False))
            else:
                for i in range(1, 8):
                    dataset = dataset.cast_column(f"image_{i}", Image(decode=False))

            # Decode images
            def map_columns(row):
                if subset == "vision":
                    return {
                        "id": row["id"],
                        "label": row["answer"],
                        "prompts": [{}],
                        "images": [row["image"]["bytes"]],
                        "metadata": {
                            "language": "en",
                            "subject": row["subject"],
                        },
                    }
                else:
                    return {
                        "id": row["id"],
                        "label": row["answer"],
                        "prompts": [
                            {
                                "question": row["question"],
                                "options": "\n".join(
                                    f"{i}: {option.strip()}"
                                    for i, option in zip(
                                        "ABCDEFGHIJ",
                                        ast.literal_eval(row["options"]),
                                        strict=False,
                                    )
                                ),
                            }
                        ],
                        "images": [
                            row[f"image_{i}"]["bytes"]
                            for i in range(1, 8)
                            if row[f"image_{i}"] is not None
                        ],
                        "metadata": {
                            "language": "en",
                            "difficulty": row["topic_difficulty"],
                            "subject": row["subject"],
                            "img_type": row["img_type"],
                            "explanation": row["explanation"],
                        },
                    }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset
