from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class CVQADataloader(HuggingFaceImageDataloader):
    """Dataloader for the CVQA (Cultural Visual Question Answering) dataset.

    CVQA is a multilingual visual question answering dataset that evaluates
    cultural understanding through multiple-choice questions about images.
    This dataloader handles loading images and preparing them for vision-language
    model inference.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ):
        """Initialize the CVQADataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )

        self.language_map = {
            "id": "('Indonesian', 'Indonesia')",
            "ms": "('Malay', 'Malaysia')",
            "tl": "('Filipino', 'Philippines')",
            "zh": "('Chinese', 'Singapore')",
            "ta": "('Tamil', 'India')",
            "jv": "('Javanese', 'Indonesia')",
            "su": "('Sundanese', 'Indonesia')",
        }

    def load_dataset(self, limit: int = None):
        """Load the CVQA dataset from HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples. Each sample contains a question, four multiple-choice options
        (A, B, C, D), and an associated image.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading. Labels are converted
            from numeric format to letter format (A, B, C, D).
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            dataset = load_dataset(self.specific_task_config["filepath"])["test"]
            dataset = dataset.filter(
                lambda example: example["Subset"] == self.language_map[self.lang],
                num_proc=self.num_workers,
            )
            dataset = dataset.cast_column("image", Image(decode=False))

            if limit is not None:
                dataset = dataset.select(range(limit))

            dataset = dataset.map(
                lambda row: {
                    "id": row["ID"],
                    "label": "ABCD"[row["Label"]],
                    "prompts": [
                        {
                            "question": row["Question"],
                            "option_a": row["Options"][0],
                            "option_b": row["Options"][1],
                            "option_c": row["Options"][2],
                            "option_d": row["Options"][3],
                        }
                    ],
                    "images": [row["image"]["bytes"]],
                    "metadata": {
                        "category": row["Category"],
                        "subset": row["Subset"],
                        "language": self.lang,
                    },
                },
                num_proc=self.num_workers,
                remove_columns=[
                    "ID",
                    "image",
                    "Label",
                    "Question",
                    "Options",
                    "Translated Question",
                    "Translated Options",
                    "Image Type",
                    "Image Source",
                    "License",
                    "Category",
                    "Subset",
                ],
            )

            self.dataset = dataset
