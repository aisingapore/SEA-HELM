from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader

logger = get_logger(__name__)


class MathVistaDataloader(HuggingFaceImageDataloader):
    """Dataloader for the MathVista mathematical reasoning with visual context dataset.

    MathVista is a benchmark for evaluating mathematical reasoning capabilities
    of vision-language models on problems that require understanding both visual
    and textual information. The dataset includes both multiple-choice and
    free-form mathematical questions with associated images.

    Supports both 'test' and 'testmini' splits, and can filter for either
    'multi_choice' or 'free_form' question types based on the task configuration.
    """

    def load_dataset(self, limit: int = None):
        """Load the MathVista dataset from HuggingFace datasets.

        Loads the dataset from the configured filepath, optionally limiting the number
        of samples. Automatically selects the appropriate split ('test' or 'testmini')
        based on the task name and filters for the specified question type
        ('multi_choice' or 'free_form').

        For multiple-choice questions, formats the choices with letter labels (A, B, C, etc.)
        and creates a 'choices_text' field with formatted options and a 'label' field
        with the correct answer letter.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and processing.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            if "mini" in self.task_name:
                split = "testmini"
            else:
                split = "test"

            dataset = load_dataset(self.specific_task_config["filepath"], split=split)
            dataset = dataset.cast_column("decoded_image", Image(decode=False))

            if "free_form" in self.task_name:
                dataset = dataset.filter(
                    lambda x: x["question_type"] == "free_form",
                    num_proc=self.num_workers,
                )
                if limit is not None:
                    dataset = dataset.select(range(limit))

                dataset = dataset.map(
                    lambda row: {
                        "id": row["pid"],
                        "label": row["answer"],
                        "prompts": [
                            {
                                "question": row["question"]
                                + (f" (Unit: {row['unit']})" if row["unit"] else ""),
                            }
                        ],
                        "images": [row["decoded_image"]["bytes"]],
                        "metadata": {
                            "question_type": row["question_type"],
                            "unit": row["unit"],
                            "precision": row["precision"],
                            "answer_type": row["answer_type"],
                            **row["metadata"],
                        },
                    },
                    num_proc=self.num_workers,
                    remove_columns=[
                        "pid",
                        "question",
                        "answer",
                        "question_type",
                        "unit",
                        "precision",
                        "answer_type",
                        "image",
                        "decoded_image",
                        "choices",
                        "query",
                    ],
                )
            else:
                dataset = dataset.filter(
                    lambda x: x["question_type"] == "multi_choice",
                    num_proc=self.num_workers,
                )
                if limit is not None:
                    dataset = dataset.select(range(limit))

                def prepare_choices(row):
                    row["id"] = row["pid"]
                    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    answer_index = row["choices"].index(row["answer"])
                    row["label"] = letters[answer_index]
                    row["prompts"] = [
                        {
                            "question": row["question"],
                            "choices_text": "\n".join(
                                f"({letter}) {choice}"
                                for letter, choice in zip(
                                    letters, row["choices"], strict=False
                                )
                            ),
                        }
                    ]
                    row["images"] = [row["decoded_image"]["bytes"]]
                    row["metadata"] = {
                        "question_type": row["question_type"],
                        "unit": row["unit"],
                        "precision": row["precision"],
                        "answer_type": row["answer_type"],
                        **row["metadata"],
                    }
                    return row

                dataset = dataset.map(
                    prepare_choices,
                    num_proc=self.num_workers,
                    remove_columns=[
                        "pid",
                        "question",
                        "answer",
                        "question_type",
                        "unit",
                        "precision",
                        "answer_type",
                        "image",
                        "decoded_image",
                        "choices",  # Remove choices for free-form questions
                        "query",
                    ],
                )

            self.dataset = dataset
