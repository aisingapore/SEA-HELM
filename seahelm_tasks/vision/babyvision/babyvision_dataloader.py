from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader

logger = get_logger(__name__)


class BabyVisionDataloader(HuggingFaceImageDataloader, JudgeDataloader):
    """Dataloader for the BabyVision dataset.

    BabyVision is a dataset for evaluating vision models on various tasks.
    This dataloader handles loading data and preparing it for model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the BabyVision dataset from HuggingFace datasets.

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

            dataset = dataset.cast_column("image", Image(decode=False))

            def map_columns(row, index):
                if row["ansType"] == "choice":
                    choices = "\n" + "\n".join(
                        f"{i}: {choice}"
                        for i, choice in zip("ABCDEF", row["options"], strict=False)
                    )
                    answer = "ABCDEF"[row["choiceAns"]]
                elif row["ansType"] == "blank":
                    choices = ""
                    answer = row["blankAns"]
                else:
                    raise ValueError(f"Unknown ansType: {row['ansType']}")

                return {
                    "id": row["taskId"],
                    "prompts": [{"question": row["question"], "choices": choices}],
                    "images": [row["image"]["bytes"]],
                    "label": answer,
                    "metadata": {
                        "language": "en",
                        "type": row["type"],
                        "subtype": row["subtype"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def get_judge_prompt_formatter(self, metric):
        judge_prompt = self.task_config.config.judge["judge_prompts"]

        def _prompt_formatter(row, idx):
            responses = row["responses"]

            conversations = []
            custom_ids = []

            # Generate judgements
            messages = [
                {
                    "role": "user",
                    "content": judge_prompt["criteria_prompt"].format(
                        question=row["prompts"][0]["question"],
                        groundtruth=row["label"],
                        modeloutput=metric.extract_response(responses),
                    ),
                },
            ]
            # Add messages and custom IDs to lists
            conversations.append(messages)
            custom_ids.append(f"{row['id']}")
            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter
