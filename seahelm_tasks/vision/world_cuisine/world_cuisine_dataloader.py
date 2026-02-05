import numpy as np
from datasets import Image, concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class WorldCuisineDataloader(HuggingFaceImageDataloader):
    """Dataloader for World Cuisine visual multiple-choice QA tasks.

    Loads multilingual cuisine-related MCQ questions with 5 options (A-E) and
    associated image URLs. Handles language code normalization, task variant
    selection (task1/task2), prompt type filtering and shuffling the correct
    answer position before projecting to a unified schema for evaluation.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ) -> None:
        """Initialize the WorldCuisineDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples.
            is_base_model: Whether the model is a base (non-instruction) model.
            model_name: Name/path of model under evaluation.
            run_base_path: Base path for output artifacts.
            inference_file_type: Output file format ("jsonl" or "csv").
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )

        self.rng = np.random.default_rng(seed=3722236)

    def load_dataset(self, limit: int = None) -> None:
        """Load the World Cuisine dataset and project to evaluation schema.

        Loads the ``test_large`` split for the derived task variant, filters by
        language, keeps allowed prompt types (task1: 1/3/4; task2: 2), shuffles
        the correct answer position among 5 options and converts each record to
        a unified structure with id, label (A-E), prompts, images and metadata.

        Args:
            limit: Optional cap on number of raw rows (before prompt-type merge).

        Note:
            Result stored in ``self.dataset``. Answer indices are re-shuffled
            deterministically via a fixed RNG seed.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            return

        # Normalize language codes expected by the underlying dataset.
        if self.lang == "id":
            lang = "id_formal"
        elif self.lang == "zh":
            lang = "zh_cn"
        elif self.lang == "jv":
            lang = "jv_krama"
        elif self.lang == "su":
            lang = "su_loma"
        else:
            lang = self.lang

        task = self.task_name.removeprefix("world_cuisine_")

        # assumes images.tar.gz has already been downloaded and extracted
        cache_path = snapshot_download(
            repo_id=self.specific_task_config["filepath"], repo_type="dataset"
        )
        dataset = load_dataset(
            self.specific_task_config["filepath"], task, split="test_large"
        )
        dataset = dataset.filter(
            lambda example: example["lang"] == lang,
            num_proc=self.num_workers,
        )
        if limit is not None:
            dataset = dataset.select(range(limit))

        # Determine which prompt types to include per task variant.
        if task == "task1":
            prompt_index_list = [1, 3, 4]
        elif task == "task2":
            prompt_index_list = [2]
        else:
            prompt_index_list = []  # Defensive fallback; should not occur.

        new_dataset = []

        for prompt_type in prompt_index_list:
            subset = dataset.filter(
                lambda example: example["prompt_type"] == prompt_type,
                num_proc=self.num_workers,
            )
            # Create a reproducible shuffled list of target insertion indices
            # for the correct answer among 5 options.
            re_sort_index = [0, 1, 2, 3, 4] * (len(subset) // 5)
            self.rng.shuffle(re_sort_index)

            subset = subset.add_column("new_index", re_sort_index)

            def re_sort_options(example):
                idx = example["new_index"]
                options = example["options"]
                correct_answer = options.pop(example["mcq_answer_index"])
                options.insert(idx, correct_answer)
                example["mcq_answer_index"] = idx
                example["options"] = options
                return example

            subset = subset.map(
                re_sort_options,
                num_proc=self.num_workers,
                remove_columns=["new_index"],
            )
            new_dataset.append(subset)

        dataset = concatenate_datasets(new_dataset) if new_dataset else dataset
        dataset = dataset.map(
            lambda row: {"image_path": f"{cache_path}/{row['image_path']}"},
            num_proc=self.num_workers,
        )
        dataset = dataset.cast_column("image_path", Image(decode=True))

        dataset = dataset.map(
            lambda row: {
                "id": row["qa_id"],
                "label": "ABCDE"[row["mcq_answer_index"]],
                "prompts": [
                    {
                        "question": row["question"],
                        "option_a": row["options"][0],
                        "option_b": row["options"][1],
                        "option_c": row["options"][2],
                        "option_d": row["options"][3],
                        "option_e": row["options"][4],
                    }
                ],
                "images": [row["image_path"]],
                "metadata": {
                    "language": self.lang,
                    "food_id": row["food_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt_type": row["prompt_type"],
                    "task": row["task"],
                },
            },
            num_proc=self.num_workers,
            remove_columns=[
                "qa_id",
                "lang",
                "food_id",
                "prompt_id",
                "prompt_type",
                "image_url",
                "task",
                "lang_status",
                "image_path",
                "question",
                "options",
                "mcq_answer_index",
                "answer",
            ],
        )

        self.dataset = dataset
